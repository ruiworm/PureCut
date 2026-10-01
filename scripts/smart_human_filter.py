#!/usr/bin/env python3
"""
PureCut Local Smart Human Filter & Product B-Roll Splitter
============================================================
Accurately detects human presence (faces + bodies) in e-commerce/UGC videos,
and extracts clean, person-free product demonstration segments (B-Roll).
"""

import sys
import os
import argparse
import json
import subprocess
import cv2
import numpy as np

# Suppress TensorFlow Lite / MediaPipe noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FFMPEG_BIN = os.environ.get("FFMPEG_PATH", "ffmpeg")


class SubtitleDetector:
    """
    Ultra-lightweight subtitle detection using Paddle PP-OCRv4-det ONNX (~4.6MB).
    Runs CPU inference in ~10ms per frame.
    """
    def __init__(self, models_dir=None):
        if not models_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models", "vision")
        self.model_path = os.path.join(models_dir, "ch_PP-OCRv4_det_infer.onnx")
        self.session = None

    def _ensure_session(self):
        if self.session is None and os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
            except Exception as e:
                print(f"[SubtitleDetector] Warning: failed to load ONNX model: {e}", file=sys.stderr)

    def detect_in_frame(self, frame_bgr, lower_ratio=0.35, thresh=0.3):
        self._ensure_session()
        if not self.session:
            return [], None

        h, w = frame_bgr.shape[:2]
        crop_top = int(h * (1.0 - lower_ratio))
        sub_roi = frame_bgr[crop_top:h, 0:w]
        rh, rw = sub_roi.shape[:2]
        if rh == 0 or rw == 0:
            return [], None

        target_h = int(np.ceil(rh / 32) * 32)
        target_w = int(np.ceil(rw / 32) * 32)
        resized = cv2.resize(sub_roi, (target_w, target_h))

        inp = (resized.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...]

        try:
            preds = self.session.run(None, {self.input_name: inp})[0][0, 0]
        except Exception:
            return [], None

        bin_mask = (preds > thresh).astype(np.uint8) * 255
        bin_mask_roi = cv2.resize(bin_mask, (rw, rh), interpolation=cv2.INTER_NEAREST)

        contours, _ = cv2.findContours(bin_mask_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw > 15 and bh > 6:
                boxes.append((bx, crop_top + by, bw, bh))

        full_mask = None
        if len(boxes) > 0:
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[crop_top:h, 0:w] = bin_mask_roi
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            full_mask = cv2.dilate(full_mask, kernel, iterations=2)

        return boxes, full_mask


class SmartHumanFilter:
    def __init__(self, models_dir=None, face_thresh=0.40, person_thresh=0.35, min_area_ratio=0.015, filter_mode="keep_hands"):
        if not models_dir:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, "models", "vision")

        self.models_dir = models_dir
        self.subtitle_detector = SubtitleDetector(models_dir=models_dir)
        self.filter_mode = filter_mode

        face_model_path = os.path.join(models_dir, "blaze_face.tflite")
        obj_model_path = os.path.join(models_dir, "efficientdet_lite0.tflite")

        if not os.path.exists(face_model_path) or not os.path.exists(obj_model_path):
            raise FileNotFoundError(f"Model files not found in {models_dir}")

        # Initialize Face Detector
        face_options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=face_model_path),
            min_detection_confidence=face_thresh
        )
        self.face_detector = vision.FaceDetector.create_from_options(face_options)

        # Initialize Object Detector (for 'person' category)
        obj_options = vision.ObjectDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=obj_model_path),
            score_threshold=person_thresh
        )
        self.obj_detector = vision.ObjectDetector.create_from_options(obj_options)

        self.min_area_ratio = min_area_ratio

    def check_frame_has_human(self, frame_bgr, filter_mode=None):
        mode = filter_mode or self.filter_mode
        h, w = frame_bgr.shape[:2]
        frame_area = w * h
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # 1. Check Face Detection
        face_res = self.face_detector.detect(mp_image)
        if face_res.detections:
            for d in face_res.detections:
                score = d.categories[0].score if d.categories else 0.0
                bbox = d.bounding_box
                area = bbox.width * bbox.height
                
                if mode == "keep_hands":
                    # In keep_hands mode, use calibrated threshold (0.58) to eliminate palm/knuckle false positives
                    if score >= 0.58 and (area / frame_area >= 0.005):
                        return True, "face"
                else:
                    if area / frame_area >= 0.005:
                        return True, "face"

        # 2. Check Person Body Detection
        obj_res = self.obj_detector.detect(mp_image)
        if obj_res.detections:
            for d in obj_res.detections:
                for cat in d.categories:
                    if cat.category_name.lower() == "person":
                        bbox = d.bounding_box
                        area = bbox.width * bbox.height
                        
                        if mode == "keep_hands":
                            # Only filter out real human upper body / standing / sitting model (origin_y near top of frame)
                            # Hand close-ups usually center around the product with origin_y >= 0.20 or lower score
                            origin_y_ratio = bbox.origin_y / h
                            height_ratio = bbox.height / h
                            if cat.score >= 0.82 and origin_y_ratio < 0.12 and height_ratio > 0.60:
                                return True, "body"
                        else:
                            # Strict: any person bounding box (including hand close-ups)
                            if area / frame_area >= self.min_area_ratio:
                                return True, "body"

        return False, None

    def analyze_video(self, video_path, sample_fps=2.5, filter_mode=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / orig_fps

        frame_step = max(1, int(round(orig_fps / sample_fps)))
        current_frame = 0

        human_timestamps = []

        while current_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            sec = current_frame / orig_fps
            has_human, reason = self.check_frame_has_human(frame, filter_mode=filter_mode)
            if has_human:
                human_timestamps.append(sec)

            current_frame += frame_step

        cap.release()

        return duration, human_timestamps, sample_fps

    def compute_intervals(self, total_duration, human_timestamps, sample_fps, margin=0.35, min_clean_dur=1.5):
        if not human_timestamps:
            return [], [[0.0, total_duration]]

        step = 1.0 / sample_fps
        raw_intervals = []
        for t in human_timestamps:
            raw_intervals.append([max(0.0, t - margin), min(total_duration, t + step + margin)])

        # Merge overlapping human intervals
        raw_intervals.sort(key=lambda x: x[0])
        merged_human = []
        for interval in raw_intervals:
            if not merged_human:
                merged_human.append(interval)
            else:
                last = merged_human[-1]
                if interval[0] <= last[1]:
                    last[1] = max(last[1], interval[1])
                else:
                    merged_human.append(interval)

        # Invert to get clean intervals
        clean_intervals = []
        cursor = 0.0
        for h_start, h_end in merged_human:
            if h_start - cursor >= min_clean_dur:
                clean_intervals.append([round(cursor, 3), round(h_start, 3)])
            cursor = max(cursor, h_end)

        if total_duration - cursor >= min_clean_dur:
            clean_intervals.append([round(cursor, 3), round(total_duration, 3)])

        return merged_human, clean_intervals

    def _detect_segment_subtitles(self, input_path, start, end, sample_interval=0.35):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = int(start * fps)
        end_frame = int(end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        step = max(1, int(fps * sample_interval))
        cur_frame = start_frame

        detected_frames = []
        boxes_collected = []

        while cur_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            boxes, _ = self.subtitle_detector.detect_in_frame(frame, lower_ratio=0.50)
            rel_t = (cur_frame - start_frame) / fps

            # Filter true subtitle lines: wide, horizontal, and centered
            valid_boxes = []
            for b in boxes:
                bx, by, bw, bh = b
                aspect = bw / max(1, bh)
                cx = bx + bw / 2
                if aspect >= 2.8 and bw >= 0.10 * w and abs(cx - w / 2) < 0.30 * w:
                    valid_boxes.append(b)

            if valid_boxes:
                detected_frames.append(rel_t)
                boxes_collected.extend(valid_boxes)
            cur_frame += step
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)

        cap.release()

        if not detected_frames or not boxes_collected:
            return [], None

        intervals = []
        c_start = detected_frames[0]
        c_end = detected_frames[0] + sample_interval
        for t in detected_frames[1:]:
            if t <= c_end + sample_interval:
                c_end = t + sample_interval
            else:
                intervals.append((max(0.0, c_start - 0.15), min(end - start, c_end + 0.15)))
                c_start = t
                c_end = t + sample_interval
        intervals.append((max(0.0, c_start - 0.15), min(end - start, c_end + 0.15)))

        # Cluster by median Y to lock onto the subtitle line and ignore bottom/droplet noise
        med_y = np.median([b[1] for b in boxes_collected])
        line_boxes = [b for b in boxes_collected if abs(b[1] - med_y) < 60]
        if not line_boxes:
            line_boxes = boxes_collected

        # Add generous padding so subtitle tops/bottoms are fully enclosed and never cut
        min_x = max(0, min(b[0] for b in line_boxes) - 30)
        min_y = max(0, min(b[1] for b in line_boxes) - 25)
        max_x = min(w, max(b[0] + b[2] for b in line_boxes) + 30)
        max_y = min(h, max(b[1] + b[3] for b in line_boxes) + 25)
        union_box = (min_x, min_y, max_x - min_x, max_y - min_y)

        return intervals, union_box

    def _inpaint_segment(self, input_path, start, end, out_path):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = int(start * fps)
        end_frame = int(end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        ffmpeg_cmd = [
            FFMPEG_BIN, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", f"{fps}",
            "-i", "-",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", input_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "19",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            out_path
        ]
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        cur = start_frame
        while cur < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            boxes, mask = self.subtitle_detector.detect_in_frame(frame, lower_ratio=0.50)
            valid_boxes = [b for b in boxes if (b[2] / max(1, b[3])) >= 2.5 and b[2] >= 0.10 * w and abs((b[0] + b[2]/2) - w/2) < 0.30 * w]
            if valid_boxes and mask is not None:
                text_mask = np.zeros_like(mask)
                for bx, by, bw, bh in valid_boxes:
                    x1 = max(0, bx - 10)
                    y1 = max(0, by - 10)
                    x2 = min(w, bx + bw + 10)
                    y2 = min(h, by + bh + 10)
                    text_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
                if np.count_nonzero(text_mask) > 30:
                    frame = cv2.inpaint(frame, text_mask, 3, cv2.INPAINT_TELEA)
            try:
                proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                break
            cur += 1

        cap.release()
        try:
            proc.stdin.close()
        except OSError:
            pass
        proc.wait()

    def cut_clean_segments(self, input_path, clean_intervals, output_dir, subtitle_mode="none", sub_height_ratio=0.16):
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_files = []

        sub_height_ratio = max(0.08, min(0.30, sub_height_ratio))
        top_offset = 1.0 - sub_height_ratio

        for idx, (start, end) in enumerate(clean_intervals, start=1):
            dur = end - start
            out_filename = f"{base_name}_clean_{idx:02d}.mp4"
            out_path = os.path.join(output_dir, out_filename)

            if subtitle_mode == "ai_inpaint":
                self._inpaint_segment(input_path, start, end, out_path)
                if os.path.exists(out_path):
                    output_files.append({
                        "index": idx,
                        "filename": out_filename,
                        "path": out_path,
                        "start": start,
                        "end": end,
                        "duration": round(dur, 3),
                        "subtitle_mode": subtitle_mode
                    })
                continue

            if subtitle_mode == "none" or not subtitle_mode:
                # Ultra-fast lossless stream copy (takes ~0.05s per segment)
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-ss", f"{start:.3f}",
                    "-to", f"{end:.3f}",
                    "-i", input_path,
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    out_path
                ]
            else:
                cmd = [
                    FFMPEG_BIN, "-y",
                    "-ss", f"{start:.3f}",
                    "-to", f"{end:.3f}",
                    "-i", input_path,
                ]

                # Apply subtitle removal and aspect ratio framing filters if specified
                if subtitle_mode == "blur":
                    # Smooth boxblur on the bottom subtitle band
                    vf = f"[0:v]split[v1][v2];[v2]crop=iw:ih*{sub_height_ratio:.3f}:0:ih*{top_offset:.3f},boxblur=15:15[blur];[v1][blur]overlay=0:H*{top_offset:.3f}"
                    cmd += ["-vf", vf]
                elif subtitle_mode == "crop":
                    # Crop off bottom subtitle region and scale back to original resolution
                    vf = f"crop=iw:ih*{top_offset:.3f}:0:0,scale=iw:ih:flags=lanczos"
                    cmd += ["-vf", vf]
                elif subtitle_mode == "crop_3_4":
                    # Crop directly to 3:4 aspect ratio with bottom cut priority (eliminating bottom subtitles)
                    vf = "crop=w='trunc(min(iw,ih*3/4)/2)*2':h='trunc(min(ih,iw*4/3)/2)*2':x='(iw-out_w)/2':y='(ih-out_h)*0.15',setsar=1"
                    cmd += ["-vf", vf]
                elif subtitle_mode == "pad_3_4":
                    # 3:4 aspect ratio with frosted glass background padding
                    vf = "[0:v]scale=1080:1440,boxblur=25:25,setsar=1[bg];[0:v]scale=1080:1440:force_original_aspect_ratio=decrease,setsar=1[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
                    cmd += ["-filter_complex", vf]
                elif subtitle_mode == "ai_dynamic_blur":
                    # AI dynamic subtitle detection + dynamic conditional boxblur
                    intervals, union_box = self._detect_segment_subtitles(input_path, start, end)
                    if intervals and union_box:
                        bx, by, bw, bh = union_box
                        bw = int(bw // 2 * 2)
                        bh = int(bh // 2 * 2)
                        conds = " + ".join([f"between(t,{s:.2f},{e:.2f})" for s, e in intervals])
                        vf = f"[0:v]split[v1][v2];[v2]crop={bw}:{bh}:{bx}:{by},boxblur=15:15[blur];[v1][blur]overlay={bx}:{by}:enable='{conds}'"
                        cmd += ["-vf", vf]

                cmd += [
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-threads", "0",
                    "-crf", "20",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-avoid_negative_ts", "make_zero",
                    out_path
                ]
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0 and os.path.exists(out_path):
                output_files.append({
                    "index": idx,
                    "filename": out_filename,
                    "path": out_path,
                    "start": start,
                    "end": end,
                    "duration": round(dur, 3),
                    "subtitle_mode": subtitle_mode
                })

        return output_files


def main():
    parser = argparse.ArgumentParser(description="PureCut Local Human Filter & Video Splitter")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output-dir", "-o", default="", help="Output directory for clean clips")
    parser.add_argument("--sample-fps", type=float, default=2.5, help="Sampling FPS for person detection")
    parser.add_argument("--min-duration", type=float, default=1.5, help="Minimum clean clip duration in seconds")
    parser.add_argument("--margin", type=float, default=0.35, help="Margin around detected human presence (seconds)")
    parser.add_argument("--face-threshold", type=float, default=0.40, help="Face detection confidence threshold")
    parser.add_argument("--person-threshold", type=float, default=0.35, help="Person body detection confidence threshold")
    parser.add_argument("--filter-mode", choices=["keep_hands", "strict_all"], default="keep_hands", help="Human filter mode: keep_hands (retain hand closeups) or strict_all (strict no human)")
    parser.add_argument("--subtitle-mode", choices=["none", "blur", "crop", "crop_3_4", "pad_3_4", "ai_dynamic_blur", "ai_inpaint"], default="none", help="Subtitle removal and framing mode")
    parser.add_argument("--subtitle-height", type=float, default=0.16, help="Bottom subtitle region height ratio (default: 0.16)")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without cutting clips")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(json.dumps({"error": f"Input video not found: {args.input}"}, ensure_ascii=False))
        sys.exit(1)

    out_dir = args.output_dir
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(args.input), "clean_broll")

    try:
        filter_engine = SmartHumanFilter(
            face_thresh=args.face_threshold,
            person_thresh=args.person_threshold,
            filter_mode=args.filter_mode
        )
        total_duration, human_ts, sample_fps = filter_engine.analyze_video(args.input, sample_fps=args.sample_fps, filter_mode=args.filter_mode)
        human_intervals, clean_intervals = filter_engine.compute_intervals(
            total_duration, human_ts, sample_fps, margin=args.margin, min_clean_dur=args.min_duration
        )

        clean_clips = []
        if not args.dry_run and clean_intervals:
            clean_clips = filter_engine.cut_clean_segments(
                args.input,
                clean_intervals,
                out_dir,
                subtitle_mode=args.subtitle_mode,
                sub_height_ratio=args.subtitle_height
            )

        result = {
            "success": True,
            "input_file": args.input,
            "total_duration": round(total_duration, 3),
            "has_human": len(human_intervals) > 0,
            "human_segments": [{"start": round(s, 3), "end": round(e, 3)} for s, e in human_intervals],
            "clean_segments": clean_intervals,
            "clean_clips": clean_clips,
            "output_dir": out_dir
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
