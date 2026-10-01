#!/usr/bin/env python3
"""
PureCut Video Fission & Auto-Multiplexing Engine
=================================================
Automated batch video multiplexing for e-commerce short videos:
1. Auto TTS voiceover synthesis
2. Auto natural sentence splitting & ASS burn-in subtitle generation
3. Anti-homogeneity random clip assembly & duration matching
4. Final video mastering with FFmpeg
"""

import sys
import os
import re
import random
import subprocess
import asyncio
import uuid
from pathlib import Path


def get_media_duration(file_path: str) -> float:
    """Gets audio or video duration in seconds via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception as e:
        print(f"[get_media_duration] Error probing {file_path}: {e}", file=sys.stderr)
        return 0.0


def get_video_dimensions(file_path: str) -> tuple[int, int]:
    """Gets video width and height via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = res.stdout.strip().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception as e:
        print(f"[get_video_dimensions] Error probing {file_path}: {e}", file=sys.stderr)
    return 1080, 1920


def split_text_to_subtitles(text: str) -> list[str]:
    """
    Splits continuous e-commerce copy into short, readable subtitle phrases
    (typically 5 to 14 characters per line).
    """
    # Replace newlines and excessive whitespace
    clean = re.sub(r'\s+', ' ', text.strip())
    # Split by common Chinese and English punctuations
    raw_parts = re.split(r'([，。！？；,!?;~\n]+)', clean)
    
    sentences = []
    current = ""
    for part in raw_parts:
        if not part:
            continue
        if re.match(r'^[，。！？；,!?;~\n]+$', part):
            current += part
            if len(current.strip()) > 0:
                sentences.append(current.strip())
                current = ""
        else:
            current += part

    if current.strip():
        sentences.append(current.strip())

    # Further break long sentences if > 16 chars
    final_sentences = []
    for s in sentences:
        s_clean = re.sub(r'[，。！？；,!?;~]+$', '', s).strip()
        if not s_clean:
            continue
        if len(s_clean) > 16:
            # Cut at middle or comma
            mid = len(s_clean) // 2
            final_sentences.append(s_clean[:mid])
            final_sentences.append(s_clean[mid:])
        else:
            final_sentences.append(s_clean)

    return final_sentences if final_sentences else [text[:20]]


def format_ass_time(seconds: float) -> str:
    """Formats float seconds into ASS timestamp: H:MM:SS.cs (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_ass_file(
    sentences: list[str],
    total_duration: float,
    out_ass_path: str,
    aspect_ratio: str = "9:16",
    color_scheme: str = "yellow",
    font_size: int = None,
    margin_v: int = None,
    font_name: str = "Microsoft YaHei",
    position_v: float = None
) -> str:
    """
    Generates an e-commerce high-conversion ASS subtitle script.
    - Large bold font (Microsoft YaHei / SimHei)
    - High-contrast outline & drop shadow or card box
    - Bottom/Center/Custom position in the safe zone
    """
    if aspect_ratio == "3:4":
        res_x = 1080
        res_y = 1440
        default_font_size = 52
        default_margin_v = 180
    elif aspect_ratio == "16:9":
        res_x = 1920
        res_y = 1080
        default_font_size = 52
        default_margin_v = 120
    else:  # 9:16 default
        res_x = 1080
        res_y = 1920
        default_font_size = 58
        default_margin_v = 320

    active_font_size = font_size if (font_size and font_size > 10) else default_font_size

    if position_v is not None and 0.0 <= position_v <= 1.0:
        active_margin_v = int(res_y * max(0.04, min(0.92, 1.0 - position_v)))
    elif margin_v is not None:
        active_margin_v = margin_v
    else:
        active_margin_v = default_margin_v

    # Colors in ASS BGR hex: &H00BBGGRR
    border_style = 1
    outline_val = 4.5
    shadow_val = 1.5

    if color_scheme == "white":
        primary_color = "&H00FFFFFF"
        outline_color = "&H00000000"
        back_color = "&H80000000"
        outline_val = 3.5
    elif color_scheme == "cyan":
        primary_color = "&H00FFFF00"
        outline_color = "&H00000000"
        back_color = "&H80000000"
        outline_val = 4.0
    elif color_scheme == "black_gold_card":
        primary_color = "&H0000FFFF"
        outline_color = "&H90151515"
        back_color = "&HB0000000"
        border_style = 3  # Opaque background card box
        outline_val = 8.0
        shadow_val = 0.0
    else:  # yellow default
        primary_color = "&H0000FFFF"
        outline_color = "&H00000000"
        back_color = "&H80000000"
        outline_val = 4.5

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{active_font_size},{primary_color},&H000000FF,{outline_color},{back_color},-1,0,0,0,100,100,0,0,{border_style},{outline_val},{shadow_val},2,40,40,{active_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    total_chars = max(1, sum(len(re.sub(r'\s+', '', s)) for s in sentences))
    # Small lead-in margin
    lead_in = 0.15
    usable_dur = max(0.5, total_duration - lead_in)
    
    events = []
    cursor = lead_in

    for i, s in enumerate(sentences):
        char_count = max(1, len(re.sub(r'\s+', '', s)))
        seg_dur = usable_dur * (char_count / total_chars)
        
        # Don't let single sentence be under 0.8s unless it's 1-2 words
        seg_dur = max(0.8, seg_dur)
        
        start_t = cursor
        end_t = min(total_duration, cursor + seg_dur)
        if i == len(sentences) - 1:
            end_t = total_duration

        start_str = format_ass_time(start_t)
        end_str = format_ass_time(end_t)
        
        events.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{s}")
        cursor = end_t

    with open(out_ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")

    return out_ass_path


async def synthesize_edge_tts(text: str, voice: str, speed: float, out_mp3_path: str):
    """Synthesize voice using edge-tts."""
    import edge_tts
    speed_pct = int(round((speed - 1.0) * 100))
    rate_str = f"{speed_pct:+d}%"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
    await communicate.save(out_mp3_path)


def run_coroutine_sync(coro):
    """Safely runs an async coroutine from synchronous code, even inside a running event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(coro))
            return future.result()
    else:
        return asyncio.run(coro)


def synthesize_voice(
    text: str,
    voice: str = "zh-CN-XiaoxiaoNeural",
    speed: float = 1.05,
    out_mp3_path: str = None,
    base_url: str = None,
    api_key: str = None,
    model: str = None
) -> str:
    """
    Synthesizes TTS audio from text, supporting Edge-TTS and OpenAI-compatible endpoints.
    """
    if not out_mp3_path:
        out_mp3_path = os.path.abspath(f"tts_{uuid.uuid4().hex[:8]}.mp3")

    # Clean text numbers and punctuation for clean speech
    clean_text = text.replace("¥", "人民币").replace("&", "和").strip()

    if not voice or "edge-tts" in str(base_url) or voice.startswith("zh-CN-"):
        # Edge TTS
        edge_voice = voice if voice.startswith("zh-CN-") else "zh-CN-XiaoxiaoNeural"
        run_coroutine_sync(synthesize_edge_tts(clean_text, edge_voice, speed, out_mp3_path))
    else:
        # OpenAI compatible endpoint
        import requests
        headers = {"Content-Type": "application/json"}
        if api_key and api_key.strip():
            headers["Authorization"] = f"Bearer {api_key.strip()}"
        elif "127.0.0.1" in str(base_url) or "localhost" in str(base_url):
            headers["Authorization"] = "Bearer local-offline"

        url = f"{base_url.rstrip('/')}/audio/speech"
        payload = {
            "model": model or "doubao-tts",
            "input": clean_text,
            "voice": voice,
            "response_format": "mp3",
            "speed": speed
        }
        res = requests.post(url, headers=headers, json=payload, timeout=40)
        if not res.ok:
            raise ValueError(f"TTS request failed ({res.status_code}): {res.text[:200]}")
        with open(out_mp3_path, "wb") as f:
            f.write(res.content)

    return out_mp3_path


def build_fission_video(
    clip_paths: list[str],
    audio_path: str,
    ass_path: str,
    out_video_path: str,
    aspect_ratio: str = "9:16",
    anti_repeat: bool = True,
    max_scene_dur: float = 3.2
) -> str:
    """
    Assembles a complete, ready-to-publish e-commerce video:
    1. Selects and shuffles clean clips to match voiceover length
    2. Takes dynamic fast-paced scene cuts (2.0~3.5s per scene)
    3. Scales and crops to target aspect ratio (9:16 or 3:4)
    4. Concatenates video tracks
    5. Mixes in TTS voiceover audio
    6. Burns in ASS stylized subtitles
    """
    if not clip_paths:
        raise ValueError("No video clips provided for fission assembly!")

    target_duration = get_media_duration(audio_path)
    if target_duration <= 0:
        target_duration = 10.0

    # Probe duration of available clips
    clip_durations = []
    for p in clip_paths:
        dur = get_media_duration(p)
        if dur > 0.8:
            clip_durations.append((p, dur))

    if not clip_durations:
        raise ValueError("Provided clips have invalid durations or are inaccessible.")

    # Select sub-clips to cover target duration with fast-paced cuts
    shuffled_pool = list(clip_durations)
    random.shuffle(shuffled_pool)

    scene_specs = []  # list of (path, start_offset, take_duration)
    accumulated_dur = 0.0

    while accumulated_dur < target_duration + 0.5:
        for p, dur in shuffled_pool:
            # Pick a dynamic scene duration between 2.0s and min(dur, max_scene_dur)
            take_dur = min(dur, random.uniform(2.0, max_scene_dur))
            if dur > take_dur + 0.5:
                start_offset = random.uniform(0.0, dur - take_dur)
            else:
                start_offset = 0.0
                take_dur = dur

            scene_specs.append((p, start_offset, take_dur))
            accumulated_dur += take_dur
            if accumulated_dur >= target_duration + 0.5:
                break
        random.shuffle(shuffled_pool)

    # Resolution target
    if aspect_ratio == "3:4":
        target_w, target_h = 1080, 1440
    else:  # 9:16
        target_w, target_h = 1080, 1920

    work_dir = os.path.dirname(os.path.abspath(out_video_path))
    os.makedirs(work_dir, exist_ok=True)

    # Relative ASS path inside work_dir prevents Windows path escaping issues in FFmpeg
    ass_filename = os.path.basename(ass_path)
    if os.path.dirname(os.path.abspath(ass_path)) != work_dir:
        import shutil
        local_ass = os.path.join(work_dir, ass_filename)
        shutil.copyfile(ass_path, local_ass)
        ass_filter_arg = f"ass={ass_filename}"
    else:
        ass_filter_arg = f"ass={ass_filename}"

    # Build FFmpeg command inputs
    cmd = ["ffmpeg", "-y"]
    for p, start_offset, take_dur in scene_specs:
        cmd.extend(["-ss", f"{start_offset:.2f}", "-t", f"{take_dur:.2f}", "-i", p])

    cmd.extend(["-i", audio_path])  # Audio is index len(scene_specs)
    audio_idx = len(scene_specs)

    # Build filter_complex
    filters = []
    for i in range(len(scene_specs)):
        speed_var = random.uniform(0.97, 1.03) if anti_repeat else 1.0
        pts_filter = f"setpts={1.0/speed_var:.4f}*PTS," if anti_repeat else ""
        hflip = "hflip," if (anti_repeat and random.random() < 0.25) else ""
        
        f_str = (
            f"[{i}:v]"
            f"{pts_filter}{hflip}"
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{target_h},"
            f"setsar=1,fps=30"
            f"[v{i}]"
        )
        filters.append(f_str)

    concat_inputs = "".join(f"[v{i}]" for i in range(len(scene_specs)))
    filters.append(f"{concat_inputs}concat=n={len(scene_specs)}:v=1:a=0[vconcat]")
    filters.append(f"[vconcat]trim=duration={target_duration:.3f},{ass_filter_arg}[vout]")

    filter_complex_str = ";".join(filters)

    cmd.extend([
        "-filter_complex", filter_complex_str,
        "-map", "[vout]",
        "-map", f"{audio_idx}:a",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-threads", "0",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", f"{target_duration:.3f}",
        out_video_path
    ])

    print(f"[build_fission_video] Assembling {len(scene_specs)} scenes for {target_duration:.1f}s video...")
    res = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if res.returncode != 0:
        err_tail = res.stderr[-800:] if res.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg assembly failed: {err_tail}")

    return out_video_path
