#!/usr/bin/env python3
"""
PureCut Web Testing Studio (本地 Web 测试工作台)
==================================================
用于直观测试与验证两大核心功能：
1. 本地人像识别与商品片段切分（Timeline 可视化、原片 vs 纯商品对比播放、一键切分）
2. 成熟电商带货 TTS 配音试听（火山引擎豆包、硅基流动 CosyVoice2、MiniMax、本地离线端点）
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import uuid
import shutil
import tempfile
import requests
import subprocess
import hashlib
import urllib.parse
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Add current repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.smart_human_filter import SmartHumanFilter
from scripts.fission_engine import (
    get_media_duration,
    get_video_dimensions,
    split_text_to_subtitles,
    generate_ass_file,
    synthesize_voice,
    build_fission_video
)

app = FastAPI(title="PureCut Studio Pro", version="2.5.0")

# Work directories & Persistent Configuration
TEMP_DIR = REPO_ROOT / "temp_web_studio"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = TEMP_DIR / "config.json"

def load_studio_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_studio_config(updates: dict):
    cfg = load_studio_config()
    cfg.update(updates)
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Config] Failed to save config: {e}", file=sys.stderr)

_cfg = load_studio_config()
if _cfg.get("input_dir") and os.path.isdir(_cfg["input_dir"]):
    UPLOADS_DIR = Path(_cfg["input_dir"])
else:
    UPLOADS_DIR = TEMP_DIR / "uploads"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

if _cfg.get("output_dir") and os.path.isdir(_cfg["output_dir"]):
    OUTPUTS_DIR = Path(_cfg["output_dir"])
else:
    OUTPUTS_DIR = TEMP_DIR / "outputs"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_DIR = TEMP_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
FISSION_DIR = TEMP_DIR / "fission"
FISSION_DIR.mkdir(parents=True, exist_ok=True)


@app.api_route("/media/{full_path:path}", methods=["GET", "HEAD"])
async def get_media_file(full_path: str):
    clean_path = (TEMP_DIR / full_path).resolve()
    # Security: prevent path traversal outside TEMP_DIR
    if not str(clean_path).startswith(str(TEMP_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")
    if not clean_path.exists() or not clean_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    filename = clean_path.name
    media_type = "video/mp4" if filename.endswith((".mp4", ".webm", ".mov")) else "audio/mpeg"
    return FileResponse(
        clean_path,
        media_type=media_type,
        filename=filename,
        headers={"Accept-Ranges": "bytes"}
    )



@app.post("/api/split-video")
async def split_video(
    file: Optional[UploadFile] = File(None),
    local_path: str = Form(""),
    sample_fps: float = Form(2.5),
    min_duration: float = Form(1.5),
    margin: float = Form(0.35),
    face_thresh: float = Form(0.40),
    person_thresh: float = Form(0.35),
    human_filter_mode: str = Form("keep_hands"),
    subtitle_mode: str = Form("none"),
    subtitle_height: float = Form(0.16)
):
    try:
        job_id = uuid.uuid4().hex[:8]
        if local_path and os.path.exists(local_path):
            saved_video_path = Path(local_path)
            saved_video_name = saved_video_path.name
            raw_url = f"/api/media/stream-local?path={urllib.parse.quote(str(saved_video_path.resolve()))}"
            orig_filename = saved_video_path.name
        elif file and file.filename:
            ext = Path(file.filename or "video.mp4").suffix or ".mp4"
            saved_video_name = f"{job_id}_raw{ext}"
            saved_video_path = UPLOADS_DIR / saved_video_name
            with open(saved_video_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            raw_url = f"/media/uploads/{saved_video_name}"
            orig_filename = file.filename
        else:
            raise ValueError("请上传视频文件或指定本地视频文件路径！")

        # Initialize detector
        models_dir = str(REPO_ROOT / "models" / "vision")
        filter_engine = SmartHumanFilter(
            models_dir=models_dir,
            face_thresh=face_thresh,
            person_thresh=person_thresh,
            filter_mode=human_filter_mode
        )

        total_duration, human_ts, actual_fps = filter_engine.analyze_video(
            str(saved_video_path), sample_fps=sample_fps, filter_mode=human_filter_mode
        )
        human_intervals, clean_intervals = filter_engine.compute_intervals(
            total_duration, human_ts, actual_fps, margin=margin, min_clean_dur=min_duration
        )

        # Cut clean segments
        job_out_dir = OUTPUTS_DIR / job_id
        job_out_dir.mkdir(parents=True, exist_ok=True)
        clean_clips = filter_engine.cut_clean_segments(
            str(saved_video_path),
            clean_intervals,
            str(job_out_dir),
            subtitle_mode=subtitle_mode,
            sub_height_ratio=subtitle_height
        )

        # Build accessible media URLs
        for clip in clean_clips:
            clip["media_url"] = f"/media/outputs/{job_id}/{clip['filename']}"

        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "filename": orig_filename,
            "raw_video_url": raw_url,
            "total_duration": round(total_duration, 2),
            "has_human": len(human_intervals) > 0,
            "human_segments": [{"start": round(s, 2), "end": round(e, 2)} for s, e in human_intervals],
            "clean_segments": [{"start": round(s, 2), "end": round(e, 2), "duration": round(e - s, 2)} for s, e in clean_intervals],
            "clean_clips": clean_clips,
            "summary": f"扫描完成：原片时长 {total_duration:.1f}s，检测到 {len(human_intervals)} 处人物出镜，成功提纯切分出 {len(clean_clips)} 个纯净商品空镜。"
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/tts")
async def test_tts(
    provider: str = Form("openai"),
    base_url: str = Form("https://ark.cn-beijing.volces.com/api/v3"),
    api_key: str = Form(""),
    model: str = Form("doubao-tts"),
    voice: str = Form("zh_female_cancan"),
    speed: float = Form(1.0),
    emotion: str = Form("neutral"),
    text: str = Form(...)
):
    try:
        clean_text = text.strip()
        if not clean_text:
            raise HTTPException(status_code=400, detail="配音文本不能为空")

        # Normalize currency and numbers for ecommerce
        normalized_text = clean_text.replace("¥", "").replace("￥", "")

        audio_id = uuid.uuid4().hex[:10]
        out_path = AUDIO_DIR / f"{audio_id}.mp3"

        base = base_url.rstrip("/")

        if provider == "edge" or "edge" in base.lower() or voice.startswith("zh-CN-"):
            import edge_tts
            rate_percent = int((speed - 1.0) * 100)
            rate_str = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"
            edge_voice = voice if voice.startswith("zh-CN-") else "zh-CN-XiaoxiaoNeural"
            communicate = edge_tts.Communicate(normalized_text, edge_voice, rate=rate_str)
            await communicate.save(str(out_path))

            return JSONResponse({
                "success": True,
                "audio_url": f"/media/audio/{audio_id}.mp3",
                "provider": "edge-tts",
                "voice": edge_voice,
                "text": clean_text
            })

        elif provider == "minimax":
            headers = {
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json"
            }
            url = f"{base}/t2a_v2"
            payload = {
                "model": model or "speech-2.6-hd",
                "text": normalized_text,
                "voice_setting": {
                    "voice_id": voice or "female-tianmei",
                    "speed": speed,
                    "emotion": emotion if emotion != "neutral" else "neutral"
                },
                "audio_setting": {
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "format": "mp3"
                }
            }
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()
            if "data" in data and "audio" in data["data"]:
                import binascii
                hex_data = data["data"]["audio"]
                with open(out_path, "wb") as f:
                    f.write(binascii.unhexlify(hex_data))
            else:
                raise ValueError(f"MiniMax 未返回音频数据: {data.get('base_resp', data)}")

        else:
            # OpenAI compatible endpoint (Volcano Ark Doubao / SiliconFlow CosyVoice / Local CosyVoice)
            headers = {"Content-Type": "application/json"}
            if api_key and api_key.strip():
                headers["Authorization"] = f"Bearer {api_key.strip()}"
            elif "127.0.0.1" in base or "localhost" in base:
                headers["Authorization"] = "Bearer local-offline"

            url = f"{base}/audio/speech"
            payload = {
                "model": model,
                "input": normalized_text,
                "voice": voice,
                "response_format": "mp3",
                "speed": speed
            }
            res = requests.post(url, headers=headers, json=payload, timeout=35)
            if not res.ok:
                err_msg = res.text[:200]
                raise ValueError(f"TTS 请求失败 ({res.status_code}): {err_msg}")

            with open(out_path, "wb") as f:
                f.write(res.content)

        return JSONResponse({
            "success": True,
            "audio_url": f"/media/audio/{audio_id}.mp3",
            "provider": provider,
            "voice": voice,
            "text": clean_text
        })

    except requests.exceptions.ConnectionError as ce:
        if "9233" in str(ce) or "127.0.0.1:9233" in str(base_url):
            err_msg = "【本地离线语音服务未启动】检测到你的电脑当前没有运行 CosyVoice 语音服务 (端口 9233)。\n\n👉 快速体验建议：\n请点击上方的【火山豆包 · 灿灿】或【硅基流动 · Bella】等云端预设，并在下方填入对应平台的 API Key 即可立即在线合成！\n\n👉 若要使用此离线免Key模式，需先在本地启动 CosyVoice2 开源服务（监听 9233 端口）。"
            return JSONResponse({"success": False, "error": err_msg}, status_code=503)
        return JSONResponse({"success": False, "error": f"网络连接被拒绝，无法访问目标服务地址: {base_url}"}, status_code=503)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


def build_directory_tree(current_dir: Path, root_path: Path = None):
    """Recursively builds a multi-level folder tree of all video assets."""
    if root_path is None:
        root_path = current_dir

    video_exts = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".flv", ".wmv"}
    node = {
        "id": "dir_" + hashlib.md5(str(current_dir.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:8],
        "name": current_dir.name or str(current_dir.resolve()),
        "rel_path": current_dir.relative_to(root_path).as_posix() if current_dir != root_path else "",
        "abs_path": str(current_dir.resolve()),
        "type": "directory",
        "expanded": True,
        "children": [],
        "video_count": 0
    }

    try:
        entries = sorted(list(current_dir.iterdir()), key=lambda e: (not e.is_dir(), e.name.lower()))
    except Exception:
        entries = []

    for entry in entries:
        if entry.name.startswith((".", "$", "System Volume Information", "node_modules", "__pycache__")):
            continue
        try:
            if entry.is_dir():
                sub_node = build_directory_tree(entry, root_path)
                # 无论子文件夹有无视频，都保留其结构节点，便于查看目录
                node["children"].append(sub_node)
                node["video_count"] += sub_node["video_count"]
            elif entry.is_file() and entry.suffix.lower() in video_exts:
                try:
                    size_mb = round(entry.stat().st_size / (1024 * 1024), 1)
                except Exception:
                    size_mb = 0.0
                file_node = {
                    "id": "file_" + hashlib.md5(str(entry.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:8],
                    "name": entry.name,
                    "rel_path": entry.relative_to(root_path).as_posix(),
                    "abs_path": str(entry.resolve()),
                    "type": "file",
                    "size_mb": size_mb,
                    "media_url": f"/api/media/stream-local?path={urllib.parse.quote(str(entry.resolve()))}"
                }
                node["children"].append(file_node)
                node["video_count"] += 1
        except Exception:
            continue

    return node


@app.get("/api/settings/directories")
async def get_directories():
    cfg = load_studio_config()
    return JSONResponse({
        "success": True,
        "input_dir": str(UPLOADS_DIR.resolve()),
        "output_dir": str(OUTPUTS_DIR.resolve()),
        "fission_dir": str(FISSION_DIR.resolve()),
        "match_mode": cfg.get("match_mode", "local"),
        "api_key": cfg.get("api_key", ""),
        "base_url": cfg.get("base_url", "")
    })


@app.post("/api/settings/directories")
async def set_directories(
    input_dir: str = Form(""),
    output_dir: str = Form(""),
    fission_dir: str = Form(""),
    match_mode: str = Form("local"),
    api_key: str = Form(""),
    base_url: str = Form("")
):
    global UPLOADS_DIR, OUTPUTS_DIR, FISSION_DIR
    if input_dir and os.path.isdir(input_dir):
        UPLOADS_DIR = Path(input_dir)
    if output_dir:
        OUTPUTS_DIR = Path(output_dir)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    if fission_dir:
        FISSION_DIR = Path(fission_dir)
        FISSION_DIR.mkdir(parents=True, exist_ok=True)

    save_studio_config({
        "input_dir": str(UPLOADS_DIR.resolve()),
        "output_dir": str(OUTPUTS_DIR.resolve()),
        "fission_dir": str(FISSION_DIR.resolve()),
        "match_mode": match_mode,
        "api_key": api_key,
        "base_url": base_url
    })

    tree = None
    if UPLOADS_DIR.exists() and UPLOADS_DIR.is_dir():
        tree = build_directory_tree(UPLOADS_DIR)

    return JSONResponse({
        "success": True,
        "input_dir": str(UPLOADS_DIR.resolve()),
        "output_dir": str(OUTPUTS_DIR.resolve()),
        "fission_dir": str(FISSION_DIR.resolve()),
        "match_mode": match_mode,
        "tree": tree
    })


@app.get("/api/media/scan-tree")
async def scan_directory_tree(dir: str = "", path: str = ""):
    raw_dir = dir or path
    target_dir = Path(raw_dir) if raw_dir and os.path.isdir(raw_dir) else UPLOADS_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        target_dir = TEMP_DIR / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)

    tree = build_directory_tree(target_dir)
    return JSONResponse({
        "success": True,
        "root_path": str(target_dir.resolve()),
        "root_name": target_dir.name or str(target_dir.resolve()),
        "total_videos": tree["video_count"],
        "tree": tree
    })


@app.post("/api/dialog/pick-folder")
def pick_folder_dialog():
    try:
        ps_cmd = [
            "powershell", "-NoProfile", "-Command",
            """
            Add-Type -AssemblyName System.Windows.Forms
            $f = New-Object System.Windows.Forms.FolderBrowserDialog
            $f.Description = '请选择包含视频素材的文件夹 (支持多层子目录)'
            $f.ShowNewFolderButton = $false
            $topForm = New-Object System.Windows.Forms.Form
            $topForm.TopMost = $true
            $topForm.Opacity = 0
            $topForm.Show()
            if ($f.ShowDialog($topForm) -eq [System.Windows.Forms.DialogResult]::OK) {
                Write-Output $f.SelectedPath
            }
            $topForm.Close()
            """
        ]
        res = subprocess.run(ps_cmd, capture_output=True, text=True, timeout=120)
        selected_path = res.stdout.strip()
        if selected_path and os.path.isdir(selected_path):
            target_path = Path(selected_path)
            global UPLOADS_DIR
            UPLOADS_DIR = target_path
            save_studio_config({"input_dir": str(target_path.resolve())})
            tree = build_directory_tree(target_path)
            return JSONResponse({
                "success": True,
                "selected_path": str(target_path.resolve()),
                "total_videos": tree["video_count"],
                "tree": tree
            })
        return JSONResponse({"success": False, "cancelled": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.api_route("/api/media/stream-local", methods=["GET", "HEAD"])
async def stream_local_video(path: str = "", request: Request = None):
    if not path:
        raise HTTPException(status_code=400, detail="Missing path")
    decoded = urllib.parse.unquote(path)
    file_path = Path(decoded).resolve()
    if not file_path.exists() or not file_path.is_file():
        alt = Path(path).resolve()
        if alt.exists() and alt.is_file():
            file_path = alt
        else:
            raise HTTPException(status_code=404, detail=f"File not found: {decoded}")

    video_exts = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".flv", ".wmv"}
    if file_path.suffix.lower() not in video_exts:
        raise HTTPException(status_code=403, detail="Not a valid video file")

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range") if request else None

    content_type = "video/mp4"
    if file_path.suffix.lower() == ".webm":
        content_type = "video/webm"
    elif file_path.suffix.lower() == ".mov":
        content_type = "video/quicktime"

    if range_header:
        byte_range = range_header.replace("bytes=", "").split("-")
        start = int(byte_range[0]) if byte_range[0] else 0
        end = int(byte_range[1]) if len(byte_range) > 1 and byte_range[1] else file_size - 1
        length = end - start + 1

        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                bytes_left = length
                while bytes_left > 0:
                    chunk_size = min(bytes_left, 128 * 1024)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": content_type
        }
        return StreamingResponse(iter_file(), status_code=206, headers=headers)
    else:
        return FileResponse(file_path, media_type=content_type, headers={"Accept-Ranges": "bytes"})


@app.get("/api/clean-clips-pool")
async def get_clean_clips_pool():
    clips = []
    for mp4 in OUTPUTS_DIR.glob("**/*_clean_*.mp4"):
        dur = get_media_duration(str(mp4))
        if dur > 0.5:
            rel = mp4.relative_to(TEMP_DIR).as_posix()
            clips.append({
                "path": str(mp4),
                "filename": mp4.name,
                "duration": round(dur, 2),
                "media_url": f"/media/{rel}"
            })
    total_dur = sum(c["duration"] for c in clips)
    return JSONResponse({
        "success": True,
        "total": len(clips),
        "total_duration": round(total_dur, 2),
        "clips": clips
    })


def resolve_temp_media_path(path_str: str) -> Path:
    """Safely resolves a path or media URL to an actual file on disk."""
    if not path_str:
        raise ValueError("视频路径不能为空")
    cleaned = path_str.strip()

    # Handle /api/media/stream-local?path=...
    if "stream-local?path=" in cleaned:
        raw = urllib.parse.unquote(cleaned.split("stream-local?path=")[-1])
        p_stream = Path(raw).resolve()
        if p_stream.exists() and p_stream.is_file():
            return p_stream

    # Handle direct absolute path
    direct = Path(cleaned).resolve()
    if direct.exists() and direct.is_file():
        return direct

    if cleaned.startswith("/media/"):
        cleaned = cleaned[len("/media/"):]
    p = (TEMP_DIR / cleaned).resolve()
    if p.exists() and p.is_file():
        return p

    # Try in OUTPUTS_DIR or UPLOADS_DIR
    fname = Path(cleaned).name
    cand1 = OUTPUTS_DIR / fname
    if cand1.exists() and cand1.is_file():
        return cand1
    cand2 = UPLOADS_DIR / fname
    if cand2.exists() and cand2.is_file():
        return cand2

    matched = list(OUTPUTS_DIR.glob(f"**/{fname}"))
    if matched and matched[0].is_file():
        return matched[0]

    raise ValueError(f"未找到目标视频文件: {path_str}")


@app.post("/api/video/generate-clean-base")
def generate_clean_base(
    video_path: str = Form(""),
    file: UploadFile = File(None),
    remove_audio: bool = Form(True),
    initial_blur: bool = Form(False),
    aspect_ratio: str = Form("9:16"),
    human_filter_mode: str = Form("none"),
    min_duration: float = Form(1.5)
):
    try:
        job_id = uuid.uuid4().hex[:8]
        if file and file.filename:
            ext = Path(file.filename).suffix or ".mp4"
            src_path = UPLOADS_DIR / f"{job_id}_raw{ext}"
            with open(src_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        elif video_path:
            src_path = resolve_temp_media_path(video_path)
        else:
            # Fallback to first available clean clip or raw upload
            clean_clips = list(OUTPUTS_DIR.glob("**/*_clean_*.mp4"))
            if clean_clips:
                src_path = clean_clips[0]
            else:
                raise ValueError("请先在素材库中选择或上传一个视频素材！")

        out_name = f"clean_base_{job_id}.mp4"
        out_path = OUTPUTS_DIR / out_name

        # If human filter is requested, detect clean intervals and concatenate them
        purified_src = src_path
        if human_filter_mode in ["keep_hands", "strict"]:
            try:
                filter_tool = SmartHumanFilter()
                dur, h_stamps, s_fps = filter_tool.detect_human_segments(str(src_path), sample_fps=2.5, filter_mode=human_filter_mode)
                _, clean_intervals = filter_tool.compute_intervals(dur, h_stamps, s_fps, margin=0.35, min_clean_dur=min_duration)
                if clean_intervals:
                    seg_dir = TEMP_DIR / "temp_segments" / job_id
                    seg_dir.mkdir(parents=True, exist_ok=True)
                    seg_paths = []
                    for idx, (st, en) in enumerate(clean_intervals):
                        seg_p = str(seg_dir / f"seg_{idx:02d}.mp4")
                        cmd_cut = ["ffmpeg", "-y", "-ss", f"{st:.3f}", "-to", f"{en:.3f}", "-i", str(src_path), "-c", "copy", "-avoid_negative_ts", "make_zero", seg_p]
                        subprocess.run(cmd_cut, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if os.path.exists(seg_p) and os.path.getsize(seg_p) > 1000:
                            seg_paths.append(seg_p)

                    if seg_paths:
                        list_file = seg_dir / "concat_list.txt"
                        with open(list_file, "w", encoding="utf-8") as f:
                            for sp in seg_paths:
                                f.write(f"file '{sp}'\n")
                        concat_out = str(seg_dir / "concatenated.mp4")
                        cmd_concat = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", concat_out]
                        subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if os.path.exists(concat_out) and os.path.getsize(concat_out) > 1000:
                            purified_src = Path(concat_out)
            except Exception as fe:
                print(f"[HumanFilter] Warning: filter failed, fallback to original: {fe}", file=sys.stderr)

        w, h = get_video_dimensions(str(purified_src))
        if aspect_ratio == "9:16":
            target_w, target_h = 1080, 1920
        elif aspect_ratio == "3:4":
            target_w, target_h = 1080, 1440
        elif aspect_ratio == "16:9":
            target_w, target_h = 1920, 1080
        else:
            target_w, target_h = w, h

        filters = []
        if aspect_ratio in ["9:16", "3:4", "16:9"]:
            src_ratio = w / h if h > 0 else 0
            tgt_ratio = target_w / target_h
            if abs(src_ratio - tgt_ratio) > 0.03 or abs(w - target_w) > 4:
                filters.append(f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},setsar=1")

        if initial_blur:
            by = int(target_h * 0.82)
            bh = int(target_h * 0.14)
            bw = int(target_w * 0.88)
            bx = int(target_w * 0.06)
            filters.append(f"delogo=x={bx}:y={by}:w={bw}:h={bh}:show=0")

        if not filters and remove_audio and purified_src == src_path:
            # Ultra-fast stream-copy when no re-scaling or blur is required
            cmd = ["ffmpeg", "-y", "-i", str(purified_src), "-c:v", "copy", "-an", str(out_path)]
        else:
            vf_arg = ["-vf", ",".join(filters)] if filters else []
            cmd = ["ffmpeg", "-y", "-i", str(purified_src)]
            cmd.extend(vf_arg)
            if remove_audio:
                cmd.append("-an")
            else:
                cmd.extend(["-c:a", "aac", "-b:a", "128k"])

            cmd.extend([
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "22",
                str(out_path)
            ])

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            raise RuntimeError(f"底片生成失败: {res.stderr[-400:]}")

        dur = get_media_duration(str(out_path))
        out_w, out_h = get_video_dimensions(str(out_path))
        rel_url = f"/media/outputs/{out_name}"

        return JSONResponse({
            "success": True,
            "filename": out_name,
            "media_url": rel_url,
            "duration": round(dur, 2),
            "width": out_w,
            "height": out_h
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/video/erase-roi")
def erase_roi(
    video_path: str = Form(...),
    regions: str = Form(...)
):
    try:
        src_path = resolve_temp_media_path(video_path)
        if not src_path.exists():
            raise ValueError(f"原始底片文件不存在: {video_path}")

        try:
            boxes = json.loads(regions)
        except Exception:
            raise ValueError("选区格式不正确，需为 JSON 数组")

        if not boxes:
            raise ValueError("请至少框选一个需要消除的字幕区域！")

        w, h = get_video_dimensions(str(src_path))

        delogo_filters = []
        for b in boxes:
            bx = b.get("x", 0)
            by = b.get("y", 0)
            bw = b.get("w", 0)
            bh = b.get("h", 0)

            # If normalized ratio (0.0 <= val <= 1.0)
            if isinstance(bx, float) and bx <= 1.0:
                pix_x = int(round(bx * w))
                pix_y = int(round(by * h))
                pix_w = int(round(bw * w))
                pix_h = int(round(bh * h))
            else:
                pix_x = int(round(bx))
                pix_y = int(round(by))
                pix_w = int(round(bw))
                pix_h = int(round(bh))

            # Clamp to safe bounds (delogo requires x+w<=width, y+h<=height, w>=2, h>=2)
            pix_x = max(0, min(w - 4, pix_x))
            pix_y = max(0, min(h - 4, pix_y))
            pix_w = max(4, min(w - pix_x, pix_w))
            pix_h = max(4, min(h - pix_y, pix_h))

            delogo_filters.append(f"delogo=x={pix_x}:y={pix_y}:w={pix_w}:h={pix_h}:show=0")

        job_id = uuid.uuid4().hex[:8]
        out_name = f"clean_base_erased_{job_id}.mp4"
        out_path = OUTPUTS_DIR / out_name

        vf_str = ",".join(delogo_filters)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src_path),
            "-vf", vf_str,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "22",
            "-an",
            str(out_path)
        ]

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            raise RuntimeError(f"区域擦除失败: {res.stderr[-400:]}")

        dur = get_media_duration(str(out_path))
        out_w, out_h = get_video_dimensions(str(out_path))
        rel_url = f"/media/outputs/{out_name}"

        return JSONResponse({
            "success": True,
            "filename": out_name,
            "media_url": rel_url,
            "duration": round(dur, 2),
            "width": out_w,
            "height": out_h,
            "erased_count": len(boxes)
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/video/render-final-package")
def render_final_package(
    base_video_path: str = Form(...),
    copy_text: str = Form(...),
    voice: str = Form("zh-CN-XiaoxiaoNeural"),
    speed: float = Form(1.05),
    aspect_ratio: str = Form("9:16"),
    subtitle_color: str = Form("yellow"),
    font_size: int = Form(58),
    position_v: float = Form(0.75),
    font_name: str = Form("Microsoft YaHei"),
    provider: str = Form("edge"),
    base_url: str = Form(""),
    api_key: str = Form(""),
    model: str = Form("")
):
    try:
        src_path = resolve_temp_media_path(base_video_path)
        clean_text = copy_text.strip()
        if not clean_text:
            raise ValueError("口播文案不能为空！")

        job_id = uuid.uuid4().hex[:8]
        job_dir = FISSION_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # 1. Synthesize Voice
        audio_name = f"voice_{job_id}.mp3"
        audio_path = str(job_dir / audio_name)
        synthesize_voice(
            text=clean_text,
            voice=voice,
            speed=speed,
            out_mp3_path=audio_path,
            base_url=base_url,
            api_key=api_key,
            model=model
        )
        audio_dur = get_media_duration(audio_path)
        if audio_dur <= 0:
            audio_dur = 8.0

        # 2. Build Subtitles
        sentences = split_text_to_subtitles(clean_text)
        ass_name = f"sub_{job_id}.ass"
        ass_path = str(job_dir / ass_name)
        generate_ass_file(
            sentences=sentences,
            total_duration=audio_dur,
            out_ass_path=ass_path,
            aspect_ratio=aspect_ratio,
            color_scheme=subtitle_color,
            font_size=font_size,
            position_v=position_v,
            font_name=font_name
        )

        # 3. Assemble with Base Video
        out_name = f"package_{job_id}.mp4"
        out_path = str(job_dir / out_name)

        video_dur = get_media_duration(str(src_path))
        loop_flag = ["-stream_loop", "-1"] if video_dur < audio_dur else []

        cmd = ["ffmpeg", "-y"]
        cmd.extend(loop_flag)
        cmd.extend([
            "-i", str(src_path),
            "-i", audio_path,
            "-filter_complex", f"[0:v]trim=duration={audio_dur:.3f},ass={ass_name}[vout]",
            "-map", "[vout]",
            "-map", "1:a",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "21",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", f"{audio_dur:.3f}",
            out_name
        ])

        res = subprocess.run(cmd, cwd=str(job_dir), capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"成片合成失败: {res.stderr[-400:]}")

        final_dur = get_media_duration(out_path)
        rel_url = f"/media/fission/{job_id}/{out_name}"

        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "filename": out_name,
            "media_url": rel_url,
            "duration": round(final_dur, 1),
            "aspect_ratio": aspect_ratio,
            "copy_text": clean_text
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/fission-batch")
def create_fission_batch(
    count: int = Form(3),
    copy_text: str = Form(...),
    voice: str = Form("zh-CN-XiaoxiaoNeural"),
    speed: float = Form(1.05),
    aspect_ratio: str = Form("9:16"),
    subtitle_color: str = Form("yellow"),
    max_scene_dur: float = Form(3.0),
    anti_repeat: bool = Form(True),
    base_url: str = Form(""),
    api_key: str = Form(""),
    model: str = Form("")
):
    try:
        all_clips = [str(p) for p in OUTPUTS_DIR.glob("**/*_clean_*.mp4") if get_media_duration(str(p)) > 0.5]
        if not all_clips:
            raise ValueError("当前素材库中没有提取到的空镜切片！请先在【智能去人像】页面提纯至少一个视频。")

        lines = [line.strip() for line in copy_text.strip().split("\n") if line.strip()]
        if not lines:
            raise ValueError("请输入至少一段带货口播文案！")

        job_id = uuid.uuid4().hex[:8]
        job_dir = FISSION_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for i in range(count):
            line_idx = i % len(lines)
            cur_text = lines[line_idx]

            # 1. Voice synthesis
            audio_filename = f"voice_{i+1}.mp3"
            audio_path = str(job_dir / audio_filename)
            synthesize_voice(
                text=cur_text,
                voice=voice,
                speed=speed,
                out_mp3_path=audio_path,
                base_url=base_url,
                api_key=api_key,
                model=model
            )
            audio_dur = get_media_duration(audio_path)

            # 2. ASS subtitles
            sentences = split_text_to_subtitles(cur_text)
            ass_filename = f"sub_{i+1}.ass"
            ass_path = str(job_dir / ass_filename)
            generate_ass_file(
                sentences=sentences,
                total_duration=audio_dur,
                out_ass_path=ass_path,
                aspect_ratio=aspect_ratio,
                color_scheme=subtitle_color
            )

            # 3. Assemble video
            out_filename = f"fission_{job_id}_{i+1:02d}.mp4"
            out_path = str(job_dir / out_filename)
            build_fission_video(
                clip_paths=all_clips,
                audio_path=audio_path,
                ass_path=ass_path,
                out_video_path=out_path,
                aspect_ratio=aspect_ratio,
                anti_repeat=anti_repeat,
                max_scene_dur=max_scene_dur
            )

            final_dur = get_media_duration(out_path)
            results.append({
                "index": i + 1,
                "filename": out_filename,
                "duration": round(final_dur, 1),
                "copy_text": cur_text,
                "media_url": f"/media/fission/{job_id}/{out_filename}",
                "aspect_ratio": aspect_ratio
            })

        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "total_generated": len(results),
            "videos": results
        })

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "studio_template.html")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Studio Template Not Found</h1>"


def start_server(port=8765):
    print(f"\n=======================================================")
    print(f"🚀 PureCut Web Studio 已启动!")
    print(f"👉 本地浏览器访问测试: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    port = 8765
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    start_server(port)
