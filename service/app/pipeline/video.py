"""Video I/O through the ffmpeg CLI (handles rotation metadata, HEVC, VFR and odd containers)."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    width: int          # display width after rotation
    height: int
    fps: float
    duration: float
    codec: str
    crop: tuple[int, int, int, int] | None = None   # (w, h, x, y) of the picture inside black borders, if any


@dataclass
class Frame:
    t: float            # seconds from start
    image: np.ndarray   # BGR, already scaled to the analysis size


def probe(path: Path) -> VideoInfo:
    p = Path(path)
    return _probe(str(p), p.stat().st_mtime)


@lru_cache(maxsize=64)
def _probe(path: str, _mtime: float) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,duration:stream_side_data=rotation:"
         "stream_tags=rotate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=60,
    )
    meta = json.loads(out.stdout)
    if not meta.get("streams"):
        raise ValueError("no video stream found")
    st = meta["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    rotation = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = rotation or int((st.get("tags") or {}).get("rotate", 0) or 0)
    if abs(rotation) % 180 == 90:
        w, h = h, w

    def _rate(s: str) -> float:
        num, _, den = s.partition("/")
        return float(num) / float(den or 1) if float(den or 1) else 0.0

    fps = _rate(st.get("avg_frame_rate", "0/1")) or _rate(st.get("r_frame_rate", "0/1")) or 30.0
    duration = float(st.get("duration") or meta.get("format", {}).get("duration") or 0)
    return VideoInfo(width=w, height=h, fps=fps, duration=duration, codec=st.get("codec_name", "?"),
                     crop=_detect_borders(path, w, h, duration))


NORMALIZED_NAME = "source.norm.mp4"
_REMUX_CODECS = {"h264", "hevc"}
_REMUX_EXT = {".mp4", ".mov", ".m4v", ".mkv"}   # not .ts/.mts/.m2ts: start offsets, AVCHD is often interlaced


def normalize(src: Path, out: Path, max_long_side: int = 1920, timeout: float = 240) -> Path:
    """Any supported upload -> one MP4 (video only, timestamps from 0) that the rest of the pipeline reads.

    H.264/HEVC in MP4/MOV/MKV is remuxed without re-encoding, so frames stay bit-exact. Everything else (MPEG-2,
    MPEG-4 part 2, VP9, transport streams, ...) or a failed remux is re-encoded to H.264: upright, deinterlaced,
    square pixels, long side at most `max_long_side`.
    """
    src, out = Path(src), Path(out)
    res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", str(src)], capture_output=True, text=True, timeout=60)
    codec = res.stdout.strip().splitlines()[0].strip() if res.stdout.strip() else ""
    base = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src), "-map", "0:v:0", "-an", "-sn", "-dn"]
    if codec in _REMUX_CODECS and src.suffix.lower() in _REMUX_EXT:
        tag = ["-tag:v", "hvc1"] if codec == "hevc" else []
        cmd = [*base, "-c:v", "copy", *tag, "-movflags", "+faststart", str(out)]
        if subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0 and _readable(out):
            return out
    k = f"min(1,{max_long_side}/max(iw*sar,ih))"
    vf = f"yadif=deint=interlaced,scale=w='trunc(iw*sar*{k}/2)*2':h='trunc(ih*{k}/2)*2',setsar=1"
    cmd = [*base, "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0 or not _readable(out):
        raise ValueError(f"could not convert {src.suffix} ({codec or 'unknown codec'}) to MP4: "
                         f"{res.stderr.strip()[-300:]}")
    return out


def _readable(path: Path) -> bool:
    res = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=codec_name:format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, timeout=60)
    return res.returncode == 0 and bool(res.stdout.strip()) and path.stat().st_size > 0


def _detect_borders(path: str, w: int, h: int, duration: float) -> tuple[int, int, int, int] | None:
    """Black letterbox / pillarbox borders (a different camera or a re-encode can add them): the picture inside."""
    boxes = []
    for frac in (0.15, 0.5, 0.85):
        res = subprocess.run(["ffmpeg", "-v", "info", "-nostdin", "-ss", f"{duration * frac:.2f}", "-i", path, "-t", "1",
                              "-vf", "cropdetect=limit=24:round=2:reset=0", "-an", "-f", "null", "-"],
                             capture_output=True, text=True, timeout=60)
        found = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", res.stderr)
        if found:
            boxes.append(tuple(int(v) for v in found[-1]))
    if not boxes:
        return None
    x1 = min(b[2] for b in boxes)
    y1 = min(b[3] for b in boxes)
    x2 = max(b[2] + b[0] for b in boxes)
    y2 = max(b[3] + b[1] for b in boxes)
    cw, ch = x2 - x1, y2 - y1
    if cw < 0.3 * w or ch < 0.3 * h or (cw > 0.96 * w and ch > 0.96 * h):
        return None   # nothing worth cropping (or a dark scene fooled the detector)
    return cw - cw % 2, ch - ch % 2, x1, y1


def _crop_filter(info: VideoInfo) -> str:
    return f"crop={info.crop[0]}:{info.crop[1]}:{info.crop[2]}:{info.crop[3]}," if info.crop else ""


def analysis_size(info: VideoInfo, max_long_side: int) -> tuple[int, int]:
    src_w, src_h = (info.crop[0], info.crop[1]) if info.crop else (info.width, info.height)
    scale = min(1.0, max_long_side / max(src_w, src_h))
    w = int(round(src_w * scale / 2) * 2)
    h = int(round(src_h * scale / 2) * 2)
    return w, h


def iter_frames(path: Path, info: VideoInfo, fps: float, max_long_side: int):
    """Yield frames resampled to a constant `fps` (ffmpeg fps filter keeps timestamps exact)."""
    w, h = analysis_size(info, max_long_side)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path),
           "-vf", f"{_crop_filter(info)}fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = w * h * 3
    i = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield Frame(t=i / fps, image=np.frombuffer(buf, np.uint8).reshape(h, w, 3))
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def grab_frames(path: Path, times: list[float], max_long_side: int | None = None) -> list[tuple[float, np.ndarray]]:
    """Frame-accurate grabs at specific timestamps (used for evidence)."""
    out = []
    info = probe(path)
    vf = ["-vf", _crop_filter(info).rstrip(",")] if info.crop else []
    for t in times:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(t, 0):.3f}", "-i", str(path), *vf,
               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
        res = subprocess.run(cmd, capture_output=True, timeout=60)
        if not res.stdout:
            continue
        img = cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        if max_long_side and max(img.shape[:2]) > max_long_side:
            s = max_long_side / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        out.append((t, img))
    return out


def dense_frames(path: Path, start: float, end: float, max_long_side: int) -> list[tuple[float, np.ndarray]]:
    """All native frames in [start, end] — used to pick the sharpest evidence frame."""
    info = probe(path)
    w, h = analysis_size(info, max_long_side)
    start = max(0.0, start)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{max(end - start, 0.05):.3f}",
           "-vf", f"{_crop_filter(info)}scale={w}:{h}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    res = subprocess.run(cmd, capture_output=True, timeout=120)
    n = len(res.stdout) // (w * h * 3)
    arr = np.frombuffer(res.stdout[: n * w * h * 3], np.uint8).reshape(n, h, w, 3) if n else np.zeros((0, h, w, 3), np.uint8)
    step = 1.0 / (info.fps or 30.0)
    return [(start + i * step, arr[i]) for i in range(n)]


def cut_clip(path: Path, out: Path, start: float, end: float) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{max(start, 0):.3f}", "-i", str(path),
           "-t", f"{max(end - max(start, 0), 0.5):.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "24", "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-movflags", "+faststart", str(out)]
    return subprocess.run(cmd, capture_output=True, timeout=180).returncode == 0


def make_preview(path: Path, out: Path, max_long_side: int = 960) -> bool:
    """Browser-playable H.264 copy of the upload (the .mov may be HEVC)."""
    info = probe(path)
    w, h = analysis_size(info, max_long_side)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path), "-an", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "26", "-pix_fmt", "yuv420p", "-vf", f"scale={w}:{h}", "-movflags", "+faststart", str(out)]
    return subprocess.run(cmd, capture_output=True, timeout=600).returncode == 0


def jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
