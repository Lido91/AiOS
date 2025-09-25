#!/usr/bin/env python3
import subprocess
from pathlib import Path

# adjust these paths or times if needed
input_video = Path("demo/short_asl.mp4")
output_video = Path("demo/short_asl_clip3.mp4")
start_time = "00:00:05"   # hh:mm:ss
duration = "00:00:30"     # length of clip

if not input_video.exists():
    raise FileNotFoundError(f"Input video not found: {input_video}")

cmd = [
    "ffmpeg", "-y",
    "-ss", start_time,
    "-i", str(input_video),
    "-t", duration,
    "-c", "copy",
    str(output_video),
]

print("Running:", " ".join(cmd))
result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

if result.returncode != 0:
    print(result.stderr)
    raise SystemExit(f"ffmpeg failed (exit code {result.returncode})")

print(f"Clip saved to {output_video.resolve()}")
