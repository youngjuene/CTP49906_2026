#!/usr/bin/env python3
"""Prepare one original/silent classroom pair, matching a reference video's length.

Run with a downloaded source file; no network access or model inference occurs.
The original audio is resampled, not replaced. Both variants share copied video
packets and independently encode the same-length audible/silent PCM tracks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path),
    ], text=True))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--media-id", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "assets/02321.mp4")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"scene[0-9]{2}", args.media_id):
        parser.error("media-id must have the form scene02")
    if not math.isfinite(args.start) or args.start < 0:
        parser.error("start must be a finite nonnegative number")
    reference = probe(args.reference)
    duration = float(next(s for s in reference["streams"] if s["codec_type"] == "video")["duration"])
    source = probe(args.source)
    if not any(s["codec_type"] == "audio" for s in source["streams"]):
        parser.error("source must contain an audio stream")
    if args.start + duration > float(source["format"]["duration"]):
        parser.error("selected interval extends beyond the source")
    if not 0 < duration <= 10:
        parser.error("reference must be at most 10 seconds for this classroom recipe")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.out_dir / (args.media_id + suffix) for suffix in (".mp4", "_silent.mp4", ".preparation.json")]
    if any(path.exists() for path in targets):
        parser.error("output already exists; use an empty output directory")
    commands = []

    def run(arguments):
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", *map(str, arguments)]
        subprocess.run(command, check=True)
        commands.append(command)

    seconds = f"{duration:.9f}"
    samples = round(duration * 16000)
    video_filter = (
        "setpts=PTS-STARTPTS,"
        "scale=w='min(640,iw)':h='min(360,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "setsar=1,fps=24,format=yuv420p"
    )
    audio_filter = f"aresample=16000,apad,atrim=end_sample={samples},asetpts=N/SR/TB"
    with tempfile.TemporaryDirectory(prefix="lesson-media-") as folder:
        temp = Path(folder)
        video, pcm = temp / "video.mp4", temp / "source.wav"
        run(["-ss", args.start, "-i", args.source, "-t", seconds, "-map", "0:v:0", "-an",
             "-vf", video_filter, "-c:v", "libx264", "-preset", "medium", "-crf", "23",
             "-map_metadata", "-1", "-map_chapters", "-1", video])
        run(["-ss", args.start, "-i", args.source, "-t", seconds, "-map", "0:a:0", "-vn",
             "-ac", "1", "-af", audio_filter, "-c:a", "pcm_s16le", pcm])
        pending = []
        for variant in ("original", "silent"):
            output = temp / (variant + ".mp4")
            run(["-i", video, "-i", pcm, "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "copy", "-af", "volume=0" if variant == "silent" else "anull",
                 "-c:a", "aac", "-b:a", "64k", "-ar", "16000", "-ac", "1",
                 "-t", seconds, "-map_metadata", "-1", "-map_chapters", "-1",
                 "-movflags", "+faststart", output])
            if output.stat().st_size > 3 * 2**20:
                raise ValueError("Prepared file exceeds 3 MiB")
            info = probe(output)
            for kind in ("video", "audio"):
                stream = next(s for s in info["streams"] if s["codec_type"] == kind)
                if not math.isclose(float(stream["duration"]), duration, abs_tol=1e-6):
                    raise ValueError(f"{variant} {kind} length does not match reference")
            pending.append(output)
        for output, target in zip(pending, targets[:2]):
            output.replace(target)
    report = {
        "schema_version": 1, "media_id": args.media_id,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "reference": {"file": args.reference.name, "sha256": sha256(args.reference), "video_duration_seconds": duration},
        "source": {"file": args.source.name, "sha256": sha256(args.source),
                   "start_seconds": args.start, "end_seconds": args.start + duration},
        "recipe": {"video_fps": 24, "audio_sample_rate": 16000, "audio_channels": 1,
                   "pcm_sample_count": samples, "script_sha256": sha256(Path(__file__))},
        "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        "commands": commands,
        "outputs": [{"file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size} for p in targets[:2]],
    }
    targets[2].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"media_id": args.media_id, "duration_seconds": duration, "files": report["outputs"]}))


if __name__ == "__main__":
    main()
