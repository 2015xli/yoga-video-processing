#!/usr/bin/env python3
"""
video_transcribe_openai_api.py

1. Extract mono AAC from video
2. Split into overlapping segments (customizable stride)
3. Transcribe each with OpenAI Whisper API (Chinese + adaptive prompts)
4. Merge SRTs into final output, copy to video directory
"""

import os
import argparse
import subprocess
import shutil
from pathlib import Path
import pysrt
from openai import OpenAI  # New client style

# === Prompt definitions ===
OPENAI_PROMPT = """瑜伽课，女老师；
好~啦~，来，开始；
盘腿:简易坐，调息，后脑勺，数息法，深吸，缓呼，长呼，慢呼；
体式：下犬式，侧板式，猫牛式，八体投地，四足跪姿，前屈，后弯，幻椅式，犁式；
身体：髋部，肩部，坐骨，髂骨，脊柱，肩胛骨，大脚球，虎口，后腰；
动作：伸展，并拢，蹬地，内收，向上提，向后推，放松；
最后，老师说Namaste，好啦~，下课了。
"""

OPENAI_First = """瑜伽课，女老师，
好~啦~，来，开始吧。
盘腿，简易坐，调息，后脑勺，数息法，深吸，缓呼，长呼，慢呼
瑜伽体式：下犬式，侧板式，猫牛式，八体投地，四足跪姿，前屈，后弯，幻椅式，犁式，山式，树式
身体部位：髋部，肩部，坐骨，髂骨，脊柱，肩胛骨，大脚球，虎口，后腰；
各种动作：伸展，并拢，蹬地，内收，向上提，向后推，放松；
"""

OPENAI_Middle = """瑜伽课，女老师，
瑜伽体式：下犬式，侧板式，猫牛式，八体投地，四足跪姿，前屈，后弯，幻椅式，犁式，山式，树式
身体部位：髋部，肩部，坐骨，髂骨，脊柱，肩胛骨，大脚球，虎口，后腰；
各种动作：伸展，并拢，蹬地，内收，向上提，向后推，放松；
"""

OPENAI_Last = """瑜伽课，女老师，
瑜伽体式：下犬式，侧板式，猫牛式，八体投地，四足跪姿，前屈，后弯，幻椅式，犁式，山式，树式
身体部位：髋部，肩部，坐骨，髂骨，脊柱，肩胛骨，大脚球，虎口，后腰；
各种动作：伸展，并拢，蹬地，内收，向上提，向后推，放松；
躺平，唤醒自己，动动手腕，动动脚踝，
最后，老师说Namaste，好啦~，下课了。
"""

# ===== Utility functions =====
def run_cmd(cmd):
    """Run shell command, error on failure."""
    subprocess.run(cmd, check=True)

def get_duration(video_path):
    """Return duration in seconds via ffprobe."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ])
    return float(out.strip())

# split and merge functions use globals STRIDE, SEG_LEN, OVERLAP, MAX_SINGLE
STRIDE     = 20 * 60  # default, overridden in main
SEG_LEN    = 22 * 60
OVERLAP    = 2  * 60
MAX_SINGLE = 30 * 60

def compute_segments(duration):
    """
    Compute list of (start_sec, length_sec) tuples for splitting.
    """
    segs = []
    base = 0.0
    while True:
        rem = duration - base
        if rem <= 0:
            break
        if STRIDE < rem < MAX_SINGLE:
            segs.append((base, rem))
            break
        if rem >= MAX_SINGLE:
            segs.append((base, SEG_LEN))
            base += STRIDE
        else:
            segs.append((base, rem))
            break
    return segs


def extract_audio(video, audio_out):
    """
    Extract mono AAC audio (128 kbps) into an M4A container.
    """
    run_cmd([
        "ffmpeg", "-y", "-i", video,
        "-vn",
        "-acodec", "aac", "-b:a", "128k",
        "-ac", "1",
        "-movflags", "+faststart",
        audio_out
    ])


def split_audio(audio_in, tempdir, base_name, segments):
    """
    Split into .m4a files using ffmpeg copy (no re-encode).
    """
    tempdir.mkdir(parents=True, exist_ok=True)
    outs = []
    for i, (st, ln) in enumerate(segments):
        out = tempdir / f"{base_name}.segment_{i:03d}.m4a"
        run_cmd([
            "ffmpeg", "-y", "-i", str(audio_in),
            "-ss", str(st), "-t", str(ln),
            "-acodec", "copy",
            str(out)
        ])
        outs.append(out)
    return outs


def transcribe_segments(segment_paths, tempdir, base_name):
    """
    Transcribe each audio with adaptive prompts into Chinese SRT.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set OPENAI_API_KEY in environment")
    client = OpenAI(api_key=api_key)

    num_segments = len(segment_paths)
    for idx, seg_path in enumerate(segment_paths):
        # select prompt
        if num_segments == 1:
            prompt = OPENAI_PROMPT
        else:
            if idx == 0:
                prompt = OPENAI_First
            elif idx == num_segments - 1:
                prompt = OPENAI_Last
            else:
                prompt = OPENAI_Middle

        srt_out = tempdir / f"{base_name}.segment_{idx:03d}.srt"
        with open(seg_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="srt",
                language="zh",
                prompt=prompt
            )
        srt_out.write_text(transcript, encoding="utf-8")


def merge_srt(segments, srt_paths, output_srt):
    """
    Merge SRTs with overlap logic, shift times, reindex.
    """
    merged = []
    half_ov = OVERLAP / 2.0
    last_idx = len(segments) - 1

    for idx, ((base, length), srt_file) in enumerate(zip(segments, srt_paths)):
        subs = pysrt.open(str(srt_file), encoding='utf-8')
        head_thr, tail_thr = half_ov, length - half_ov

        for sub in subs:
            st = sub.start.ordinal / 1000.0
            en = sub.end.ordinal   / 1000.0

            if idx == 0 and en <= tail_thr:
                sub.shift(seconds=base); merged.append(sub)
            elif 0 < idx < last_idx and head_thr <= st <= tail_thr:
                sub.shift(seconds=base); merged.append(sub)
            elif idx == last_idx and st >= head_thr:
                sub.shift(seconds=base); merged.append(sub)

    merged.sort(key=lambda s: s.start.ordinal)
    final = pysrt.SubRipFile(merged)
    final.clean_indexes()
    final.save(str(output_srt), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(
        description="Extract, split, transcribe (Chinese + adaptive prompts), and merge SRTs"
    )
    parser.add_argument("video", help="Input video file")
    parser.add_argument("tempdir", help="Temp directory for audio & .srt segments")
    parser.add_argument(
        "--stride",
        type=float,
        default=20,
        help="Segment stride in minutes (default: 20)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=2,
        help="Segment overlapping in minutes (default: 2)"
    )
    args = parser.parse_args()

    # override global split constants
    global STRIDE, SEG_LEN, OVERLAP, MAX_SINGLE
    STRIDE     = args.stride * 60
    OVERLAP    = args.overlap * 60
    SEG_LEN    = (args.stride + args.overlap) * 60
    MAX_SINGLE = args.stride * 1.5 * 60

    video_path = Path(args.video)
    base_name = video_path.stem
    temp = Path(args.tempdir)
    temp.mkdir(parents=True, exist_ok=True)

    # 1) Extract audio
    audio_file = temp / f"{base_name}.m4a"
    extract_audio(str(video_path), str(audio_file))

    # 2) Compute segments
    duration = get_duration(str(video_path))
    segments = compute_segments(duration)

    # 3) Split audio
    segment_paths = split_audio(audio_file, temp, base_name, segments)

    # 4) Transcribe segments with prompts
    transcribe_segments(segment_paths, temp, base_name)

    # 5) Merge SRTs into temp
    srt_paths = [temp / f"{base_name}.segment_{i:03d}.srt" for i in range(len(segments))]
    merged_temp = temp / f"{base_name}.openai.srt"
    merge_srt(segments, srt_paths, merged_temp)

    # 6) Copy final SRT to video dir
    final_srt = video_path.parent / f"{base_name}.openai.srt"
    shutil.copy(str(merged_temp), str(final_srt))
    print(f"Merged SRT available at {final_srt}")

if __name__ == "__main__":
    main()
