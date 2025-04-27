#!/usr/bin/env python3
import argparse, re
import subprocess
from pathlib import Path
import shutil

class SubtitleGenerator:
    # Supported video extensions
    SUPPORTED_EXT = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']

    # Prompt for finer transcription guidance
    PROMPT = (
        """ 
    瑜伽名词(glossary)：盘腿，金刚跪坐，简易坐，至善坐，调息，后脑勺，数息法，深吸，缓呼，长呼，慢呼；
    体式：下犬式，侧板式，猫牛式，八体投地，四足跪姿，前屈，后弯，幻椅式，犁式；
    身体:髋部，肩部，坐骨，髂骨，脊柱，肩胛骨，大脚球，虎口，后腰；
    动作:伸展，并拢，蹬地，内收，向上提，向后推，扭转；
    """
    )
    # Subtitle formatting
    MAX_LINE_WIDTH = 14  # For OpenAI Whisper
    MAX_LINE_COUNT = 2   # For OpenAI Whisper

    def extract_audio(self, video_path: Path, audio_path: Path) -> None:
        """Extracts audio from video at 16kHz mono WAV format."""
        if audio_path.exists():
            return

        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")

    def format_time(self, seconds: float) -> str:
        """Converts seconds to SRT time format (HH:MM:SS,ms)."""
        ms_total = int(seconds * 1000)
        hours, remainder = divmod(ms_total, 3600000)
        minutes, remainder = divmod(remainder, 60000)
        sec, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{sec:02d},{ms:03d}"

    def generate_srt_faster(self, segments) -> str:
        """
        Builds SRT content for faster-whisper, splitting long lines at commas,
        merging small pieces back together up to MAX_LINE_WIDTH, and
        estimating per-piece timings proportionally.
        """
        entries = []
        idx = 1

        for seg in segments:
            text = seg.text.strip()

            # 1) If already short enough, emit as one entry:
            if len(text) <= SubtitleGenerator.MAX_LINE_WIDTH:
                start = self.format_time(seg.start)
                end   = self.format_time(seg.end)
                entries.append(f"{idx}\n{start} --> {end}\n{text}\n")
                idx += 1
                continue

            # 2) Initial split at commas (ASCII or Chinese):
            raw_pieces = re.findall(r".*?[，,]|[^，,]+$", text)

            # 3) Clean up & merge consecutive small pieces:
            merged = []
            current = ""
            for piece in raw_pieces:
                clean = piece.strip().rstrip("，,")
                if not clean:
                    continue

                if not current:
                    current = clean
                else:
                    # would adding this piece stay under the width?
                    if len(current) + 1 + len(clean) <= SubtitleGenerator.MAX_LINE_WIDTH:
                        current = f"{current}，{clean}"
                    else:
                        merged.append(current)
                        current = clean

            if current:
                merged.append(current)

            # 4) Distribute timing across each merged line
            total_chars = sum(len(m) for m in merged)
            duration    = seg.end - seg.start
            cum_chars   = 0

            for m in merged:
                rel_start = duration * (cum_chars / total_chars)
                rel_end   = duration * ((cum_chars + len(m)) / total_chars)
                start_t   = self.format_time(seg.start + rel_start)
                end_t     = self.format_time(seg.start + rel_end)

                entries.append(f"{idx}\n{start_t} --> {end_t}\n{m}\n")
                idx += 1
                cum_chars += len(m)

        return "\n".join(entries)


    def process_faster_whisper(self, video_path: Path, temp_path: Path, model) -> Path:
        """Processing pipeline for faster-whisper."""
        from faster_whisper import WhisperModel

        audio_path = temp_path / f"{video_path.stem}_audio.wav"
        srt_temp = temp_path / f"{video_path.stem}.srt"
        
        try:
            self.extract_audio(video_path, audio_path)
            segments, _ = model.transcribe(
                str(audio_path),
                language='zh',
                beam_size=5,
                vad_filter=True,
                initial_prompt=SubtitleGenerator.PROMPT
            )
            srt_text = self.generate_srt_faster(segments)
            with open(srt_temp, 'w', encoding='utf-8') as f:
                f.write(srt_text)
            final_srt = video_path.with_suffix('.srt')
            shutil.move(str(srt_temp), str(final_srt))
            print(f"srt_temp subtitles: {final_srt}")
            
        finally:          
            audio_path.unlink(missing_ok=True)
               
        return final_srt
            

    def process_openai_whisper(self, video_path: Path, temp_path: Path, model) -> Path:
        """Processing pipeline for OpenAI Whisper."""
        from whisper.utils import get_writer

        audio_path = temp_path / f"{video_path.stem}_audio.wav"
        srt_temp = audio_path.with_suffix('.srt')
        final_srt = video_path.with_suffix('.srt')
        
        try:
            self.extract_audio(video_path, audio_path)
            result = model.transcribe(
                str(audio_path),
                language='zh',
                initial_prompt=SubtitleGenerator.PROMPT,
                word_timestamps=True
            )
            writer = get_writer("srt", temp_path)
            word_options = {
                "max_line_width": SubtitleGenerator.MAX_LINE_WIDTH,
                "max_line_count": SubtitleGenerator.MAX_LINE_COUNT
            }
            writer(result, str(audio_path), word_options)
            shutil.move(str(srt_temp), str(final_srt))
            print(f"srt_temp subtitles: {final_srt}")
            
        finally:
            audio_path.unlink(missing_ok=True)

        return final_srt


    def transcribe(self, video_path: Path, temp_path: Path, model_type: str = 'faster-whisper' ) -> Path:
        temp_path.mkdir(parents=True, exist_ok=True)
        result_srt = video_path.with_suffix('.srt');

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if video_path.suffix.lower() not in SubtitleGenerator.SUPPORTED_EXT:
            raise ValueError(f"Unsupported file extension: {video_path.suffix}")
        if result_srt.exists():
            print(f"Subtitle already exists for {video_path.name}")
            return result_srt

        if model_type == "faster-whisper":
            from faster_whisper import WhisperModel
            model = WhisperModel("medium", device="auto", compute_type="float16")
            result_srt = self.process_faster_whisper(video_path, temp_path, model)
        elif model_type == "openai-whisper":
            import whisper
            result_srt = whisper.load_model("medium")
            return self.process_openai_whisper(video_path, temp_path, model)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return result_srt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Transcribe video to SRT using specified Whisper model')
    parser.add_argument('video_file', type=Path, help='Path to the input video file')
    parser.add_argument('temp_dir', type=Path, help='Directory for temporary files')
    parser.add_argument('--model', choices=['faster-whisper', 'openai-whisper'],
                        default='faster-whisper', help='ASR model to use')
    args = parser.parse_args()
 
    subtitle_generator = SubtitleGenerator()
    subtitle_generator.transcribe(args.video_file, args.temp_dir, args.model)