#!/usr/bin/env python3

import argparse
import subprocess
import socket
from pathlib import Path

class VideoCompressor:
    # Supported video file extensions
    SUPPORTED_EXT = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']

    def compress(self, input_file: Path, output_dir: Path, erase_original: bool) -> Path:
        # Determine host-specific ffmpeg configuration
        hostname = socket.gethostname().split('.')[0]
        if hostname == 'xf-workstation':
            ffmpeg_path = '/usr/bin/ffmpeg'
            video_codec = 'hevc_nvenc'
        elif hostname == 'XF-NAS':
            ffmpeg_path = '/usr/local/bin/ffmpeg5'
            video_codec = 'libx265'
        else:
            ffmpeg_path = 'ffmpeg'
            video_codec = 'libx265'

        if not input_file.exists() or not input_file.is_file():
            print(f"Input file does not exist: {input_file}")
            return None

        if input_file.suffix.lower() not in VideoCompressor.SUPPORTED_EXT:
            print(f"Unsupported file type: {input_file.suffix}")
            return None

        if 'x265' in input_file.name:
            print(f"Skipping potentially already compressed file: {input_file}")
            return input_file

        output_dir.mkdir(parents=True, exist_ok=True)

        new_filename = input_file.stem + f".x265{input_file.suffix.lower()}"
        output_path = output_dir / new_filename

        if output_path.exists():
            print(f"Skipping existing output: {output_path}")
            return output_path

        # Build and run ffmpeg command
        cmd = [
            ffmpeg_path, '-nostdin', '-i', str(input_file),
            '-c:v', video_codec, '-crf', '24', '-preset', 'slow',
            '-c:a', 'copy', '-y', str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            print(f"Successfully compressed: {output_path}")

            if erase_original and output_path.exists():
                try:
                    input_file.unlink()
                    print(f"Removed original: {input_file}")
                except OSError as e:
                    print(f"Error removing original: {e}")
        except subprocess.CalledProcessError as e:
            print(f"Compression failed for {input_file}: {e}")
            if output_path.exists():
                output_path.unlink()
                
        return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress a single video file with H.265 encoding.')
    parser.add_argument('input_file', type=Path, help='Path to input video file')
    parser.add_argument('output_dir', type=Path, help='Directory to save the compressed video')
    parser.add_argument('--erase-original', action='store_true', help='Delete original file after successful compression')
    args = parser.parse_args()
    
    video_compressor = VideoCompressor()
    video_compressor.compress(args.input_file, args.output_dir, args.erase_original)
