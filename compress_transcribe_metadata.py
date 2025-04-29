#!/usr/bin/env python3
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 导入重构后的模块（需要将原三个文件重构为以下模块）
try:
    from video_compress import VideoCompressor
    from video_transcribe import SubtitleGenerator
    from video_metadata import MetadataGenerator
    from srt_refine import SrtRefiner
except ImportError:
    logger.error("Required modules not found. Please ensure video_compress.py, video_transcribe.py and video_metadata.py are available.")
    sys.exit(1)


class VideoPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.compressor = VideoCompressor()
        self.transcriber = SubtitleGenerator()
        self.srt_refiner = SrtRefiner()
        self.metadata_gen = MetadataGenerator()

    def process_file(self, input_path: Path, temp_root: Path, output_root: Path) -> Path:
        """处理单个视频文件的完整流程"""
        try:
            # 1. 压缩视频
            compressed_path = self._handle_compression(input_path, temp_root)
            
            # 2. 生成字幕
            srt_path = self._generate_subtitle(compressed_path, temp_root)
            
            refined_srt = self._refine_srt(srt_path)
            
            # 4. 嵌入元数据
            video_path = self._embed_metadata(compressed_path, srt_path, temp_root)
            
            # 5. 移到ouput目录，清理临时文件
            if temp_root != output_root:
                self._cleanup(video_path, temp_root, output_root)
            
            return video_path
            
        except Exception as e:
            logger.error(f"Failed to process {input_path}: {str(e)}")
            raise

    def _handle_compression(self, input_path: Path, temp_root: Path) -> Optional[Path]:
        """处理压缩阶段"""
        if self.config['skip_compression']:
            logger.info(f"Skipping compression for {input_path}")
            return input_path
            
        return self.compressor.compress(
            input_file=input_path,
            output_dir=temp_root,
            erase_original=self.config['erase_original']
        )

    def _generate_subtitle(self, video_path: Path, temp_root: Path) -> Path:
        """生成字幕文件"""
        return self.transcriber.transcribe(
            video_path=video_path,
            temp_path=temp_root,
            model_type=self.config['model']
        )

    def _refine_srt(self, subtitle_path: Path) -> Path:

        if self.config['skip_ai']:
            logger.info(f"Skipping subtitle refining for {subtitle_path}")
            return subtitle_path
            
        return self.srt_refiner.refine(
            subtitle_path = subtitle_path,  
            api_type = self.config['api'], 
            split_method=self.config['split'],
            stride=self.config['stride']
        )  


    def _embed_metadata(self, video_path: Path, srt_path: Path, temp_root: Path) -> Path:
        """嵌入元数据"""
        
        return self.metadata_gen.generate(
            subtitle_path=srt_path,
            video_path=video_path,
            api_type=self.config['api'],
            split_method=self.config['split'],
            stride=self.config['stride'],
            overlap=self.config['overlap'],
            skip_ai=self.config['skip_ai']
        )
        
    def _cleanup(self, video_path: Path, temp_root: Path, output_root: Path):
        if not video_path.is_file():
            raise ValueError(f"{video_path} is not a valid file")

        if temp_root.resolve() == output_root.resolve():
            return

        if not temp_root in video_path.parents:
            raise ValueError(f"{video_path} is not under {temp_root}")

        video_dir = video_path.parent
        base_name = video_path.stem  # without suffix

        # Find all files in the same directory whose names start with the base_name
        related_files = [f for f in video_dir.iterdir() if f.is_file() and f.name.startswith(base_name)]

        # Compute relative path from temp_root to video_path
        rel_path = video_path.relative_to(temp_root)
        output_dir = output_root / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy files to mirrored output structure
        for file in related_files:
            shutil.copy2(file, output_dir / file.name)

        # Optionally remove original files
        if self.config['erase_temp']:
            for file in related_files:
                file.unlink()


    def process_directory(self, input_root: Path, temp_root: Path, output_root: Path):
        """处理整个目录树"""
        if self.config['skip_compression']:
            if input_root.resolve() != temp_root.resolve():
                raise ValueError(f"Input and temp should be the same when skip-compression: \n Input {str(input_root)} != temp: {str(temp_root)} ")

         
        for path in input_root.rglob('*'):
            if not path.is_file():
                continue
                
            if path.suffix.lower() not in VideoCompressor.SUPPORTED_EXT:
                continue
                                
            relative = path.relative_to(input_root)
            temp_dir = temp_root / relative.parent
            output_dir = output_root / relative.parent
            
            self.process_file(path, temp_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Video processing pipeline')
    parser.add_argument('input_root', type=Path, help='Root directory for original videos')
    parser.add_argument('temp_root', type=Path, help='Directory for intermediate files')
    parser.add_argument('output_root', type=Path, help='Directory for final output')
    parser.add_argument('--skip-compression', action='store_true', help='Skip compression step. Input and temp are same, and use input as temp.')
    parser.add_argument('--erase-temp', action='store_true', help='Delete all temporary files')
    parser.add_argument('--erase-original', action='store_true', help='Delete original files after compression')
    parser.add_argument('--model', choices=['faster-whisper', 'openai-whisper'], 
                      default='faster-whisper', help='ASR model to use')
    parser.add_argument('--api', choices=['deepseek', 'openai'], 
                      default='deepseek', help='Metadata API provider')
    parser.add_argument('--split', choices=['time', 'token'], 
                      default='token', help='Chunk splitting method')
    parser.add_argument('--stride', type=int, default=15,
                      help='Time chunk stride in minutes (time split only)')
    parser.add_argument('--overlap', type=int, default=5,
                      help='Chunk overlap in minutes (time split only)')
    parser.add_argument('--skip-ai', action='store_true',
                      help='Skip AI processing for metadata')

    args = parser.parse_args()
    
    config = {
        'skip_compression': args.skip_compression,
        'erase_original': args.erase_original,
        'erase_temp': args.erase_temp,
        'model': args.model,
        'api': args.api,
        'split': args.split,
        'stride': args.stride,
        'overlap': args.overlap,
        'skip_ai': args.skip_ai
    }
    
    logger.info(f"Config: {config}")
    pipeline = VideoPipeline(config)
    try:
        pipeline.process_directory(
            input_root=args.input_root,
            temp_root=args.temp_root,
            output_root=args.output_root
        )
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()