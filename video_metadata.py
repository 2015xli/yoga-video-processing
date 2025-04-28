import re
import argparse
import json, pysrt
import shutil
import subprocess, os
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import tiktoken

class YogaMetadataGenerator:
    def __init__(self, srt_parser, video_path, api_type='deepseek', split_method='token', stride=15, overlap=5):
        self.srt_parser = srt_parser
        self.video_path = video_path
        self.api_client = self._init_api(api_type)
        self.time_format = "%H:%M:%S"
        self.model = "deepseek-reasoner" if "deepseek" in api_type else "gpt-4o" #"o4-mini" or "gpt-4o"
        # self.encoder = tiktoken.encoding_for_model(self.model)
        # tiktoken暂时不支持deepseek models
        self.encoder = tiktoken.encoding_for_model("gpt-4o")
        self.max_context_tokens = 30720  # DeepSeek上下文限制, 大致为64KB文件
        self.split_method = split_method    
        self.chunk_stride = stride * 60   # 划分一个chunk的stride时间，一个chunk的实际时间为stride+overlap
        self.chunk_overlap = overlap * 60  # 两个相邻chunk之间的重叠时间
        self.chunk_max = (stride+overlap) * 1.5 * 60 # 最长chunk的时间, 1.5个stride，用于计算最后一个chunk大小。
        self.chapter_length_min = 1 * 60 # 最短一章时间为1分钟（秒）

        self.chunks = self._partition_subtitles_to_chunks(srt_parser.segments)
        
        
    def _init_api(self, api_type):
        if api_type == 'deepseek':
            return OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com/v1"
            )
        elif api_type == 'openai':
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        raise ValueError("Unsupported API type")

    def _load_template_file(self, filename):
        # 计算模板文件路径（脚本所在目录的同级文件）
        base_dir = Path(__file__).parent
        prompt_path = base_dir / filename
        template = prompt_path.read_text(encoding="utf-8")
        return template

    def _process_srt_prompt(self, subtitles):
      
        # 转为字符串
        json_section = subtitles

        # 计算模板文件路径（脚本所在目录的同级文件）
        template = self._load_template_file("prompt_process_srt.txt")

        # 使用 f-string 渲染模板中的表达式（例如 {self.chunk_stride + self.chunk_overlap}）
        prompt_body = eval(f"f'''{template}'''")

        # 在函数内部追加 JSON 格式的 simplified 内容
        prompt = f"{prompt_body}\n{json_section}"

        return prompt

    def _process_chunk_prompt(self, segments_chunk):
        simplified = [{
            "start": seg['start_str'],
            "text": re.sub(r'\s+', ' ', seg['text'])
        } for seg in segments_chunk]
        
        # 转为字符串
        json_section = json.dumps(simplified, ensure_ascii=False, indent=2)

        # 计算模板文件路径（脚本所在目录的同级文件）
        template = self._load_template_file("prompt_process_chunk.txt")

        # 使用 f-string 渲染模板中的表达式（例如 {self.chunk_stride + self.chunk_overlap}）
        prompt_body = eval(f"f'''{template}'''")

        # 在函数内部追加 JSON 格式的 simplified 内容
        prompt = f"{prompt_body}\n{json_section}"

        return prompt

    def _process_summary_prompt(self, summaries):
        simplified = {f"chapter{i+1}": content for i, content in enumerate(summaries)}
        
        # 转为字符串
        json_section = json.dumps(simplified, ensure_ascii=False, indent=2)

        # 计算模板文件路径（脚本所在目录的同级文件）
        template = self._load_template_file("prompt_process_summary.txt")

        # 使用 f-string 渲染模板中的表达式（例如 {self.chunk_stride + self.chunk_overlap}）
        prompt_body = eval(f"f'''{template}'''")

        # 在函数内部追加 JSON 格式的 simplified 内容
        prompt = f"{prompt_body}\n{json_section}"

        return prompt

    def _process_chapters_prompt(self, chapters):
        # 构造简化后的章节字典
        simplified = {f"section{i+1}": content for i, content in enumerate(chapters)}
        # 转为字符串
        json_section = json.dumps(simplified, ensure_ascii=False, indent=2)

        # 计算模板文件路径（脚本所在目录的同级文件）
        template = self._load_template_file("prompt_process_chapters.txt")

        # 使用 f-string 渲染模板中的表达式（例如 {self.chunk_stride + self.chunk_overlap}）
        prompt_body = eval(f"f'''{template}'''")

        # 在函数内部追加 JSON 格式的 simplified 内容
        prompt = f"{prompt_body}\n{json_section}"

        return prompt

    def _ai_process(self, content):
        messages = [{
            "role": "user",
            "content": content,
            "temperature": 0.2
        }]
        response = self.api_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        json_response=self.clean_json_string(response.choices[0].message.content)
        print(f"{json_response}")
        return json.loads(json_response)

    def _merge_chapters(self, chapters):
        merged = []
        for chap in sorted(chapters, key=lambda x: x['start']):
            if not merged:
                merged.append(chap)
                continue
                
            last = merged[-1]
            last_end = self._time_to_sec(last['start']) + self.chapter_length_min  # 假设最小章节时长
            current_start = self._time_to_sec(chap['start'])
            
            if current_start < last_end:  
                # 开始时间相差不足一分钟，认为时间重叠，合并为一个章节
                if chap['posture'] in last['posture']:
                    # 体式已经在章节中包含，忽略chap内容
                    pass
                elif last['posture'] in chap['posture']:
                    # chap体式更多，作为章节体式
                    last['posture'] = chap['posture']
                else:
                    # 否则，把chap体式加入章节体式
                    last['posture'] += f" → {chap['posture']}"
            else:
                # 否则，独立为一个章节
                merged.append(chap)
        return merged

    def _split_time_chunks(self, segments):
        self.total_duration = max(s['end_sec'] for s in segments)

        chunks = []
        current_start = 0

        while current_start < self.total_duration:
            # 先计算本chunk的最后一段的开始时间，以及下个chunk第一段的开始时间
            if self.total_duration - current_start < self.chunk_max:
                # 如果剩余长度小于允许的最大chunk，则不再分段，直接把本chunk的终止时间设为最后
                next_start = self.total_duration
                current_end = next_start
            else:
                # 否则，本chunk的最后一段的开始时间设为加上一个标准chunk大小(stride+overlap)
                next_start = current_start + self.chunk_stride
                current_end = next_start + self.chunk_overlap
            # 将开始时间落在开始段、终止段之间的段放入chunk
            chunk = [
                seg for seg in segments
                if seg['start_sec'] >= current_start
                and seg['start_sec'] < current_end
            ]
            
            if chunk:
                chunks.append(chunk)
            
            # 前面的一个chunk分出来后，新的chunk开始
            current_start = next_start

        return chunks

    def _split_token_chunks(self, segs):
        # 1) Precompute and cache each seg’s token_count
        total_tokens = 0
        for seg in segs:
            if 'token_count' not in seg:
                seg['token_count'] = len(self.encoder.encode(seg['text']))
                
            total_tokens += seg['token_count']
        
        # 如果total_tokens并不多，那就用缺省的time分段方式来切分，不会出现超标问题
        if total_tokens < self.max_context_tokens * 1.2:
            return self._split_time_chunks(segs)

        # 如果total_tokens可能超标，就用按token分段方式来切分，这样也不会出现超标问题
        # 一段大小控制在 0.2 * max_context_tokens, 因为如果太大，识别效果会较差
        chunks = []
        current_chunk = []
        current_tokens = 0

        for seg in segs:
            # Easy access now
            seg_tokens = seg['token_count']

            # If adding this seg would overflow our soft limit...
            if current_chunk and (current_tokens + seg_tokens > self.max_context_tokens * 0.2):
                # === close out the old chunk ===
                prev_chunk = current_chunk

                # compute max allowed overlap: min(self.chunk_overlap, half of prev_chunk duration)
                prev_duration = prev_chunk[-1]['start_sec'] - prev_chunk[0]['start_sec']
                allowed_overlap = min(self.chunk_overlap, prev_duration * 0.5)

                # gather overlap segments from the end of prev_chunk
                overlap_segs = []
                for p in reversed(prev_chunk):
                    if seg['start_sec'] - p['start_sec'] < allowed_overlap:
                        overlap_segs.insert(0, p)
                    else:
                        break

                # save the completed chunk
                chunks.append(prev_chunk)

                # start new chunk seeded with the overlap
                current_chunk = overlap_segs.copy()
                current_tokens = sum(p['token_count'] for p in current_chunk)

            # always append the current segment
            current_chunk.append(seg)
            current_tokens += seg_tokens

        # append any remaining
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _analyze_postures_in_chunks(self, chunks): 
        
        all_responses = []
        for chunk in chunks:
            content = self._process_chunk_prompt(chunk)
            response = self._ai_process(content)
            all_responses.append(response)
        
        print(all_responses)
        
        if len(chunks) == 1:                    
            analysis = all_responses[0];
            
        elif len(chunks) == 2:                    
            all_summaries = [response['summary'] for response in all_responses ]
            all_chapters = [response['chapters'] for response in all_responses]
            chapters = self._merge_chapters(all_chapters)

            content = self._process_summary_prompt(all_summaries)
            description = self._ai_process(content)['summary']
            
            analysis = {"summary": description, "chapters": chapters}
        else:
            all_chapters = [ response['chapters'] for response in all_responses ] 
            
            content = self._process_chapters_prompt(all_chapters)
                      
            analysis = self._ai_process(content)
            # mostly the analysis is good enough, but sometimes it needs merge chapters.
            description = analysis['summary']
            chapters = self._merge_chapters(analysis['chapters'])
            analysis = {"summary": description, "chapters": chapters)
        
        return analysis

    def _partition_subtitles_to_chunks(self, segments):
        
        if self.split_method == 'time':
            chunks = self._split_time_chunks(segments)
        else:
            chunks = self._split_token_chunks(segments)
        
        return chunks

    def analyze_postures(self):
        
        analysis = self._analyze_postures_in_chunks(self.chunks)
            
        return analysis
       
    def refine_srt_words(self):
        subtitle_path = self.srt_parser.subtitle_path

        #TODO 
        all_subtitles = []
        for chunk in self.chunks:
            content = self._process_srt_prompt(chunk)
            chunk_subtitles = self._ai_process(content)['refined_srt']
            all_subtitles.extend(chunk_subtitles)
        #use new subtitles to replace segment contents, the chunks data structure remains
        
        new_subtitles = json.dump(all_subtitles)
        new_subtitle_file = f"{subtitle_path.stem}.refined.srt"
        new_subtitle_path = subtitle_path.parent / Path(new_subtitle_file)
        new_subtitle_path.write_text(new_subtitles, encoding='utf-8')
        return new_subtitle_path
          
    def _time_to_sec(self, time_str):
        dt = datetime.strptime(time_str, "%H:%M:%S")
        return dt.hour * 3600 + dt.minute * 60 + dt.second

    def generate_ffmetadata(self, analysis: dict, video_path: Path) -> Path:
        metadata = [";FFMETADATA1"]
        metadata.append(f"title={analysis.get('title', '瑜伽课程')}")
        metadata.append(f"artist={analysis.get('artist', '诗桉,Mason,Lucy')}")
        metadata.append(f"description={analysis.get('summary', '瑜伽课程:诗桉,Mason,Lucy')}\n")
        
        for i, chap in enumerate(analysis['chapters']):
            start_sec = self._time_to_sec(chap['start'])
            metadata.extend([
                "[CHAPTER]",
                "TIMEBASE=1/1",
                f"START={start_sec}",
                f"END={start_sec + 60}",
                f"title=Chapter {i+1}: {chap['posture']}\n"
            ])
        
        base_name = video_path.stem
        metadata_path = video_path.parent / Path(f"{base_name}.metadata.txt")
        metadata_path.write_text('\n'.join(metadata), encoding='utf-8')
        return metadata_path

    def clean_json_string(self, json_string):
        pattern = r'^```json\s*(.*?)\s*```$'
        cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
        return cleaned_string.strip()
    
class SrtParser:

    def __init__(self, subtitle_path)：
        self.subtitle_path = subtitle_path
        segments = self._parse(subtitle_path)
        self.segments = self._merge_segments(segments, max_gap=0.1, max_duration=60.0)
    
    def _parse(self, subtitle_path):
        """使用 pysrt 解析 SRT 文件"""
        subs = pysrt.open(subtitle_path)
        entries = []
        
        for sub in subs:
            entry = {
                # 转换 pysrt 的 Time 对象为 datetime.time
                'start': sub.start.to_time(),
                'end': sub.end.to_time(),
                'text': sub.text.replace('\n', ' ').strip(),
                # 直接计算秒数（包含毫秒的小数部分）
                'start_sec': sub.start.ordinal / 1000.0,
                'end_sec': sub.end.ordinal / 1000.0,
                # 生成 HH:MM:SS 格式字符串
                'start_str': f"{sub.start.hours:02}:{sub.start.minutes:02}:{sub.start.seconds:02}"
            }
            entries.append(entry)
        return entries

    def _merge_segments(self, entries, max_gap=0.1, max_duration=60.0):
        """合并相邻条目（相差不超过0.1秒，最长合并不超过60秒）,减少冗余的时间戳"""
        if not entries:
            return []
            
        merged = [entries[0].copy()]
        for current in entries[1:]:
            last = merged[-1]
            
            # 计算两段的时间间隔
            gap = current['start_sec'] - last['end_sec']
            # 计算上一段的时长
            duration = current['start_sec'] - last['start_sec']
            
            if gap <= max_gap and duration <= max_duration:
                # 计算两段的时间间隔很小，而且合并后的段也不长
                # 合并文本
                last['text'] += " " + current['text']
                # 更新结束时间
                last['end'] = current['end']
                last['end_sec'] = current['end_sec']
            else:
                merged.append(current.copy())
                
        return merged


class MetadataGenerator:
   
    @staticmethod
    def _extract_creation_time(filename):
        match = re.search(r'_(\d{8})_(\d{6})_', filename)
        if not match:
            return None
        date_str = match.group(1)
        time_str = match.group(2)
        try:
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            hour = time_str[:2]
            minute = time_str[2:4]
            second = time_str[4:6]
            datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
        except ValueError:
            return None

    @staticmethod
    def _generate_title(filename):
        match = re.search(r'(\d{8})_(\d{6})', filename)
        if match:
            return f"瑜伽课程_{match.group(1)}_{match.group(2)}"
        return "瑜伽课程"

    def generate(self, subtitle_path: Path, video_path: Path, 
              api_type: str = 'deepseek', split_method: str = 'token',
              stride: int = 15, overlap: int = 5, skip_ai: bool = False) -> Path:

        srt_parser = SrtParser(subtitle_path)

        processor = YogaMetadataGenerator(srt_parser, video_path, api_type, split_method, stride, overlap)
        
        if not skip_ai:
            subtitle_path = processor.refine_srt_words()
                
        
        filename = video_path.name
        base_name = video_path.stem
        summary_file = f"{base_name}.summary.json"
        summary_path = video_path.parent / Path(summary_file)
                
        if not skip_ai:              
            analysis = processor.analyze_postures()                
            analysis['title'] = MetadataGenerator._generate_title(filename)
            analysis['artist'] = "诗桉,Mason,Lucy"
            
            summary_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
        else:
            with summary_path.open('r', encoding='utf-8') as f:
                analysis = json.load(f)
        
        metadata_path = processor.generate_ffmetadata(analysis, video_path)
        output_path = video_path.parent / Path(f"{base_name}.metadata.{video_path.suffix}")
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-i', str(metadata_path),
            '-map_metadata', '1',
            '-map_chapters', '1',
        ]
        creation_time = MetadataGenerator._extract_creation_time(filename)
        if creation_time:
            cmd.extend(['-metadata', f'creation_time={creation_time}'])

        cmd.extend(['-c', 'copy', str(output_path)])

        #Sometimes output of ffmpeg cannot be decoded as utf-8. Just don't decode it, leave it binary.
        #result = subprocess.run(cmd, capture_output=True, text=True, encoding='latin-1')
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            shutil.move(str(output_path), str(video_path))
            print("元数据嵌入成功")
        else:
            output_path.unlink(missing_ok=True)
            print(f"错误：{result.stderr.decode('utf-8')}")
            
        return video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yoga video metadata generation based on subtitle file.')
    parser.add_argument('subtitle', type=Path, help='SRT字幕文件路径')
    parser.add_argument('video', type=Path, help='视频文件路径')
    parser.add_argument('--api', choices=['deepseek', 'openai'], default='deepseek')
    parser.add_argument("--split", choices=['time', 'token'], default='token')
    parser.add_argument("--stride", type=float, default=15, help="chunk stride in minutes (default: 15)" )
    parser.add_argument("--overlap", type=float, default=5, help="chunk overlapping in minutes (default: 5)")
    parser.add_argument('--skipai', action='store_true', help='跳过AI处理，使用本地缓存 （default: False)')
    args = parser.parse_args()
    
    metadata_generator = MetadataGenerator()
    metadata_generator.generate(args.subtitle, args.video, args.api, args.split, args.stride, args.overlap, args.skipai)
