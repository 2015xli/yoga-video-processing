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
    def __init__(self, api_type='deepseek', split_method='token', stride=15, overlap=5):
        self.api_client = self._init_api(api_type)
        self.time_format = "%H:%M:%S"
        self.model = "deepseek-reasoner" if "deepseek" in api_type else "o4-mini"  # was gpt-4o
        # self.encoder = tiktoken.encoding_for_model(self.model)
        # tiktoken暂时不支持deepseek models
        self.encoder = tiktoken.encoding_for_model("gpt-4o")
        self.max_context_tokens = 30720  # DeepSeek上下文限制, 大致为64KB文件
        self.split_method = split_method    
        self.chunk_stride = stride * 60   # 划分一个chunk的stride时间，一个chunk的实际时间为stride+overlap
        self.chunk_overlap = overlap * 60  # 两个相邻chunk之间的重叠时间
        self.chunk_max = (stride+overlap) * 1.5 * 60 # 最长chunk的时间, 1.5个stride，用于计算最后一个chunk大小。
        self.chapter_length_min = 1 * 60 # 最短一章时间为1分钟（秒）
        
    def _init_api(self, api_type):
        if api_type == 'deepseek':
            return OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com/v1"
            )
        elif api_type == 'openai':
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        raise ValueError("Unsupported API type")

    def _process_chunk_prompt(self, segments_chunk):
        simplified = [{
            "start": seg['start_str'],
            "text": re.sub(r'\s+', ' ', seg['text'])
        } for seg in segments_chunk]
        
        return f"""请严格按以下规则分析瑜伽课程字幕, 给出重点训练体式的总结，并分出重点体式的章节，输出必须包含且只能包含summary (包含几个重点体式), chapters (开始时间和体式名称)。 在总结时，不要使用梵文、英文。只用json格式输出结果，不要包含任何marker如```json，或其他分析内容：

1. 重点体式识别规则：
   - 课程指令中，有时会提到特定的体式：
     * 战士一式、板式，侧板式，八体投地，四足跪姿，下犬式，猫牛式，龙式，鸽子式，海豹式，前屈，后弯，幻椅式，犁式，山式，等等。
     * 各种体式并非都重要，不重要的可以忽略
        ** 有些体式或动作是为了休息调整，比如平躺放松、婴儿式、大拜式，分析总结时需要把这些内容去掉。
        ** 有些体式比较简单，是例行的过渡体式，比如八体投地，四足跪姿，下犬式，猫牛式，可以不必在总结中提到，除非是专门的训练，或者是流瑜伽序列中一个环节。
     * 你需要重点关注:1、比较困难的、有挑战的体式；2、多次重复训练的体式；3、耗时比较长的体式（超过2分钟）；4、流瑜伽序列。

   - 比较困难的、有挑战的体式，需要在summary中重点提到。
     * 比如，各种流序列、倒立、战士式、龙式、鸽子式、犁式、等，都是难度很大的体式。
     * 遇到难度大的体式，指令可能会有关键词，比如需要力量，有难度，很吃力，有挑战，尝试一下，加油，坚持一下，或者是某个动作的"加深"练习
     * 一个有挑战的体式做完后，往往需要做"婴儿式"或"大拜式"来休息放松一下，根据这个可以判断之前的体式后又挑战。
     * 一个有挑战的伸展类体式做完后，可能会需要做反向的收回体式来收一下，比如前侧链拉伸做完后，需要拉伸后侧链来收回，或者反过来。如果遇到收回动作，说明之前的体式比较重要。
     * 对于特别重点的体式，请在括号中标出"挑战"、"力量"、或者"困难"之类，不用每个体式都标注，只标注特别重点的体式。

   - 如果遇到下面关键词，说明正在进行一项重要体式，需要提取出当时的体式或动作：
     * 倒计时（5...4...3...； 五...四...三...）
     * 重复提示（"再做一次"）
     * 保持提示（"保持3个呼吸"、"停留片刻"）

   - 有时课程中没有给出具体体式的名称，可以描述实际动作，作为名称
     * 比如髋部伸展，脊柱扭转，胸部后弯，倒立，等
     * 有时原来的词可能不是很正确，可以找近似的同音词试试    


2. 章节划分逻辑：
   - 找到第一个完整姿势作为第一章
   - 将有挑战的体式、或者重要的体式尽量作为一章的开始
   - 过渡内容（"休息一下"、"婴儿式"、"大拜式"）不要列为章节，需要把这些内容去掉
   - 检测到另一种有挑战的体式、或者重要的体式时结束当前章节
   - 有时，多个不同体式会按顺序（即"阿斯汤加流"、"流瑜伽"、"流动序列"）循环两遍或多遍，尽量将一个完整流序列放在一章内，而不要跨章节。
     * 比如，一个流动序列可能包含山式→前屈→斜板→八体投地→上犬→下犬多个体式，循环几遍。请把一遍循环的多个体式放在一章，几遍循环就分成几章。可能连续几章都是相同的流序列。
     * 类似的比如，战士二式→侧角伸展式→逆转战士式→三角伸展式→半月式的流动循环。
     * 如果一个流只出现了一遍，没有循环几次，则它包含的几个体式可以分开在不同章节，这个不严格要求。
     * 对于循环的提示序列，尽量给出序列的名称，后面括号内给出具体的体式,比如，
       ** 太阳式流瑜伽序列（山式→前屈→斜板式→八体投地→上犬式→下犬式）
       ** 战士系列组合（战士二式→侧角伸展式→逆转战士式→三角伸展式→半月式）
       
   - 章节时间点选择，不严格要求，主要是选择体式进行中的时间点，比如：
     * 对于倒计时：取倒计时的中间点（比如，5秒倒计时取第3秒）
     * 对于呼吸保持：取保持期的中点
     * 对于重复练习：取第一个体式练习的时间中点
   - 每个章节的时间不能超过10分钟。一个课程不少于5个章节。如果有流序列，章节数可以减少。
   - 一个章节可以包含几个不同的体式

3. 示例输出格式(必须严格按照此格式输出, summary包含几个重点体式, chapters包含开始时间和体式名称)：
{{
  "summary": "...",
  "chapters": [
    {{"start": "00:01:23", "posture": "战士二式"}},
    {{"start": "00:02:59", "posture": "三角伸展式"}},
    {{"start": "00:05:12", "posture": "山式，前屈， 脊柱力量训练"}}
    {{"start": "00:07:19", "posture": "龙式（挑战）"}}
    {{"start": "00:10:01", "posture": "手肘倒立（困难）"}}
    {{"start": "00:20:42", "posture": "战士系列组合（战士二式→侧角伸展式→逆转战士式→三角伸展式→半月式）"}},
    {{"start": "00:25:58", "posture": "太阳式流（山式→前屈→斜板式→八体投地→上犬式→下犬式）"}},
    ...
  ]
}}

下面是实际字幕内容，请总结并按要求输出：
{json.dumps(simplified, ensure_ascii=False, indent=2)}"""

    def _process_chapters_prompt(self, chapters):
        simplified = {f"section{i+1}": content for i, content in enumerate(chapters)}
        
        return f"""下面是一节瑜伽课的几个段落section的内容总结，每个段落都有几个chapters。段落之间有时间上的相互重叠，每个段落大致{self.chunk_stride + self.chunk_overlap}分钟，两个相邻段落之间有大概{self.chunk_overlap}分钟的重叠，请根据这个内容，总结出这个瑜伽课的多个章节（chapters)，包含每个章节的训练内容(start and posture)，以及对总体课程内容进行一个概况描述(summary)。只用json格式输出结果，不要包含任何marker如```json，或其他分析内容。


1. 总结原则：
- 因为相邻段落的内容有重叠，在其各自总结里，可能对时间重叠的章节的体式描述有所不同，甚至表面上看起来有所冲突，你需要找出合理的方式来描述这些章节里的内容，让整个课程各个章节的内容看起来清晰一致。
- 对于重复几遍的多个瑜伽体式的流动序列，请把一遍循环的多个体式放在一章，几遍循环就分为几章。可能连续几章都是相同的流序列。这样容易看出来重复了几遍，也容易看出来这次课程的重点训练内容。比如"战士系列组合（战士二式→侧角伸展式→逆转战士式→三角伸展式→半月式）"。
- 有的训练几遍的体式序列虽然每遍不完全一样的，但主要是一些小的变化，比如一遍是左侧、一遍是右侧；或者一遍是外旋，一遍是内旋；这类训练最好也在章节中给出说明，比如"半莲花式外旋站立及反侧"、"双侧半盘莲花扭转（坐姿+站姿）"。
- 在总体描述summary中，你需要强调那些训练时间较长的体式、和重复多次的体式、以及在后面有标注为"困难""挑战""力量"的体式。如果是流瑜伽序列，在给出流的名称及训练遍数，并在括号里给出具体的体式名称。

2. 严格按照下面格式进行输出：
{{
  "summary": "本课核心内容...",
  "chapters": [
    {{"start": "00:01:23", "posture": "战士二式，三角伸展式"}},
    {{"start": "00:05:12", "posture": "山式，前屈， 脊柱力量训练"}}
    {{"start": "00:15:58", "posture": "太阳式流（山式→前屈→斜板式→八体投地→上犬式→下犬式）"}},
    {{"start": "00:17:08", "posture": "太阳式流（山式→前屈→斜板式→八体投地→上犬式→下犬式）"}},
    ...
    ]
}}

3. 下面是本次瑜伽课各阶段的内容，请总结并按要求输出：
{json.dumps(simplified, ensure_ascii=False, indent=2)}"""

    def _process_summary_prompt(self, summaries):
        simplified = {f"chapter{i+1}": content for i, content in enumerate(summaries)}
        
        return f"""下面是一个瑜伽课程的几个阶段的训练内容, 请给出整个课程的重点训练体式的总结，输出必须包含且只能包含summary。 只用json格式输出结果，不要包含任何marker如```json，或其他分析内容。字数在100字以内。

1. 体式名称有很多，可以是标准名称，也可以是动作描述
     * 常见的体式名称： 战士一式、板式，侧板式，八体投地，四足跪姿，下犬式，猫牛式，龙式，鸽子式，海豹式，前屈，后弯，幻椅式，犁式，山式，等等。
     * 常见的动作描述： 比如髋部伸展，脊柱扭转，胸部后弯，倒立，等
     * 有时原来的词可能不是很正确，可以找近似的同音词试试
2. 你需要强调那些训练时间较长的体式、和重复多次的体式、以及在后面有标注为"困难""挑战""力量"的体式。如果是流瑜伽序列，在给出流的名称及训练遍数，并在括号里给出具体的体式名称。

3. 示例输出格式(必须严格按照此格式输出)：
{{
  "summary": "本课核心内容..."
}}

下面是课程各阶段的内容，请总结并按要求输出：
{json.dumps(simplified, ensure_ascii=False, indent=2)}"""

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
        
        return analysis

    def analyze_postures(self, segments):
        if self.split_method == 'time':
            chunks = self._split_time_chunks(segments)
        else:
            chunks = self._split_token_chunks(segments)
        
        analysis = self._analyze_postures_in_chunks(chunks)
            
        return analysis
                
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
    @staticmethod
    def parse(subtitle_path):
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

    @staticmethod
    def merge_segments(entries, max_gap=0.1, max_duration=60.0):
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
        segments = SrtParser.parse(subtitle_path)
        segments = SrtParser.merge_segments(segments, max_gap=0.1, max_duration=60.0)
        
        filename = video_path.name
        base_name = video_path.stem
        summary_file = f"{base_name}.summary.json"
        summary_path = video_path.parent / Path(summary_file)
        
        processor = YogaMetadataGenerator(api_type, split_method, stride, overlap)
        
        if not skip_ai:              
            analysis = processor.analyze_postures(segments)                
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
