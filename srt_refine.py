import re, json
import argparse
import pysrt
import os
from pathlib import Path
from openai import OpenAI
import tiktoken
import shutil

class YogaSrtRefiner:
    def __init__(self, subtitle_path, api_type='deepseek', split_method='time', stride=15):
        self.subtitle_path = Path(subtitle_path)
        self.api_client = self._init_api(api_type)
        #"deepseek-reasoner" "deepseek-chat"  "gpt-4o" "gpt-4.1-mini" "o4-mini"
        self.model = "deepseek-reasoner" if "deepseek" in api_type else "gpt-4o"
        self.split_method = split_method
        self.stride = stride * 60  # 转换为秒
        self.encoder = tiktoken.encoding_for_model("gpt-4")

        # output tokens is very limited. Different from metadata generation, which outputs much less data.
        max_deepseek_tokens = 8000 #deepseek 
        max_openai_tokens = 32000 #gpt-4 
        self.max_context_tokens = max_deepseek_tokens if "deepseek" in api_type else max_openai_tokens
        
        # 解析原始SRT文件
        self.parser = SrtParser(subtitle_path)
        self.chunks = self._split_into_chunks()
        
    def _init_api(self, api_type):
        if api_type == 'deepseek':
            return OpenAI(
                api_key=os.getenv('DEEPSEEK_API_KEY'),
                base_url="https://api.deepseek.com/v1"
            )
        elif api_type == 'openai':
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        raise ValueError("Unsupported API type")

    def _load_template(self):
        base_dir = Path(__file__).parent
        prompt_path = base_dir / "process_srt.prompt"
        return prompt_path.read_text(encoding="utf-8")

    def _split_into_chunks(self):
        if self.split_method == 'time':
            return self._split_by_time()
        return self._split_by_token()

    def _split_by_time(self):
        chunks = []
        current_start = 0
        max_chunk = self.stride * 1.1

        template = self._load_template()
        template_len = len(self.encoder.encode(template))

        total_tokens = 0
        for seg in self.parser.segments:
            total_tokens += len(self.encoder.encode(seg['text']))    

        chunk_tokens = total_tokens * self.stride / self.parser.total_duration
        print( total_tokens, self.stride, self.parser.total_duration, template_len, self.max_context_tokens)
        # 如果total_tokens并不多，那就用缺省的token分段方式来切分，不会出现超标问题
        if chunk_tokens + template_len > self.max_context_tokens * 0.3:
            print("使用按token分段方式，确保AI API能正确处理。")
            return self._split_by_token()
        

        while current_start < self.parser.total_duration:
            # 修正判断逻辑
            if current_start + max_chunk > self.parser.total_duration:
                chunk_end = self.parser.total_duration
            else:
                chunk_end = current_start + self.stride

            chunk = [
                seg for seg in self.parser.segments
                if seg['start_sec'] >= current_start
                and seg['start_sec'] < chunk_end
            ]
            
            if chunk:
                chunks.append(chunk)
                current_start = chunk[-1]['end_sec']
            else:
                current_start += self.stride

        return chunks

    def _split_by_token(self):
        chunks = []
        current_chunk = []
        current_tokens = 0

        template = self._load_template()
        template_len = len(self.encoder.encode(template))
        
        max_tokens = int((self.max_context_tokens - template_len) * 0.3)

        for idx, seg in enumerate(self.parser.segments):
            seg_tokens = len(self.encoder.encode(seg['text']))
            if current_tokens + seg_tokens > max_tokens:
                chunks.append(current_chunk)
                current_chunk = [seg]
                current_tokens = seg_tokens
            else:
                current_chunk.append(seg)
                current_tokens += seg_tokens
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _process_chunk(self, chunk):
        # 只发送文本内容和索引
        texts = [{"index": idx, "text": seg['text']} for idx, seg in enumerate(chunk)]
        prompt = self._load_template() + "\n" + json.dumps(texts, ensure_ascii=False)
        
        # 调用AI处理
        response = self._ai_process(prompt)
        
        # 解析并更新文本
        return self._parse_response(response, chunk)

    def _ai_process(self, prompt):
        #print(f"Prompt:{prompt}")
        response = self.api_client.chat.completions.create(
            model = self.model,
            messages=[{
                "role": "user", 
                "content": prompt,  
                "temperature": 0.2
            }]
        )
        response = response.choices[0].message.content
        #print(f"Reponse:{response}")
        return response

    def _parse_response(self, response_text, original_chunk):
        try:
            # 去除可能的代码块标记
            cleaned = re.sub(r'^```(json)?\s*|```$', '', response_text, flags=re.MULTILINE)
            refined_data = json.loads(cleaned)
            
            # 按索引更新原始文本
            for item in refined_data:
                idx = item['index']
                original_chunk[idx]['text'] = item['text']
            return original_chunk
        except Exception as e:
            print(f"解析失败: {str(e)}")
            return original_chunk

    def refine(self):
        # 处理所有分块
        for chunk in self.chunks:
            self._process_chunk(chunk)
        
        # 生成新的SRT文件
        self._backup_original()
        self._generate_refined_srt()
        return self.subtitle_path

    def _generate_refined_srt(self):
        subs = pysrt.SubRipFile()
        for idx, seg in enumerate(self.parser.segments):
            item = pysrt.SubRipItem(
                index=idx + 1,  # SRT索引从1开始递增
                start=pysrt.SubRipTime.from_time(seg['start']),
                end=pysrt.SubRipTime.from_time(seg['end']),
                text=seg['text']
            )
            subs.append(item)
        subs.save(self.subtitle_path, encoding='utf-8')

    def _backup_original(self):
        base = self.subtitle_path.stem
        ext = self.subtitle_path.suffix
        parent = self.subtitle_path.parent
        
        counter = 0
        while (backup_path := parent / f"{base}.{counter}{ext}").exists():
            counter += 1
        shutil.copy(self.subtitle_path, backup_path)

class SrtParser:
    def __init__(self, subtitle_path):
        self.subtitle_path = Path(subtitle_path)
        self.segments = self._parse()
        self.total_duration = max(s['end_sec'] for s in self.segments) if self.segments else 0

    def _parse(self):
        subs = pysrt.open(self.subtitle_path)
        return [{
            'start': sub.start.to_time(),
            'end': sub.end.to_time(),
            'text': sub.text.replace('\n', ' ').strip(),
            'start_sec': sub.start.ordinal / 1000.0,
            'end_sec': sub.end.ordinal / 1000.0
        } for sub in subs]
        
class SrtRefiner:
    
    def refine(self, subtitle_path, api_type, split_method, stride):
        
        refiner = YogaSrtRefiner(
            subtitle_path=subtitle_path,
            api_type=api_type,
            split_method=split_method,
            stride=stride
        )
        
        refined_path = refiner.refine()
        print(f"优化完成，新文件已保存至: {refined_path}")        
        return refined_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SRT字幕优化工具')
    parser.add_argument('subtitle', type=str, help='SRT字幕文件路径')
    parser.add_argument('--api', choices=['deepseek', 'openai'], default='deepseek')
    parser.add_argument('--split', choices=['time', 'token'], default='token')
    parser.add_argument('--stride', type=int, default=15, help='分块步长（分钟）')
    
    args = parser.parse_args()
    
    refiner = SrtRefiner()
    refiner.refine(args.subtitle,  args.api, args.split, args.stride)
    
