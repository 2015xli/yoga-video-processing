## Yoga video processing 

This is an AI task pipeline to process the input video files that are Yoga training classes. Each task can execute independently if used separately. 
PC reads video files from a smartphone to a video server. Server iterates the given input directory tree to identify the uploaded video files, then compress the video, transcribe to srt, refine the srt, generate metadata for the video. 

**Tasks in the pipeline:**

1. move-mtp-mp4-windows.ps1: 
	- find and move video files from USB devices to server. Run on PC when USB is plugged.
2. compress_transcribe_metadata.py [Explanatioin](#Driver-Compress-transcribe-metadata) 
	- the entry driver for all tasks: compress the video, transcribe to srt, refine the srt, generate metadata.
3. video_compress.py [Explanatioin](#Compress-video)
	- compress the video with H.265.
4. video_transcribe.py [Explanatioin](#Transcribe-video) 
	- transcribe to subtitle srt file with local AI models.
	- (with built-in short initial prompt)
5. srt_refine.py [Explanatioin](#Refine-SRT-subtitles)
	- refine the srt file subtitles by correcting the wordings with remote AI models.
	- (prompt file: process_srt.prompt)
6. video_metadata.py [Explanatioin](#Generate-video-metadata)
	- summarize the srt file with remote AI models, generate metadata and embed into the video files.
	- (prompt files:  process_chunk.prompt, process_chapters.prompt, process_summary.prompt)
7. (Optional) video_transcribe_openai_api.py [Explanatioin](#transcribe-with-openAI-whisper)
	- The same functionality as video_transcribe.py, but use remote API.

---
### Driver: Compress-transcribe-metadata

**Input arguments:**

* Input root directory: for original video files,
* Temp root directory: for intermediate files, such as the compressed video files, extracted audio files, generated srt files, metadata files, etc. The temp root dir is used as the output dir of video_compress.py, and as the temp dir of video_transcribe.py, as the working dir of srt_refine.py and video_metadata.py.
* Output root directory: for final results, including the compressed video file with embedded metadata, generated srt file, also the summary and metadata files just for fun.
* The directory tree structures of the three dirs are the same.
* Options
	* --skip-compression, the video files are compressed already. Skip the compression step. Use the input root directory as the temp root dir for the compressed videos to transcribe and embed metadata. If the output dir is the same as the input dir, no need to move them to the output root dir.
	* --erase-original is to remove the original video. default is false.
	* --erase-temp is to remove the intermediate files. default is false.
	* --model is to specify faster-whisper or openai-whisper for ASR. default is faster-whisper.
	* --api is to specify openai or deepseek for summarization. default is deepseek.
	* --split to choose how to split the subtitle into chunks, in time or token. default is time.
	* --stride to specify the STRIDE value of a time-split chunk (only valid when split is time). default is 15.
	* --overlap to specify the OVERLAP value between two chunks (only valid when split is time). default is 5.

**Functionality**

Iterate the input root directory to find video files, then process them one after another by invoking the pipeline tasks.

---
### Compress video

Compress a video file with H.265 encoder. 

**Input arguments:** 
* The path of input video file.
* The output directory for the result compressed file.
* Options
	* --erase-original to indicate if the original video file is removed once compression is done.

**Functionality:**

If the input video file is not H.265 encoded, then compress it with H.265 encoder. Use hardware acceleration hevc_nvenc on host xf-workstation; or software encoder libx265 on host XF-NAS and other hosts. The resulted file has "x265" inserted in the original filename right before the suffix. Conduct the compression in the output directory. Remove the original video files if the compression is finished successfully, and the option --erase-original is optional.

---
### Transcribe video

Transcribe a video file to srt subtitle file. 

**Input arguments:**

* The path of the input video file.
* A temp directory for intermediate files like the extracted audio or srt file(s)
* Options
	* --model to choose either faster-whisper or openai-whisper for ASR, default is faster-whisper.

**Functionality:**

Extract the audio of the input video into the temporary dir, and then use faster-whisper or openai-whisper package to transcribe the audio into subtitle srt file. Both original and target languages are Chinese. Most of the audio is yoga exercise instructions given by the Yoga teacher. An initial prompt is given to the model to assist the transcription. Keep the subtitle file basename same as the video file basename. The temp files are removed once everything is done. 

---
### Refine SRT subtitles

Refine the srt subtitles by correcting the wordings.

**Input arguments:**

* The srt file.
* Options
	* --api to choose either deepseek or openai.
	* --split to choose how to split the subtitle into chunks, in time duratipon or in token size, ensuring chunk size is within api rate limit.
	* --stride to specify the STRIDE value of a time-split chunk 

**Functinality:**

Correct the subtitles with AI model. ASR in video transcription may generate lots of minor errors. A reasonable prompt can help a lot, especially for those Yoga gestures.
The subtitles are split into chunks according to the max context token numbers of deepseek and openai models. Although the input token size is not really big (~64KB for one hour video), the output token size (the same as the input) can exceed the limits which are usually much smaller than the input size. The splitting does not need to worry about that one line of subtitle is broken across the chunk boundary, since the subtitles are organized as segments internally and the split is conducted in segment granularity.

---
### Generate video metadata

Summarize the yoga training class content into video file metadata.

**Input arguments:**

* A subtitle file.
* The video of the subtitle.
* Options
	* --api to choose either deepseek or openai.
	* --split to choose how to split the subtitle into chunks, in time duratipon or in token size, ensuring chunk size is within api rate limit.
	* --stride to specify the STRIDE value of a time-split chunk 
	* --overlap to specify the OVERLAP value of overlapping duration between two chunks 

**Functionality:**

Partition the whole yago class into chunks according to the option of split, and summarize every chunk with a summary and chapters (each chapter has a start time and main postures). Then summarize the whole-class description based on the chunk summaries. The chapters of different chunks may overlap, so the script merge the overlapped chapers. The summarization is done by calling with deepseek API or OpenAI API. The API keys are given in environment like DEEPSEEK_API_KEY and OPENAI_API_KEY. Then the description and chapters are converted to video metadata file and written into the video file, together with other metadata like artist, creation time, title, etc.

A Yoga pose may span a few minutes, so chunk split should avoid breaking one pose into two chunks. Overlapping is used for neighbor chunks. In this way, if a pose is broken to two chunks at the end of a chunk, it should be covered completely by its next neighbor chunk. The default overlapping duration is 5min, which is long enough to cover a posture that is broken at the boundary of 5min later.  After both chunks have been processed separately, their resulted chapters (recognized Yoga poses) will be processed again by AI to figure out the right ones. Sometimes Yoga exercise has not just poses, but also flow of poses, which are more easily crossing the boundaries. Second pass of AI processing on all the chunks results is important here. It is amazing that AI can often figure out flows and even count their repeat times.

```
chunk len = stride + overlap 
  stide overlap 
0 -----|--
      1 -----|--
            2 -----|--
                  3 -----|----
                          remaining 
           (chunk_len * 0.5 + overlap) <= remaining <= (chunk_len * 1.5)
```

If split is by token, and the API max context token number is large enough for the srt file, then the code falls back to use time based split, in order to have smaller chunk size. Smaller chunk size makes the posture recognition more accurate. 

### Transcribe with OpenAI API to Whisper

Previous video_transcribe.py uses local models (faster-whisper or openai-whisper) that has a benefit that the audio file is processed iterateively and automatically till finish. video_transcribe_openai_api.py uses cloud OpenAI API for the transcription. The problem with this approach is, it can only process less than ~20MB audio file. 
The code splits the audio file into segments with overlapping duration to avoid broken sentences, transcribes the segments one by one, and then merge the resulted segments of subtitles.
        
**The input arguments:**

* The video file to transcribe
* Temp dir for itermediate files like audio file and segments, srt segments, etc.
* Options
	* --stride, Segment stride in minutes (default: 20). This is short enough to fit into OpenAI API.  
	* --overlap, Segment overlapping duration (default: 2). One minute after the boundary is long enough to finish the last sentence. We only take a subtitle line after half_ov

The merge of subtitles is simply to take the subtitle lines that start after base + half_overlap, and end before base + length - half_overlap.

        # Voice may cross the boundary of segments, so when we take a subtitle line, 
        # we should not start after the boundary; instead, we start after half_ov.
        # Then for every sub line, take it in the final srt if,
        # 1. it is in the first segment, and starts before tail_thr (i.e., length - half_ov)
        # 2. it is in a mid segment, starts after head_thr and before tail_thr
        # 3. it is in the last segment, starts after head_thr. 
        # Conditions should be
        #  if (idx == 0 and st <= tail_thr) or
        #     (0 < idx < last_idx and head_thr < st <= tail_thr) or
        #     (idx == last_idx and st > head_thr):

**Functionalities:**

For remote API access, we now extract 128k aac audio to m4a container, instead of wav audio to save bandwidth.
