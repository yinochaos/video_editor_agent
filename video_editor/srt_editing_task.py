import os
import json
from openai import OpenAI
from config.settings import LLMConfig
import ffmpeg
import re
from datetime import timedelta
import tempfile
import glob
from pathlib import Path
from typing import List, Optional

# --- 配置 ---
# 从 config/settings.py 加载配置并转换为 Langroid 格式
config = LLMConfig()

# --- 配置 OpenAI 客户端 ---
client = OpenAI(
    api_key=config.api_key,
    base_url=config.base_url,
    timeout=getattr(config, 'timeout', 120),
)

SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.MP4'}

def _find_video_file(video_stem: str, source_dir: str) -> Optional[str]:
    """Finds a video file matching a stem, checking all supported extensions."""
    for ext in SUPPORTED_VIDEO_FORMATS:
        path = Path(source_dir) / f"{video_stem}{ext}"
        if path.exists():
            return str(path)
    return None

# --- Agent prompts ---
system_message_v1 = """
你是一个专业的美食视频剪辑师，擅长从美食vlog素材中提取精华内容。你的任务是分析视频字幕，识别并保留最有价值的片段，生成高质量的剪辑建议。

输入格式:
video: [视频文件名]
srt_content: [对应的字幕内容]

你需要仔细分析每个字幕片段,按照以下标准筛选:
1. 保留核心内容:
    - 食材介绍和烹饪步骤讲解
    - 关键的美食评价和感受
2. 删减以下内容:
    - 重复的描述和废话等NG台词，尤其是连续重复的NG台词
    - 无关的闲聊和口误
    - 过长的停顿和空白
3. 确保剪辑后的内容:
    - 故事完整,逻辑连贯;保留开场白和结尾总结
    - 节奏紧凑,信息密度高
    - 突出美食亮点和看点

返回格式(严格JSON,不要包含其他文本),按照视频时间顺序排序,确保开头在前、结尾在后:
[
    {
    "name": "视频文件名",
            "srt_index_list": [
              {
        "index": "字幕序号",
        "text": "字幕内容"
        }
    ]
    }
]

注意事项:
1. 只保留最精华的30-40%内容
2. 确保选取的片段前后文连贯
3. 优先保留画面感强、情绪饱满的片段
4. 注意剪辑节奏,避免跳跃感
"""

system_message_v2 = """
你是一个专业的美食视频剪辑师，擅长从美食vlog素材中提取精华内容。你的任务是分析视频字幕，识别并删除所有NG台词，只保留最有价值的内容。

输入格式:
video: [视频文件名]
srt_content: [对应的字幕内容]

你需要仔细分析每个字幕片段,按照以下标准筛选:
1. 保留核心内容:
    - 食材介绍和烹饪步骤讲解
    - 关键的美食评价和感受
    - 开场白和结尾总结
    - 画面感强、情绪饱满的片段

2. 严格删除以下NG内容:
    - 所有重复的台词和描述
    - 口误、错误表达和需要重来的片段
    - 无关的闲聊和题外话
    - 脏话、不文明用语
    - 过长的停顿和空白
    - 任何不专业或影响观看体验的内容

3. 确保剪辑后的内容:
    - 内容完整、逻辑连贯
    - 节奏紧凑、表达准确
    - 突出美食重点

返回格式(严格JSON,不要包含其他文本):
[
    {
    "name": "视频文件名",
    "srt_index_list": [
        {
        "index": "字幕序号",
        "text": "字幕内容"
        }
    ]
    }
]

注意事项:
1. 优先级是删除所有NG内容，确保最终呈现的都是高质量片段
2. 确保删除NG内容后，剩余片段前后连贯
3. 注意剪辑节奏，避免跳跃感
"""

# --- LLM 调用函数 ---
def call_llm(system_message: str, user_message: str) -> str:
    """
    调用 LLM 并返回结果内容
    """
    try:
        completion = client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=config.max_tokens,
            stream=config.stream,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return ""

# --- Task 1: 去除NG内容任务 ---
def remove_ng_content(subtitle_text: str) -> str:
    print("第一步 - 正在调用LLM去除NG内容...")
    return call_llm(system_message_v2, subtitle_text)

# --- Task 2: 最终剪辑编排任务 ---
def final_edit_task(content_text: str) -> str:
    print("第二步 - 正在调用LLM进行最终剪辑编排...")
    final_edit_system_message = """
你是一个专业的视频最终剪辑师，负责将多个素材片段整理合并成一个完整的成品视频。你需要对已经去除NG内容的字幕素材进行重新编排和剪辑优化。

输入格式:
[
    {
        "name": "素材视频文件名",
        "srt_index_list": [
            {
                "index": "字幕序号", 
                "text": "字幕内容"
            }
        ]
    }
]

你的任务是将所有素材片段重新组织编排成一个连贯的故事：

1. 内容重组：
   - 分析所有素材片段的主题和内容
   - 按照故事逻辑重新组织片段顺序
   - 将相似或相关的内容片段合并整理
   - 构建清晰的叙事结构和故事线

2. 剪辑优化：
   - 选择最精彩、最有价值的片段
   - 优化片段之间的衔接和过渡
   - 调整节奏和情感起伏
   - 确保开场、发展、高潮、结尾的完整结构

3. 最终输出要求：
   - 保持每个片段的原始字幕序号不变
   - 确保每个片段内容完整有意义
   - 整体内容紧凑精炼，突出重点

返回格式(严格JSON，不要包含其他文本):
[
    {
        "name": "原素材视频名",
        "srt_index_list": [
            {
                "index": "原字幕序号",
                "text": "字幕内容"
            }
        ]
    }
]

注意事项：
1. 这是素材到成片的创作过程，重点是构建完整的故事
2. 保持原有字幕序号，但可以完全重新排列组合片段顺序
3. 确保故事脉络清晰，情感表达丰富
4. 最终呈现一个引人入胜的完整作品
"""
    return call_llm(final_edit_system_message, content_text)

def parse_llm_response(content: str) -> list:
    """
    解析 LLM 返回的 JSON 格式结果
    """
    try:
        return json.loads(content.split('```json\n')[1].split('```')[0])
    except json.JSONDecodeError:
        print(f"Error decoding JSON from LLM response: {content}")
        return []

def process_subtitle_content(subtitle_dict: dict) -> list:
    """
    第一步：处理字幕内容并去除NG内容
    
    Args:
        subtitle_dict: 字幕内容字典，key是视频名称，value是SRT文本内容
    
    Returns:
        list: 去除NG内容后的剪辑结果列表
    """
    # 将字幕字典转换为字符串格式，便于LLM处理
    subtitle_text = ""
    for video_name in sorted(subtitle_dict.keys()):
        subtitle_text += f"video: {video_name}\nsrt_content:\n{subtitle_dict[video_name]}\n\n"
    
    # 执行第一步：去除NG内容任务
    result_content = remove_ng_content(subtitle_text)
    print("第一步 - 去除NG内容结果:")
    #print(result_content)
    
    return parse_llm_response(result_content)

def final_edit_content(ng_removed_content: list) -> list:
    """
    第二步：对去除NG内容后的结果进行最终剪辑编排
    
    Args:
        ng_removed_content: 第一步去除NG内容后的结果列表
    
    Returns:
        list: 最终剪辑编排后的结果列表
    """
    # 将第一步的结果转换为JSON字符串，便于LLM处理
    content_text = json.dumps(ng_removed_content, ensure_ascii=False, indent=2)
    
    # 执行第二步：最终剪辑编排任务
    result_content = final_edit_task(content_text)
    print("第二步 - 最终剪辑编排结果:")
    print(result_content)
    
    return parse_llm_response(result_content)

def process_video_editing(subtitle_dict: dict) -> tuple[list, list]:
    """
    完整的视频剪辑处理流程：去除NG内容 + 最终剪辑编排
    
    Args:
        subtitle_dict: 字幕内容字典，key是视频名称，value是SRT文本内容
    
    Returns:
        list: 最终剪辑编排后的结果列表
    """
    print("=== 开始视频剪辑处理流程 ===")
    
    # 第一步：去除NG内容
    print("\n--- 第一步：去除NG内容 ---")
    ng_removed_result = process_subtitle_content(subtitle_dict)
    
    # 过滤掉少于3条字幕的结果
    ng_removed_result = [result for result in ng_removed_result if len(result.get('srt_index_list', [])) >= 3]
    print(f"过滤后剩余片段数: {len(ng_removed_result)}")
    for result in ng_removed_result:
        print(f"result: {result}")
    #print(f"ng_removed_result: {ng_removed_result}")
    if not ng_removed_result:
        print("第一步处理失败，无法继续")
        return [], []
    
    # 第二步：最终剪辑编排
    print("\n--- 第二步：最终剪辑编排 ---")
    final_result = final_edit_content(ng_removed_result)
    
    if not final_result:
        print("第二步处理失败，返回第一步结果")
        return ng_removed_result, ng_removed_result
    
    print("\n=== 视频剪辑处理完成 ===")
    return final_result, ng_removed_result


# --- 视频剪辑和SRT生成 ---

def parse_srt_time(time_str: str) -> timedelta:
    """将SRT时间字符串转换为timedelta对象"""
    parts = re.split(r'[:,]', time_str)
    return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]), milliseconds=int(parts[3]))

def format_srt_time(td: timedelta) -> str:
    """将timedelta对象格式化为SRT时间字符串"""
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds / 1000
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{int(milliseconds):03}"

def parse_srt_file(filepath: str) -> list:
    """解析SRT文件，返回字幕条目列表"""
    if not os.path.exists(filepath):
        print(f"Error: SRT file not found at {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entries = []
    blocks = content.strip().split('\n\n')
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                time_parts = lines[1].split(' --> ')
                start = parse_srt_time(time_parts[0])
                end = parse_srt_time(time_parts[1])
                text = '\n'.join(lines[2:])
                entries.append({'index': index, 'start': start, 'end': end, 'text': text})
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse SRT block in {filepath}: {block} - {e}")
    return entries

def generate_editing_plan(final_result: list, source_dir: str) -> tuple[list, list]:
    """
    Generates a video editing plan and final SRT data, keeping original SRT fragments.
    """
    print("Generating video editing plan and final SRT data...")
    
    editing_plan = []
    final_srt_entries = []
    current_timeline_duration = timedelta(0)
    new_srt_index = 1
    
    all_source_srts = {}  # Cache for parsed SRT data

    for video_info in final_result:
        video_name = video_info['name']
        srt_index_list = video_info.get('srt_index_list', [])

        if not srt_index_list:
            continue

        if video_name not in all_source_srts:
            srt_file_path = os.path.join(source_dir, f"{video_name}.srt")
            all_source_srts[video_name] = parse_srt_file(srt_file_path)
        
        source_srt_data = all_source_srts[video_name]
        if not source_srt_data:
            print(f"Warning: No SRT data for {video_name}. Skipping.")
            continue

        # Group consecutive subtitle indices to optimize video cutting
        sorted_clips = sorted(srt_index_list, key=lambda x: int(x['index']))
        
        if not sorted_clips:
            continue

        grouped_clips = []
        current_group = [sorted_clips[0]]

        for i in range(1, len(sorted_clips)):
            prev_index = int(current_group[-1]['index'])
            current_index = int(sorted_clips[i]['index'])
            if current_index == prev_index + 1:
                current_group.append(sorted_clips[i])
            else:
                grouped_clips.append(current_group)
                current_group = [sorted_clips[i]]
        grouped_clips.append(current_group)

        # Process each group of consecutive clips
        for group in grouped_clips:
            start_index_of_group = int(group[0]['index'])
            end_index_of_group = int(group[-1]['index'])

            start_entry_of_group = next((item for item in source_srt_data if item['index'] == start_index_of_group), None)
            end_entry_of_group = next((item for item in source_srt_data if item['index'] == end_index_of_group), None)

            if not start_entry_of_group or not end_entry_of_group:
                print(f"Warning: Could not find start/end SRT entries for group {start_index_of_group}-{end_index_of_group} in {video_name}. Skipping.")
                continue

            start_time_of_group = start_entry_of_group['start']
            end_time_of_group = end_entry_of_group['end']
            
            video_file_path = _find_video_file(video_name, source_dir)
            if not video_file_path:
                 print(f"Error: Video file not found for stem '{video_name}' in directory '{source_dir}'. Skipping.")
                 continue

            # Add one entry to the video editing plan for the entire group
            editing_plan.append({
                'video': video_file_path,
                'start': str(start_time_of_group),
                'end': str(end_time_of_group)
            })

            # Generate individual SRT entries for each clip in the group
            for clip_info in group:
                srt_index = int(clip_info['index'])
                original_srt_entry = next((item for item in source_srt_data if item['index'] == srt_index), None)
                
                if not original_srt_entry:
                    print(f"Warning: Could not find original SRT entry for index {srt_index} in {video_name}. Skipping SRT entry.")
                    continue

                original_clip_start_time = original_srt_entry['start']
                original_clip_end_time = original_srt_entry['end']
                
                # Calculate start time of this clip relative to the start of the merged group clip
                relative_start_offset = original_clip_start_time - start_time_of_group
                
                # Duration of the original clip
                clip_duration = original_clip_end_time - original_clip_start_time

                # Calculate new timestamps for the final timeline
                new_start_time = current_timeline_duration + relative_start_offset
                new_end_time = new_start_time + clip_duration

                final_srt_entries.append({
                    'index': new_srt_index,
                    'start': new_start_time,
                    'end': new_end_time,
                    'text': clip_info['text']
                })
                new_srt_index += 1
            
            # Update the total timeline duration by the duration of the entire group
            group_duration = end_time_of_group - start_time_of_group
            current_timeline_duration += group_duration

    return editing_plan, final_srt_entries

def execute_video_editing(editing_plan: list, output_video_path: str):
    """
    Executes the video editing plan to cut and concatenate clips.
    """
    print("Executing video editing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        clips_to_concat = []
        for i, clip_info in enumerate(editing_plan):
            clip_path = os.path.join(temp_dir, f"clip_{i}.mp4")
            print(f"Cutting clip {i+1}/{len(editing_plan)} from {clip_info['video']}")
            try:
                (
                    ffmpeg
                    .input(clip_info['video'], ss=clip_info['start'], to=clip_info['end'])
                    .output(clip_path, preset='fast', vcodec='libx264', acodec='aac')
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
                clips_to_concat.append(clip_path)
            except Exception as e:
                print(f"Error cutting video {clip_info['video']}: {e}")
                # The ffmpeg-python library often puts detailed errors in stderr
                if hasattr(e, 'stderr') and e.stderr:
                    print(f"FFmpeg stderr: {e.stderr.decode()}")
                continue
        
        if not clips_to_concat:
            print("No video clips were generated to concatenate.")
            return

        print("Concatenating all video clips...")
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for clip_path in clips_to_concat:
                f.write(f"file '{os.path.abspath(clip_path)}'\n")

        try:
            (
                ffmpeg
                .input(concat_list_path, format='concat', safe=0)
                .output(output_video_path, vcodec='libx264', acodec='aac')
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            print(f"Final video saved to: {output_video_path}")
        except Exception as e:
            print(f"Error concatenating videos: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"FFmpeg stderr: {e.stderr.decode()}")

def write_srt_file(srt_entries: list, output_srt_path: str):
    """
    Writes the final SRT entries to a file.
    """
    print(f"Writing final SRT file to {output_srt_path}...")
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for entry in srt_entries:
            f.write(f"{entry['index']}\n")
            f.write(f"{format_srt_time(entry['start'])} --> {format_srt_time(entry['end'])}\n")
            f.write(f"{entry['text']}\n\n")
    print("SRT file written successfully.")


def run_editing_workflow(srt_files: List[str], source_directory: str) -> tuple[list, list, list, list]:
    """
    Runs the data processing part of the editing workflow.
    
    Reads SRTs, processes them with an LLM to get editing decisions,
    and generates a video editing plan.
    
    Args:
        srt_files (List[str]): A list of paths to the SRT files.
        source_directory (str): The directory containing the source video files.

    Returns:
        A tuple containing:
        - final_result (list): The final editing decisions from the LLM.
        - ng_removal_results (list): The intermediate results after removing NG content.
        - editing_plan (list): The detailed plan for video cutting.
        - final_srt_entries (list): The data for the final SRT file.
    """
    print(f"--- Starting data processing for {len(srt_files)} SRT file(s) ---")

    # --- Step 1: Read SRT files ---
    print(f"\n--- Step 1: Reading {len(srt_files)} SRT files ---")
    if not srt_files:
        print("No SRT files provided. Exiting.")
        return [], [], [], []
        
    sample_subtitles = {}
    for srt_file in srt_files:
        video_name = Path(srt_file).stem
        try:
            with open(srt_file, 'r', encoding='utf-8') as f:
                srt_content = f.read()
                if len(srt_content.splitlines()) >= 12:
                    content_lines = srt_content.splitlines()
                    filtered_lines = [line for line in content_lines if '-->' not in line]
                    sample_subtitles[video_name] = '\n'.join(filtered_lines)
                else:
                    print(f"Skipping {srt_file} - less than 12 lines")
        except Exception as e:
            print(f"Error reading {srt_file}: {e}")
    
    if not sample_subtitles:
        print("No valid subtitle files could be read. Exiting.")
        return [], [], [], []

    # --- Step 2: Run LLM processing ---
    print("\n--- Step 2: Running LLM processing for editing decisions ---")
    final_result, ng_removal_results = process_video_editing(sample_subtitles)
    
    # --- Step 3: Generate editing plan from LLM result ---
    editing_plan, final_srt_entries = [], []
    if final_result:
        print("\n--- Step 3: Generating editing plan from LLM results ---")
        editing_plan, final_srt_entries = generate_editing_plan(final_result, source_directory)
    else:
        print("\nNo final result from LLM, skipping plan generation.")
        
    print("\n--- Data processing finished ---")
    return final_result, ng_removal_results, editing_plan, final_srt_entries


# =====================================================================================
# == Functions for Service Integration
# =====================================================================================

def parse_srt_string(srt_content: str) -> list:
    """Parses an SRT string and returns a list of subtitle entries."""
    entries = []
    blocks = srt_content.strip().split('\n\n')
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                time_parts = lines[1].split(' --> ')
                start = parse_srt_time(time_parts[0])
                end = parse_srt_time(time_parts[1])
                text = '\n'.join(lines[2:])
                entries.append({'index': index, 'start': start, 'end': end, 'text': text})
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse SRT block from string: {block} - {e}")
    return entries

def generate_editing_plan_from_content(final_result: list, original_srts_map: dict) -> tuple[list, list]:
    """
    Generates a video editing plan from content, not files.
    `original_srts_map` is a dict of {video_name: parsed_srt_data}
    """
    print("Generating video editing plan from content...")
    
    editing_plan = []
    final_srt_entries = []
    current_timeline_duration = timedelta(0)
    new_srt_index = 1
    
    for video_info in final_result:
        video_name = video_info['name']
        srt_index_list = video_info.get('srt_index_list', [])

        if not srt_index_list:
            continue
        
        source_srt_data = original_srts_map.get(video_name)
        if not source_srt_data:
            print(f"Warning: No source SRT data provided for {video_name}. Skipping.")
            continue

        sorted_clips = sorted(srt_index_list, key=lambda x: int(x['index']))
        if not sorted_clips:
            continue

        grouped_clips = []
        current_group = [sorted_clips[0]]

        for i in range(1, len(sorted_clips)):
            if int(sorted_clips[i]['index']) == int(current_group[-1]['index']) + 1:
                current_group.append(sorted_clips[i])
            else:
                grouped_clips.append(current_group)
                current_group = [sorted_clips[i]]
        grouped_clips.append(current_group)

        for group in grouped_clips:
            start_index = int(group[0]['index'])
            end_index = int(group[-1]['index'])

            start_entry = next((item for item in source_srt_data if item['index'] == start_index), None)
            end_entry = next((item for item in source_srt_data if item['index'] == end_index), None)

            if not start_entry or not end_entry:
                continue

            start_time_of_group = start_entry['start']
            end_time_of_group = end_entry['end']
            
            editing_plan.append({
                'video_name': video_name,
                'start': start_time_of_group.total_seconds(),
                'end': end_time_of_group.total_seconds()
            })

            for clip_info in group:
                srt_index = int(clip_info['index'])
                original_entry = next((item for item in source_srt_data if item['index'] == srt_index), None)
                if not original_entry:
                    continue

                relative_start = original_entry['start'] - start_time_of_group
                clip_duration = original_entry['end'] - original_entry['start']
                
                new_start_time = current_timeline_duration + relative_start
                new_end_time = new_start_time + clip_duration

                final_srt_entries.append({
                    'index': new_srt_index,
                    'start': new_start_time,
                    'end': new_end_time,
                    'text': clip_info['text']
                })
                new_srt_index += 1
            
            group_duration = end_time_of_group - start_time_of_group
            current_timeline_duration += group_duration

    return editing_plan, final_srt_entries

def run_editing_from_content(srt_pairs: list) -> tuple[list, list]:
    """
    Main entry point for service. Takes srt content and returns editing data.
    `srt_pairs` is a list of {"video_name": str, "srt_content": str}
    """
    print(f"--- Starting editing workflow from content for {len(srt_pairs)} videos ---")

    # Step 1: Prepare data for LLM and for timestamp lookup
    subtitle_dict_for_llm = {}
    original_srts_map = {}
    for pair in srt_pairs:
        video_name = pair['video_name']
        srt_content = pair['srt_content']
        
        # For LLM: strip timestamps
        content_lines = srt_content.splitlines()
        filtered_lines = [line for line in content_lines if '-->' not in line]
        subtitle_dict_for_llm[video_name] = '\n'.join(filtered_lines)
        
        # For timestamp lookup: parse full SRT content
        original_srts_map[video_name] = parse_srt_string(srt_content)

    if not subtitle_dict_for_llm:
        print("No valid subtitle content provided. Exiting.")
        return [], []

    # Step 2: Run LLM processing
    final_result, _ = process_video_editing(subtitle_dict_for_llm)
    
    # Step 3: Generate editing plan
    editing_plan, final_srt_entries = [], []
    if final_result:
        editing_plan, final_srt_entries = generate_editing_plan_from_content(final_result, original_srts_map)
    
    print("\n--- Content editing workflow finished ---")
    return editing_plan, final_srt_entries


# 使用示例
if __name__ == "__main__":
    source_dir = "tmp/kaoya"
    output_dir = "tmp/kaoya" # Can be set to a different directory

    # Run the data processing workflow to get the editing plan and assets
    final_result, ng_removal_results, editing_plan, final_srt_entries = run_editing_workflow(source_directory=source_dir)

    # Proceed with file I/O and video generation if the workflow was successful
    if final_result and editing_plan:
        os.makedirs(output_dir, exist_ok=True)
        
        # --- Save LLM and plan results ---
        print(f"\n--- Saving all results to {output_dir} ---")
        
        # Save final LLM result
        try:
            output_file = os.path.join(output_dir, "final_editing_results.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            print(f"Final editing results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving final results: {e}")
        
        # Save NG removal backup
        try:
            backup_file = os.path.join(output_dir, "ng_removal_results.json")
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(ng_removal_results, f, ensure_ascii=False, indent=2)
            print(f"NG removal results backup saved to: {backup_file}")
        except Exception as e:
            print(f"Error saving backup results: {e}")

        # Save editing plan
        try:
            editing_plan_path = os.path.join(output_dir, "editing_plan.json")
            with open(editing_plan_path, 'w', encoding='utf-8') as f:
                json.dump(editing_plan, f, ensure_ascii=False, indent=2)
            print(f"Editing plan saved to: {editing_plan_path}")
        except Exception as e:
            print(f"Error saving editing plan: {e}")

        # --- Generate final SRT and video ---
        print(f"\n--- Generating final video and assets in {output_dir} ---")
        
        # Write final SRT file
        final_srt_path = os.path.join(output_dir, "final_cut.srt")
        write_srt_file(final_srt_entries, final_srt_path)
        
        # Execute video editing (currently commented out)
        final_video_path = os.path.join(output_dir, "final_cut.mp4")
        # execute_video_editing(editing_plan, final_video_path)

        print("\n--- Main workflow finished ---")
    else:
        print("\nWorkflow finished with no results to process for video generation.")
