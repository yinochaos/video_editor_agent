# -*- coding: utf-8 -*-
"""
ASR服务客户端
支持MP3/MP4文件输入，自动转换MP4为MP3，输出SRT文件
"""

import os
import sys
import asyncio
import tempfile
import subprocess
import logging
import json
from pathlib import Path
from typing import Optional, Union, List
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASRClient:
    """ASR服务客户端"""

    def __init__(self, base_url: str = "https://ms-nfp2p2tq-100039220333-sw.gw.ap-beijing.ti.tencentcs.com/ms-nfp2p2tq"):
        """
        初始化ASR客户端

        Args:
            base_url: ASR服务地址
        """
        self.base_url = base_url.rstrip('/')
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.MP4'}
        self.supported_audio_formats = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.MP3'}
        self.auth_token = "3a2838109a4dae7"

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: 服务是否正常
        """
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            response = requests.get(f"{self.base_url}/health", headers=headers, timeout=10)
            if response.status_code == 200:
                logger.info("ASR服务健康检查通过")
                return True
            else:
                logger.error(f"ASR服务健康检查失败: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"ASR服务连接失败: {e}")
            return False

    def convert_video_to_mp3(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        将视频文件转换为MP3音频文件

        Args:
            video_path: 视频文件路径
            output_path: 输出MP3文件路径，如果为None则自动生成

        Returns:
            str: 输出MP3文件路径
        """
        if output_path is None:
            # 生成临时MP3文件路径
            video_name = Path(video_path).stem
            output_path = os.path.join(tempfile.gettempdir(), f"{video_name}_converted.mp3")

        # FFmpeg转换命令
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',  # 不包含视频
            '-ar', '16000',  # 采样率16kHz
            '-ac', '1',  # 单声道
            '-b:a', '32k',  # 比特率32k
            '-y',  # 覆盖输出文件
            output_path
        ]

        logger.info(f"开始转换视频: {video_path} -> {output_path}")
        logger.info(f"转换命令: {' '.join(cmd)}")

        try:
            # 执行转换
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode == 0:
                logger.info(f"视频转换成功: {output_path}")
                return output_path
            else:
                error_msg = f"视频转换失败: {result.stderr}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        except subprocess.TimeoutExpired:
            error_msg = "视频转换超时"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"视频转换异常: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def get_file_info(self, file_path: str) -> dict:
        """
        获取文件信息

        Args:
            file_path: 文件路径

        Returns:
            dict: 文件信息
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 获取文件大小
        size = file_path.stat().st_size

        # 判断文件类型
        suffix = file_path.suffix.lower()
        if suffix in self.supported_video_formats:
            file_type = "video"
        elif suffix in self.supported_audio_formats:
            file_type = "audio"
        else:
            file_type = "unknown"

        return {
            "path": str(file_path),
            "name": file_path.name,
            "size": size,
            "size_mb": size / (1024 * 1024),
            "type": file_type,
            "suffix": suffix
        }

    def recognize_speech(self, file_path: str, language: str = "zh",
                        output_srt: Optional[str] = None) -> dict:
        """
        语音识别主函数

        Args:
            file_path: 输入文件路径（MP3/MP4）
            language: 语言类型 (zh/en)
            output_srt: 输出SRT文件路径，如果为None则自动生成

        Returns:
            dict: 识别结果
        """
        # 获取文件信息
        file_info = self.get_file_info(file_path)
        logger.info(f"处理文件: {file_info}")

        # 检查文件类型
        if file_info["type"] == "unknown":
            raise ValueError(f"不支持的文件格式: {file_info['suffix']}")

        # 如果是视频文件，先转换为MP3
        audio_file_path = file_path
        temp_audio_file = None

        try:
            if file_info["type"] == "video":
                logger.info("检测到视频文件，开始转换为MP3...")
                temp_audio_file = self.convert_video_to_mp3(file_path)
                audio_file_path = temp_audio_file

            # 执行语音识别
            logger.info(f"开始语音识别，语言: {language}")
            result = self._call_asr_service(audio_file_path, language)

            # 保存SRT文件
            if result.get("success") and result.get("srt_content"):
                srt_content = result["srt_content"]

                if output_srt is None:
                    # 自动生成SRT文件路径
                    base_name = Path(file_path).stem
                    output_srt = f"{base_name}.srt"

                self._save_srt_file(srt_content, output_srt)
                result["srt_file"] = output_srt
                logger.info(f"SRT文件已保存: {output_srt}")

            return result

        finally:
            # 清理临时文件
            if temp_audio_file and os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
                logger.info(f"清理临时文件: {temp_audio_file}")

    def _call_asr_service(self, audio_file_path: str, language: str) -> dict:
        """
        调用ASR服务

        Args:
            audio_file_path: 音频文件路径
            language: 语言类型

        Returns:
            dict: 服务响应结果
        """
        try:
            with open(audio_file_path, 'rb') as f:
                files = {'audio_file': f}
                data = {'language': language}
                headers = {"Authorization": f"Bearer {self.auth_token}"}

                logger.info(f"发送请求到ASR服务: {self.base_url}/asr/recognize")
                response = requests.post(
                    f"{self.base_url}/asr/recognize",
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=300  # 5分钟超时
                )

                response.raise_for_status()
                result = response.json()

                if result.get("success"):
                    logger.info("ASR识别成功")
                else:
                    logger.error(f"ASR识别失败: {result.get('error_message')}")

                return result

        except requests.exceptions.RequestException as e:
            error_msg = f"ASR服务请求失败: {e}"
            logger.error(error_msg)
            return {"success": False, "error_message": error_msg}
        except Exception as e:
            error_msg = f"ASR服务调用异常: {e}"
            logger.error(error_msg)
            return {"success": False, "error_message": error_msg}

    def _save_srt_file(self, srt_content: str, output_path: str):
        """
        保存SRT文件

        Args:
            srt_content: SRT内容
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            logger.info(f"SRT文件保存成功: {output_path}")
        except Exception as e:
            error_msg = f"保存SRT文件失败: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def intelligent_cut(self, srt_files: List[str], output_plan_path: str = "editing_plan.json", output_srt_path: str = "final_cut.srt") -> dict:
        """
        Calls the intelligent editing service to get an editing plan and a final SRT.

        Args:
            srt_files: A list of paths to the source SRT files.
            output_plan_path: The file path to save the generated video editing plan.
            output_srt_path: The file path to save the final generated SRT file.

        Returns:
            A dictionary containing the server's response.
        """
        logger.info("Starting intelligent editing task...")

        srt_pairs = []
        for srt_path in srt_files:
            try:
                video_name = Path(srt_path).stem
                with open(srt_path, 'r', encoding='utf-8') as f:
                    srt_content = f.read()
                srt_pairs.append({"video_name": video_name, "srt_content": srt_content})
                logger.info(f"Successfully read SRT file: {srt_path}")
            except Exception as e:
                logger.error(f"Could not read SRT file {srt_path}: {e}")
                continue # Skip this file and continue with others

        if not srt_pairs:
            error_msg = "No valid SRT files could be read."
            logger.error(error_msg)
            return {"success": False, "error_message": error_msg}

        request_body = {"srt_pairs": srt_pairs}
        endpoint = f"{self.base_url}/editing/intelligent_cut"
        headers = {"Authorization": f"Bearer {self.auth_token}"}

        try:
            logger.info(f"Sending request to: {endpoint}")
            response = requests.post(
                endpoint,
                json=request_body,
                headers=headers,
                timeout=600  # 10 minute timeout for LLM processing
            )
            response.raise_for_status()
            result = response.json()
            logger.info(f"Response from server: {result}")

            if result.get("success"):
                logger.info("Intelligent editing task completed successfully.")
                # Save the results to local files
                if result.get("video_part_list"):
                    try:
                        with open(output_plan_path, 'w', encoding='utf-8') as f:
                            json.dump(result["video_part_list"], f, ensure_ascii=False, indent=2)
                        logger.info(f"Editing plan saved to: {output_plan_path}")
                    except Exception as e:
                        logger.error(f"Failed to save editing plan: {e}")

                if result.get("srt_list"):
                    try:
                        srt_content = ""
                        for i, entry in enumerate(result["srt_list"]):
                            srt_content += f"{i + 1}\n"
                            srt_content += f"{entry['start']} --> {entry['end']}\n"
                            srt_content += f"{entry['text']}\n\n"
                        self._save_srt_file(srt_content, output_srt_path)
                    except Exception as e:
                        logger.error(f"Failed to save final SRT file: {e}")
            else:
                logger.error(f"Intelligent editing failed on server: {result.get('error_message')}")

            return result

        except requests.exceptions.RequestException as e:
            error_msg = f"Request to intelligent editing service failed: {e}"
            logger.error(error_msg)
            return {"success": False, "error_message": error_msg}
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            logger.error(error_msg)
            return {"success": False, "error_message": error_msg}


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="ASR识别与智能剪辑客户端")
    parser.add_argument("input_file", nargs='?', default=None, help="输入文件路径（MP3/MP4），ASR任务需要")
    parser.add_argument("-l", "--language", default="zh", choices=["zh", "en"],
                       help="语言类型 (zh/en)")
    parser.add_argument("-o", "--output", help="输出SRT文件路径 (ASR) 或输出文件名前缀 (智能剪辑)")
    parser.add_argument("-u", "--url", default="https://ms-nfp2p2tq-100039220333-sw.gw.ap-beijing.ti.tencentcs.com/ms-nfp2p2tq",
                       help="服务地址")
    parser.add_argument("--check-only", action="store_true",
                       help="仅检查服务状态")
    parser.add_argument("--intelligent-cut", nargs='+', help="智能剪辑任务，后跟一个或多个SRT文件路径")

    args = parser.parse_args()

    # 创建客户端
    client = ASRClient(args.url)

    # 检查服务状态
    if not client.health_check():
        logger.error("服务不可用，请检查服务是否启动")
        sys.exit(1)

    if args.check_only:
        logger.info("服务状态检查完成")
        return

    # --- Handle Intelligent Cutting Task ---
    if args.intelligent_cut:
        logger.info("开始智能剪辑任务...")
        output_prefix = args.output if args.output else "final_cut"
        plan_path = f"{output_prefix}_plan.json"
        srt_path = f"{output_prefix}.srt"

        result = client.intelligent_cut(
            srt_files=args.intelligent_cut,
            output_plan_path=plan_path,
            output_srt_path=srt_path
        )
        if result.get("success"):
            logger.info("智能剪辑流程成功完成。")
            print(f"Editing plan saved to: {plan_path}")
            print(f"Final SRT saved to: {srt_path}")
        else:
            logger.error(f"智能剪辑流程失败: {result.get('error_message')}")
            sys.exit(1)
        return

    # --- Handle ASR Task ---
    if not args.input_file:
        parser.error("ASR任务需要一个输入文件。请提供 input_file 参数或使用 --intelligent-cut 任务。")

    # 执行语音识别
    try:
        result = client.recognize_speech(
            file_path=args.input_file,
            language=args.language,
            output_srt=args.output
        )

        if result.get("success"):
            logger.info("语音识别完成")
            if result.get("srt_file"):
                print(f"SRT文件已生成: {result['srt_file']}")
        else:
            logger.error(f"语音识别失败: {result.get('error_message')}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
