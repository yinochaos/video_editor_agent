import os
import glob
import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from asr_client import ASRClient
from video_editor.srt_editing_task import run_editing_workflow, execute_video_editing, write_srt_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoCutoffAgent:
    """
    An agent for semi-automated video cutting using ASR and LLM-based editing plans.
    """
    def __init__(self, video_dir: str):
        """
        Initializes the agent.

        Args:
            video_dir: The directory containing the source video files.
        """
        if not os.path.isdir(video_dir):
            logger.error(f"Directory not found: {video_dir}")
            raise ValueError(f"Directory not found: {video_dir}")
        self.video_dir = os.path.abspath(video_dir)
        self.asr_client = ASRClient()
        self.editing_plan: Optional[List[Dict[str, Any]]] = None
        self.final_srt_entries: Optional[List[Dict[str, Any]]] = None
        
        # Data store for video and SRT file paths
        self.video_data: Dict[str, Dict[str, Optional[str]]] = {}
        self._scan_video_directory()

    def _scan_video_directory(self):
        """Scans the video directory to populate video and SRT file data."""
        logger.info("Scanning video directory for video and SRT files...")
        self.video_data = {}
        video_files = []
        for ext in self.asr_client.supported_video_formats:
            video_files.extend(glob.glob(os.path.join(self.video_dir, f"*{ext}")))

        for video_path in video_files:
            stem = Path(video_path).stem
            srt_path = os.path.join(self.video_dir, f"{stem}.srt")
            self.video_data[stem] = {
                "video_path": video_path,
                "srt_path": srt_path if os.path.exists(srt_path) else None
            }
        logger.info(f"Scan complete. Found {len(self.video_data)} videos.")
    
    def _find_video_file_for_stem(self, stem: str) -> Optional[str]:
        """Finds a video file matching a stem from the internal state."""
        if stem in self.video_data:
            return self.video_data[stem].get("video_path")
        return None

    def run_asr_on_videos(self):
        """
        Runs ASR on all video files that do not have a corresponding SRT file.
        """
        logger.info("--- Starting ASR processing ---")
        #if not self.asr_client.health_check():
        #    logger.error("ASR service is not available. Please check the service.")
        #    return

        videos_to_process = [
            data for data in self.video_data.values() if data and data.get("srt_path") is None
        ]

        if not videos_to_process:
            logger.info("All videos already have SRT files. No ASR processing needed.")
            return

        logger.info(f"Found {len(videos_to_process)} video(s) to process for ASR.")
        for video_info in videos_to_process:
            video_path = video_info["video_path"]
            if not video_path:
                continue
            
            video_name = Path(video_path).stem
            srt_path = os.path.join(self.video_dir, f"{video_name}.srt")

            logger.info(f"Running ASR on: {video_path}")
            try:
                result = self.asr_client.recognize_speech(video_path, output_srt=srt_path)
                if result.get("success"):
                    logger.info(f"Successfully generated SRT file: {srt_path}")
                    # Update internal state
                    if video_name in self.video_data:
                        self.video_data[video_name]["srt_path"] = srt_path
                else:
                    logger.error(f"ASR failed for {video_path}: {result.get('error_message')}")
            except Exception as e:
                logger.error(f"An exception occurred during ASR for {video_path}: {e}")
        logger.info(f'video_data: {self.video_data}')
        logger.info("--- ASR processing finished ---")

    def generate_editing_plan(self):
        """
        Generates a video editing plan by analyzing the SRT files in the directory.
        The plan and final SRT data are stored in the agent's state.
        """
        logger.info("--- Starting editing plan generation ---")
        
        srt_files_to_process = [
            data["srt_path"]
            for data in self.video_data.values()
            if data and data.get("srt_path")
        ]

        if not srt_files_to_process:
            logger.error("No SRT files found to generate an editing plan. Please run ASR first.")
            return

        # run_editing_workflow now takes a list of srt files
        final_result, _, editing_plan, final_srt_entries = run_editing_workflow(
            srt_files=srt_files_to_process,
            source_directory=self.video_dir
        )

        if not editing_plan or not final_srt_entries:
            logger.error("Failed to generate an editing plan. The LLM may not have returned valid results.")
            return

        self.editing_plan = editing_plan
        self.final_srt_entries = final_srt_entries

        # Save artifacts for inspection
        plan_path = os.path.join(self.video_dir, "editing_plan.json")
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(self.editing_plan, f, ensure_ascii=False, indent=2)
        
        final_llm_result_path = os.path.join(self.video_dir, "final_llm_result.json")
        with open(final_llm_result_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        logger.info(f"Successfully generated editing plan with {len(self.editing_plan)} clips.")
        logger.info(f"Editing plan saved to: {plan_path}")
        logger.info(f"Final LLM result saved to: {final_llm_result_path}")
        logger.info("--- Editing plan generation finished ---")

    def execute_video_cut(self):
        """
        Executes the stored editing plan to generate the final video and SRT file.
        """
        logger.info("--- Starting video execution ---")
        if not self.editing_plan or not self.final_srt_entries:
            logger.error("No editing plan available. Please run step 2 (Generate editing plan) first.")
            return

        # The plan generated by srt_editing_task might have video filenames without extensions.
        # This part corrects the paths to ensure ffmpeg can find the files.
        corrected_plan = []
        for clip in self.editing_plan:
            video_stem = Path(clip['video']).stem
            video_path = self._find_video_file_for_stem(video_stem)
            
            if not video_path:
                logger.warning(f"Could not find a matching video file for '{video_stem}'. Skipping this clip.")
                continue
            
            corrected_clip = clip.copy()
            corrected_clip['video'] = video_path
            corrected_plan.append(corrected_clip)

        if not corrected_plan:
            logger.error("Could not find any source video files for the clips in the editing plan.")
            return
        
        output_video_path = os.path.join(self.video_dir, "final_cut.mp4")
        output_srt_path = os.path.join(self.video_dir, "final_cut.srt")

        # 1. Write the final SRT file
        write_srt_file(self.final_srt_entries, output_srt_path)

        # 2. Execute the video cutting and concatenation
        execute_video_editing(corrected_plan, output_video_path)

        logger.info("--- Video execution finished ---")
        logger.info(f"Final video available at: {output_video_path}")
        logger.info(f"Final SRT available at: {output_srt_path}")

    def run(self):
        """
        Runs the main interactive loop for the agent.
        """
        while True:
            print("\n" + "="*30)
            print("  Video Cutoff Agent Menu")
            print("="*30)
            print(f"Working Directory: {self.video_dir}\n")

            # Display status
            total_videos = len(self.video_data)
            videos_with_srt = sum(1 for data in self.video_data.values() if data.get("srt_path"))
            print(f"Status: {total_videos} videos loaded, {videos_with_srt} with SRT files.\n")

            print("1. [Step 1] Run ASR on videos (without SRTs)")
            print("2. [Step 2] Generate editing plan from SRTs")
            print("3. [Step 3] Execute video editing to create final MP4")
            print("4. Rescan video directory")
            print("5. Exit")
            
            try:
                choice = input("\nEnter your choice (1-5): ")
                if choice == '1':
                    self.run_asr_on_videos()
                elif choice == '2':
                    self.generate_editing_plan()
                elif choice == '3':
                    self.execute_video_cut()
                elif choice == '4':
                    self._scan_video_directory()
                elif choice == '5':
                    logger.info("Exiting Video Cutoff Agent.")
                    break
                else:
                    logger.warning("Invalid choice, please enter a number between 1 and 5.")
            except KeyboardInterrupt:
                logger.info("\nExiting Video Cutoff Agent.")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}", exc_info=True)


def main():
    """
    Main function to parse arguments and start the agent.
    """
    parser = argparse.ArgumentParser(
        description="A command-line agent for intelligent video cutting.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "video_dir", 
        help="The directory containing your source video files."
    )
    args = parser.parse_args()

    try:
        agent = VideoCutoffAgent(args.video_dir)
        agent.run()
    except ValueError as e:
        logger.error(f"Failed to initialize agent: {e}")
    except ImportError as e:
        logger.error(f"A required module is missing: {e}")
        logger.error("Please ensure you have all dependencies installed and that the script is run from the correct directory.")


if __name__ == "__main__":
    main()
