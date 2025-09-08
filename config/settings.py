import os
from pathlib import Path
from typing import Dict, List, Optional

class LLMConfig():
    """LLM 配置"""
    provider: str = "google"
    model_name: str = "gemini-2.5-flash"
    api_key: Optional[str] = "API_KEY"
    base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 60000
    timeout: int = 260  # 增加超时时间
    max_retries: int = 3  # 最大重试次数
    retry_delay: float = 2.0  # 增加重试延迟
    stream: bool = False  # 禁用流式传输
