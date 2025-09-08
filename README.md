修改配置KEY
8 class LLMConfig():
  7     """LLM 配置"""
  6     provider: str = "google"
  5     model_name: str = "gemini-2.5-flash"
  4     api_key: Optional[str] = "API_KEY" #这里填写自己的key
  3     base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"
  2     temperature: float = 0.7

## 使用说明
python3 video_cutoff_agent.py <video_dir>
