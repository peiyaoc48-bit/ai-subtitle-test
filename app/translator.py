"""字幕翻译器。

两种模式：
- mock：不调用大模型，返回带 [EN] 前缀的假翻译。用于本地调试和 CI，不花钱、结果稳定。
- llm ：调用兼容 OpenAI 接口的大模型（默认 DeepSeek）。

通过环境变量 TRANSLATOR_MODE 切换，见 .env.example。
"""
import os
import re
from pathlib import Path

from app.srt import Cue

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# 同一个字连续出现 5 次及以上（如 62 个"呜"）
REPEAT_RE = re.compile(r"(.)\1{4,}")


def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text(encoding="utf-8")


def collapse_repeats(text: str) -> str:
    """把连续重复 5 次以上的字压缩成 3 个，防止大模型陷入重复输出。

    "呜呜呜……（62 个）李娜真坏" → "呜呜呜李娜真坏"
    """
    return REPEAT_RE.sub(r"\1\1\1", text)


class MockTranslator:
    def translate(self, cues: list[Cue], target_lang: str) -> list[Cue]:
        return [Cue(c.index, c.start, c.end, f"[{target_lang.upper()}] {c.text}") for c in cues]


class LLMTranslator:
    def __init__(self, prompt_version: str = "v1"):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        )
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")
        self.system_prompt = load_prompt(prompt_version)

    def translate(self, cues: list[Cue], target_lang: str) -> list[Cue]:
        # 逐条翻译：实现简单，且不会出现"模型合并/拆分字幕行"的问题。
        # TODO(进阶)：改成批量翻译以提速，再用格式测试验证没有漏行。
        result = []
        for c in cues:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": self.system_prompt.format(target_lang=target_lang)},
                    {"role": "user", "content": collapse_repeats(c.text)},
                ],
            )
            result.append(Cue(c.index, c.start, c.end, resp.choices[0].message.content.strip()))
        return result


def get_translator():
    if os.getenv("TRANSLATOR_MODE", "mock") == "llm":
        return LLMTranslator(os.getenv("PROMPT_VERSION", "v1"))
    return MockTranslator()
