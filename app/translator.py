"""字幕翻译器。

两种模式：
- mock：不调用大模型，返回带 [EN] 前缀的假翻译。用于本地调试和 CI，不花钱、结果稳定。
- llm ：调用兼容 OpenAI 接口的大模型（默认 DeepSeek）。

通过环境变量 TRANSLATOR_MODE 切换，见 .env.example。
"""
import json
import os
import re
import time
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
        self.prompt_version = prompt_version
        self.system_prompt = load_prompt(prompt_version)
        # Prompt 要求输出 JSON（直译→反思→意译）时，按 JSON 解析，取 free 字段作为译文
        self.json_mode = "JSON" in self.system_prompt
        # 成本统计：调用次数、token 用量、耗时，用于对比不同 Prompt 版本的代价
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        # 最近一次翻译的中间结果（直译、反思），方便分析失败原因
        self.last_detail = {}

    def translate(self, cues: list[Cue], target_lang: str) -> list[Cue]:
        # 逐条翻译：实现简单，且不会出现"模型合并/拆分字幕行"的问题。
        # TODO(进阶)：改成批量翻译以提速，再用格式测试验证没有漏行。
        result = []
        for c in cues:
            extra = {"response_format": {"type": "json_object"}} if self.json_mode else {}
            start = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                # JSON 模式要输出直译、反思、意译三段，需要更多额度；仍保留上限防止输出失控
                max_tokens=600 if self.json_mode else 200,
                **extra,
                messages=[
                    {"role": "system", "content": self.system_prompt.format(target_lang=target_lang)},
                    {"role": "user", "content": collapse_repeats(c.text)},
                ],
            )
            self.stats["calls"] += 1
            self.stats["seconds"] += time.perf_counter() - start
            if resp.usage:
                self.stats["prompt_tokens"] += resp.usage.prompt_tokens
                self.stats["completion_tokens"] += resp.usage.completion_tokens
            result.append(Cue(c.index, c.start, c.end, self._extract(resp.choices[0].message.content)))
        return result

    def _extract(self, content: str) -> str:
        content = (content or "").strip()
        if not self.json_mode:
            return content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            self.last_detail = {"error": "JSON 解析失败", "raw": content}
            return content
        self.last_detail = data
        return str(data.get("free") or data.get("direct") or "").strip()


def get_translator():
    if os.getenv("TRANSLATOR_MODE", "mock") == "llm":
        return LLMTranslator(os.getenv("PROMPT_VERSION", "v1"))
    return MockTranslator()
