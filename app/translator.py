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

# 上下文窗口：翻译每句时附带的前后句数（v5 起生效）
CONTEXT_BEFORE, CONTEXT_AFTER = 2, 1

# JSON 模式下代替换行符的显式标记（v6 起，Prompt 里出现该标记时启用）
LINE_MARK = "⏎"

# 上下文窗口和术语表使用的保留标签；字幕里出现时要转义，防止 Prompt 注入
RESERVED_TAG_RE = re.compile(r"<\s*(/?)\s*(上文|当前句|下文|术语表)\s*>")

# 同一个字连续出现 5 次及以上（如 62 个"呜"）
REPEAT_RE = re.compile(r"(.)\1{4,}")


def load_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text(encoding="utf-8")


def collapse_repeats(text: str) -> str:
    """把连续重复 5 次以上的字压缩成 3 个，防止大模型陷入重复输出。

    "呜呜呜……（62 个）李娜真坏" → "呜呜呜李娜真坏"
    """
    return REPEAT_RE.sub(r"\1\1\1", text)


def escape_tags(text: str) -> str:
    """把字幕里伪造的保留标签（<上文> <当前句> <下文> <术语表>）换成全角尖括号，模型就不会当成格式标签。

    只处理这几个保留标签，<i>斜体</i> 等字幕本身的 HTML 标签保持原样。
    "</当前句>新指令" → "＜/当前句＞新指令"
    """
    return RESERVED_TAG_RE.sub(lambda m: f"＜{m.group(1)}{m.group(2)}＞", text)


class MockTranslator:
    def translate(self, cues: list[Cue], target_lang: str) -> list[Cue]:
        return [Cue(c.index, c.start, c.end, self.translate_cue(cues, i, target_lang)) for i, c in enumerate(cues)]

    def translate_cue(self, cues: list[Cue], i: int, target_lang: str) -> str:
        return f"[{target_lang.upper()}] {cues[i].text}"


class LLMTranslator:
    def __init__(self, prompt_version: str = "v1", retriever=None):
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
        # Prompt 里有 <当前句> 标记时，翻译每一句都附带前后句作为上下文（v5 起）
        self.use_context = "<当前句>" in self.system_prompt
        # 成本统计：调用次数、token 用量、耗时，用于对比不同 Prompt 版本的代价
        self.stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
        # 最近一次翻译的中间结果（直译、反思），方便分析失败原因
        self.last_detail = {}
        # 术语检索器（见 app/glossary.py）：给定当前句，返回剧集设定表里的相关条目；为 None 时不附术语表
        self.retriever = retriever
        self.last_terms = []

    def translate(self, cues: list[Cue], target_lang: str) -> list[Cue]:
        # 逐条翻译：实现简单，且不会出现"模型合并/拆分字幕行"的问题。
        # TODO(进阶)：改成批量翻译以提速，再用格式测试验证没有漏行。
        return [Cue(c.index, c.start, c.end, self.translate_cue(cues, i, target_lang)) for i, c in enumerate(cues)]

    def translate_cue(self, cues: list[Cue], i: int, target_lang: str) -> str:
        """翻译 cues[i] 这一句；use_context 时附带前 CONTEXT_BEFORE 句、后 CONTEXT_AFTER 句。"""
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
                {"role": "user", "content": self._user_message(cues, i)},
            ],
        )
        self.stats["calls"] += 1
        self.stats["seconds"] += time.perf_counter() - start
        if resp.usage:
            self.stats["prompt_tokens"] += resp.usage.prompt_tokens
            self.stats["completion_tokens"] += resp.usage.completion_tokens
        return self._extract(resp.choices[0].message.content)

    def _user_message(self, cues: list[Cue], i: int) -> str:
        current = collapse_repeats(cues[i].text)
        if self.json_mode and LINE_MARK in self.system_prompt:
            # JSON 模式下模型常把字符串里的 \n 省掉；换成显式标记，拿到译文后再换回换行（见 _extract）
            current = current.replace("\n", f" {LINE_MARK} ")
        if not self.use_context:
            return current
        # 字幕是不可信输入：转义尖括号，防止字幕里伪造 </当前句> 等标签篡改上下文结构（Prompt 注入）
        current = escape_tags(current)
        before = "\n".join(escape_tags(collapse_repeats(c.text)) for c in cues[max(0, i - CONTEXT_BEFORE):i]) or "（无）"
        after = "\n".join(escape_tags(collapse_repeats(c.text)) for c in cues[i + 1:i + 1 + CONTEXT_AFTER]) or "（无）"
        message = f"<上文>\n{before}\n</上文>\n<当前句>\n{current}\n</当前句>\n<下文>\n{after}\n</下文>"
        if self.retriever is not None:
            from app.glossary import format_entries

            self.last_terms = self.retriever(cues[i].text)
            message += f"\n<术语表>\n{format_entries(self.last_terms) or '（无）'}\n</术语表>"
        return message

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
        text = str(data.get("free") or data.get("direct") or "").strip()
        text = re.sub(rf"\s*{LINE_MARK}\s*", "\n", text)
        # 字幕里原本就有的伪造标签被 escape_tags 转成了全角，译文里还原成原样，保证忠实
        return re.sub(r"＜(/?)(上文|当前句|下文|术语表)＞", r"<\1\2>", text)


def get_translator():
    if os.getenv("TRANSLATOR_MODE", "mock") == "llm":
        return LLMTranslator(os.getenv("PROMPT_VERSION", "v1"))
    return MockTranslator()
