"""MQM 错误标注评分（参考 GEMBA-MQM，Kocmi & Federmann, WMT 2023）。

和 v1 评测体系（DeepEval 二元及格判定）的区别：
- 不需要为每条用例手写"翻译要点"，只要原文 + 译文，评测集可以低成本扩大；
- 输出每一处错误的类别和严重程度，能看出版本之间"错在哪里"的变化；
- 分数是连续值，配合 bootstrap 能做置信区间和显著性检验。

计分：MQM 惯例的严重度权重，critical 25、major 5、minor 1，每句扣分求和（0 分为无错误）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

WEIGHTS = {"critical": 25, "major": 5, "minor": 1}
CATEGORIES = ["accuracy", "fluency", "style", "terminology", "other"]

PROMPT = """你是专业的中英字幕翻译质量评审员，按 MQM（Multidimensional Quality Metrics）标准标注译文中的错误。

{context}原文（中文字幕）：
{source}

译文（英文字幕）：
{target}

请找出译文中的所有错误。每个错误给出：
- span：译文中出错的片段（漏译时写原文中被漏掉的片段）
- category：accuracy（漏译、误译、增译、未翻译）/ fluency（语法、拼写、标点、不通顺）/
  style（生硬、不自然、语气不对、不像字幕口语）/ terminology（人名、专有名词、术语不当或不一致）/ other
- severity：
  - critical：严重误导或完全无法理解，观众会得到错误信息
  - major：意思有明显偏差或缺失，但大致还能理解
  - minor：不影响理解的小问题（措辞不够自然、轻微的风格问题）
- note：一句话说明

没有错误时返回空列表。不要因为译文和你自己的译法不同就判错，只标注真正的错误。
只输出 JSON：{{"errors": [{{"span": "...", "category": "...", "severity": "...", "note": "..."}}]}}"""


@dataclass
class MQMResult:
    errors: list[dict] = field(default_factory=list)
    parse_error: str = ""

    @property
    def score(self) -> int:
        """MQM 分数：0 为完美，越负越差。"""
        return -sum(WEIGHTS.get(e.get("severity"), 0) for e in self.errors)

    @property
    def worst(self) -> str:
        """最严重的错误级别：none / minor / major / critical，用于和人工标注对比。"""
        for level in ("critical", "major", "minor"):
            if any(e.get("severity") == level for e in self.errors):
                return level
        return "none"


def build_prompt(source: str, target: str, before: list[str] | None = None) -> str:
    context = f"上文（只用于理解语境，不需要评审）：\n{' / '.join(before)}\n\n" if before else ""
    return PROMPT.format(context=context, source=source, target=target)


def parse(content: str) -> MQMResult:
    """解析裁判输出；对类别、级别做规范化，格式不对时记录原因而不是让整批评测崩掉。"""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return MQMResult(parse_error=f"不是合法 JSON：{str(content)[:80]}")
    raw = data.get("errors") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return MQMResult(parse_error="缺少 errors 列表")
    errors = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        sev = str(e.get("severity", "")).lower().strip()
        cat = str(e.get("category", "other")).lower().split("/")[0].strip()
        if sev not in WEIGHTS:
            continue  # 模型偶尔写 "no-error" 或空级别，跳过
        errors.append({"span": str(e.get("span", "")), "category": cat if cat in CATEGORIES else "other",
                       "severity": sev, "note": str(e.get("note", ""))})
    return MQMResult(errors=errors)


def judge(client, model: str, source: str, target: str, before: list[str] | None = None) -> tuple[MQMResult, dict]:
    """调用裁判模型，返回结果和 token 用量。"""
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=800, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": build_prompt(source, target, before)}],
    )
    usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens} if resp.usage else {}
    return parse(resp.choices[0].message.content), usage
