"""翻译质量评测（DeepEval，LLM-as-Judge，裁判模型为 DeepSeek）。

需要真实调用大模型，默认跳过。运行方式：
    TRANSLATOR_MODE=llm PROMPT_VERSION=v1 pytest tests/quality -v
    TRANSLATOR_MODE=llm PROMPT_VERSION=v2 pytest tests/quality -v

评测数据：
- data/eval/cases.json：单句用例，每条包含原文和质量要求
- data/eval/context_cases.json：上下文用例，额外带 before（上文）和 after（下文），
  只有看了上下文才能翻对；只跑这部分：pytest tests/quality -k ctx
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TRANSLATOR_MODE") != "llm", reason="质量评测需要 TRANSLATOR_MODE=llm"
)

EVAL_DIR = Path(__file__).parents[2] / "data" / "eval"
CASES = [
    case
    for name in ("cases.json", "context_cases.json")
    if (EVAL_DIR / name).exists()
    for case in json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))
]


@pytest.fixture(scope="module")
def judge():
    from deepeval.models import DeepSeekModel

    return DeepSeekModel(model="deepseek-chat", api_key=os.environ["LLM_API_KEY"], temperature=0)


@pytest.fixture(scope="module")
def translator():
    from app.translator import get_translator

    t = get_translator()
    t.outputs = {}
    yield t
    # 评测结束后保存成本统计和每条译文（含直译、反思），供 eval_report.py 对比分析
    version = os.getenv("PROMPT_VERSION", "v1")
    out = Path(__file__).parents[2] / "reports" / f"detail_{version}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"stats": t.stats, "outputs": t.outputs}, ensure_ascii=False, indent=2),
                   encoding="utf-8")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_translation_quality(case, judge, translator):
    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    from app.srt import Cue

    before, after = case.get("before", []), case.get("after", [])
    lines = [*before, case["source"], *after]
    cues = [Cue(i + 1, "00:00:00,000", "00:00:01,000", text) for i, text in enumerate(lines)]
    output = translator.translate_cue(cues, len(before), "en")
    translator.outputs[case["id"]] = {"output": output, **translator.last_detail}

    # 裁判也要看到上下文，否则无法判断指代、省略是否翻对
    judge_input = case["source"]
    if before or after:
        judge_input = (f"上文：{' / '.join(before) or '（无）'}\n当前句：{case['source']}\n"
                       f"下文：{' / '.join(after) or '（无）'}\n（只需评判当前句的翻译）")

    metric = GEval(
        name="字幕翻译质量",
        criteria=(
            "判断 actual output 是否是 input 的准确英文字幕翻译：没有漏译、没有误译、"
            f"语气自然、适合屏幕阅读。额外要求：{case['requirement']}"
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        threshold=0.7,
        async_mode=False,
    )
    assert_test(LLMTestCase(input=judge_input, actual_output=output), [metric])
