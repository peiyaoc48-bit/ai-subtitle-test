"""翻译质量评测（DeepEval，LLM-as-Judge，裁判模型为 DeepSeek）。

需要真实调用大模型，默认跳过。运行方式：
    TRANSLATOR_MODE=llm PROMPT_VERSION=v1 pytest tests/quality -v
    TRANSLATOR_MODE=llm PROMPT_VERSION=v2 pytest tests/quality -v

评测数据在 data/eval/cases.json，每条包含原文和质量要求。
"""
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TRANSLATOR_MODE") != "llm", reason="质量评测需要 TRANSLATOR_MODE=llm"
)

CASES = json.loads(
    (Path(__file__).parents[2] / "data" / "eval" / "cases.json").read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def judge():
    from deepeval.models import DeepSeekModel

    return DeepSeekModel(model="deepseek-chat", api_key=os.environ["LLM_API_KEY"], temperature=0)


@pytest.fixture(scope="module")
def translator():
    from app.translator import get_translator

    return get_translator()


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_translation_quality(case, judge, translator):
    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    from app.srt import Cue

    output = translator.translate([Cue(1, "00:00:00,000", "00:00:01,000", case["source"])], "en")[0].text

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
    assert_test(LLMTestCase(input=case["source"], actual_output=output), [metric])
