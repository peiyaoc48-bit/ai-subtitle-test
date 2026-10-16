"""翻译质量评测（DeepEval，LLM-as-Judge）。

需要真实调用大模型，默认跳过。运行方式：
    TRANSLATOR_MODE=llm pytest tests/quality -v

TODO(你来做，第 3 个周末)：
1. 把 CASES 扩充到 100 条，覆盖：人名、俚语/网络用语、成语、多行字幕、超短句
2. 跑 PROMPT_VERSION=v1 和 v2 各一遍，记录通过率，作为简历上的"版本对比"数据
3. 对照 DeepEval 文档确认裁判模型的配置方式（不同版本的 API 可能有变化）
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("TRANSLATOR_MODE") != "llm", reason="质量评测需要 TRANSLATOR_MODE=llm"
)

CASES = [
    # (原文, 质量要求说明)
    ("林晓，你今天怎么来得这么早？", "人名译为 Lin Xiao；是疑问句"),
    ("导演说下午要重拍第三场戏。", "'第三场戏'译为 scene three 或同义表达"),
    ("别担心，我已经背熟了。", "'背熟了'指背熟台词，不能译成背诵课文之类"),
]


@pytest.mark.parametrize("source, requirement", CASES)
def test_translation_quality(source, requirement):
    from deepeval import assert_test
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    from app.srt import Cue
    from app.translator import get_translator

    output = get_translator().translate([Cue(1, "00:00:00,000", "00:00:01,000", source)], "en")[0].text

    metric = GEval(
        name="字幕翻译质量",
        criteria=(
            "判断 actual output 是否是 input 的准确英文字幕翻译：没有漏译、没有误译、"
            f"语气自然、适合屏幕阅读。额外要求：{requirement}"
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.7,
    )
    assert_test(LLMTestCase(input=source, actual_output=output), [metric])
