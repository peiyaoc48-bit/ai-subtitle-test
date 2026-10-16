"""接口测试：正常流程 + 参数校验 + 异常输入。"""
import pytest


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_translate_ok(client, sample_srt):
    resp = client.post("/translate", json={"srt": sample_srt, "target_lang": "en"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cue_count"] == 5
    assert body["srt"].strip()


@pytest.mark.parametrize("lang", ["en", "ja", "ko", "th", "es"])
def test_supported_langs(client, sample_srt, lang):
    resp = client.post("/translate", json={"srt": sample_srt, "target_lang": lang})
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "payload, status, case",
    [
        ({"srt": "", "target_lang": "en"}, 400, "空字幕"),
        ({"srt": "   \n\n  ", "target_lang": "en"}, 400, "只有空白"),
        ({"srt": "1\n00:00:01,000 --> 00:00:02,000\n你好", "target_lang": "fr"}, 400, "不支持的语言"),
        ({"srt": "一\n00:00:01,000 --> 00:00:02,000\n你好", "target_lang": "en"}, 400, "序号非数字"),
        ({"srt": "1\n00:00:01 --> 00:00:02\n你好", "target_lang": "en"}, 400, "时间轴缺毫秒"),
        ({"srt": "1\n00:00:01,000 --> 00:00:02,000", "target_lang": "en"}, 400, "缺少字幕文本"),
        ({"target_lang": "en"}, 422, "缺少 srt 字段"),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_invalid_input(client, payload, status, case):
    resp = client.post("/translate", json=payload)
    assert resp.status_code == status, case


def test_too_many_cues(client):
    cue = "{i}\n00:00:01,000 --> 00:00:02,000\n你好"
    srt = "\n\n".join(cue.format(i=i) for i in range(1, 502))
    resp = client.post("/translate", json={"srt": srt, "target_lang": "en"})
    assert resp.status_code == 413


# TODO(你来补)：
# - 超长单条字幕（如 5000 字）应该怎么处理？先想清楚预期，再写用例
# - 字幕里带 HTML 标签（<i>斜体</i>）、emoji、特殊符号
# - Windows 换行符 \r\n 的字幕文件
