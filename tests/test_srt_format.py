"""格式正确性测试：翻译不能破坏字幕结构。

不管翻译质量如何，以下规则必须 100% 成立，否则字幕在播放器里会错位：
- 条数不变、序号不变
- 时间轴一字不改
- 每条译文非空
"""
import pytest

from app.srt import parse_srt


@pytest.fixture(scope="module")
def pair(client, any_srt):
    resp = client.post("/translate", json={"srt": any_srt, "target_lang": "en"})
    assert resp.status_code == 200
    return parse_srt(any_srt), parse_srt(resp.json()["srt"])


def test_cue_count_unchanged(pair):
    src, out = pair
    assert len(out) == len(src)


def test_indexes_unchanged(pair):
    src, out = pair
    assert [c.index for c in out] == [c.index for c in src]


def test_timestamps_unchanged(pair):
    src, out = pair
    for s, o in zip(src, out):
        assert (o.start, o.end) == (s.start, s.end), f"第 {s.index} 条时间轴被改动"


def test_no_empty_text(pair):
    _, out = pair
    assert all(c.text.strip() for c in out)


def test_line_breaks_preserved(pair):
    src, out = pair
    for s, o in zip(src, out):
        assert o.text.count("\n") == s.text.count("\n"), f"第 {s.index} 条换行数不一致"


MAX_RATIO = 10  # 译文长度最多是原文的几倍


def test_no_runaway_output(pair):
    """译文不能比原文长太多，防止大模型重复输出失控。"""
    src, out = pair
    for s, o in zip(src, out):
        ratio = len(o.text) / len(s.text)
        assert ratio <= MAX_RATIO, f"第 {s.index} 条译文长度是原文的 {ratio:.0f} 倍"
