"""SRT 字幕解析与序列化。

一条字幕（cue）的格式：
    1
    00:00:01,000 --> 00:00:03,500
    你好，欢迎来到爱奇艺。
"""
import re
from dataclasses import dataclass

TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}$")


class SrtFormatError(ValueError):
    """SRT 格式不合法。"""


@dataclass
class Cue:
    index: int
    start: str
    end: str
    text: str


def parse_srt(content: str) -> list[Cue]:
    content = content.replace("\r\n", "\n").strip()
    if not content:
        raise SrtFormatError("字幕内容为空")

    cues = []
    for block in re.split(r"\n\s*\n", content):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            raise SrtFormatError(f"字幕块不完整：{block[:30]!r}")
        if not lines[0].strip().isdigit():
            raise SrtFormatError(f"序号不是数字：{lines[0]!r}")
        times = [t.strip() for t in lines[1].split("-->")]
        if len(times) != 2 or not all(TIME_RE.match(t) for t in times):
            raise SrtFormatError(f"时间轴格式错误：{lines[1]!r}")
        cues.append(Cue(int(lines[0]), times[0], times[1], "\n".join(lines[2:])))
    return cues


def to_srt(cues: list[Cue]) -> str:
    return "\n\n".join(f"{c.index}\n{c.start} --> {c.end}\n{c.text}" for c in cues) + "\n"
