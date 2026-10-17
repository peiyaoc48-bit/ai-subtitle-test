"""⑤ 合并：把人工审核通过（accept 为 true）的用例加入评测集，并统计采纳率。

运行：python scripts/merge.py
合并上下文用例：python scripts/merge.py context
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
CONTEXT = len(sys.argv) > 1 and sys.argv[1] == "context"
LABELED = ROOT / "data" / "corpus" / ("context_labeled.json" if CONTEXT else "candidates_labeled.json")
CASES = ROOT / "data" / "eval" / ("context_cases.json" if CONTEXT else "cases.json")
FIELDS = ("id", "category", "before", "source", "after", "requirement") if CONTEXT else ("id", "category", "source", "requirement")


def main():
    labeled = json.loads(LABELED.read_text(encoding="utf-8"))
    pending = [c for c in labeled if c["accept"] is None]
    if pending:
        print(f"还有 {len(pending)} 条未审核（accept 为 null），请先审核完")
        return

    accepted = [c for c in labeled if c["accept"] is True]
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    existing = {c["id"] for c in cases}
    new = [
        {k: c[k] for k in FIELDS}
        for c in accepted
        if c["id"] not in existing
    ]
    cases.extend(new)
    CASES.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"采纳率：{len(accepted)}/{len(labeled)} = {len(accepted) / len(labeled):.0%}")
    print(f"新增 {len(new)} 条，评测集共 {len(cases)} 条")
    print("分类分布：", dict(Counter(c["category"] for c in cases)))


if __name__ == "__main__":
    main()
