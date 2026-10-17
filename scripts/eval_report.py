"""汇总评测结果：按分类对比各 Prompt 版本的通过率。

运行：python scripts/eval_report.py v1 v2
输入：reports/eval_<版本>.xml（pytest --junitxml 生成）
输出：终端表格 + reports/eval_summary.md
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
CASES = {c["id"]: c for c in json.loads((ROOT / "data" / "eval" / "cases.json").read_text(encoding="utf-8"))}


def load_results(version: str) -> dict[str, bool]:
    """返回 {用例 id: 是否通过}。"""
    tree = ET.parse(ROOT / "reports" / f"eval_{version}.xml")
    results = {}
    for tc in tree.iter("testcase"):
        m = re.search(r"\[(.+)\]$", tc.get("name", ""))
        if m and tc.find("skipped") is None:
            results[m.group(1)] = tc.find("failure") is None and tc.find("error") is None
    return results


def main(versions: list[str]):
    results = {v: load_results(v) for v in versions}
    by_cat = defaultdict(lambda: defaultdict(list))
    for v, res in results.items():
        for case_id, ok in res.items():
            by_cat[CASES[case_id]["category"]][v].append(ok)

    def rate(oks):
        return f"{sum(oks)}/{len(oks)} ({sum(oks) / len(oks):.0%})" if oks else "-"

    lines = ["| 分类 | " + " | ".join(versions) + " |", "|---|" + "---|" * len(versions)]
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c][versions[0]])):
        lines.append(f"| {cat} | " + " | ".join(rate(by_cat[cat][v]) for v in versions) + " |")
    lines.append("| **总计** | " + " | ".join(f"**{rate(list(results[v].values()))}**" for v in versions) + " |")

    # 列出版本间结果不同的用例，方便分析
    diff = [cid for cid in CASES if len({results[v].get(cid) for v in versions}) > 1]
    lines += ["", f"版本间结果不同的用例（{len(diff)} 条）："]
    lines += [f"- {cid}（{CASES[cid]['category']}）：{CASES[cid]['source']} → "
              + "、".join(f"{v} {'✓' if results[v].get(cid) else '✗'}" for v in versions) for cid in diff]

    report = "\n".join(lines)
    print(report)
    (ROOT / "reports" / "eval_summary.md").write_text("# 翻译质量评测结果\n\n" + report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["v1", "v2"])
