"""④ 人工审核（表格版）：导出成 CSV 用 WPS / Excel 审核，审完再导回。

导出：python scripts/review_sheet.py export   → data/corpus/review.csv
导入：python scripts/review_sheet.py import

表格里只需要填"采纳"列（Y 或 N）；要改翻译要点或分类，直接改对应单元格。
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LABELED = ROOT / "data" / "corpus" / "candidates_labeled.json"
SHEET = ROOT / "data" / "corpus" / "review.csv"
COLUMNS = ["id", "采纳(Y/N)", "原文", "翻译要点", "分类", "难度", "不采纳原因"]


def to_flag(accept):
    return {True: "Y", False: "N"}.get(accept, "")


def export():
    items = json.loads(LABELED.read_text(encoding="utf-8"))
    # utf-8-sig 带 BOM，WPS / Excel 打开中文不乱码
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for c in items:
            w.writerow([c["id"], to_flag(c["accept"]), c["source"], c["requirement"],
                        c["category"], c["difficulty"], c.get("reject_reason", "")])
    print(f"已导出 {len(items)} 条到 {SHEET.relative_to(ROOT)}，用 WPS 打开审核，保存时选 CSV 格式")


def import_():
    items = {c["id"]: c for c in json.loads(LABELED.read_text(encoding="utf-8"))}
    with SHEET.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    problems = []
    for row in rows:
        c = items.get(row["id"])
        if c is None:
            problems.append(f"未知 id：{row['id']}")
            continue
        flag = row["采纳(Y/N)"].strip().upper()
        if flag not in ("Y", "N", ""):
            problems.append(f"{row['id']} 的采纳列填了 {flag!r}，只能填 Y 或 N")
            continue
        if row["翻译要点"].strip() != c["requirement"] or row["分类"].strip() != c["category"]:
            c["edited"] = True
        c["requirement"], c["category"] = row["翻译要点"].strip(), row["分类"].strip()
        c["accept"] = {"Y": True, "N": False}.get(flag)
        c["reject_reason"] = row["不采纳原因"].strip()

    LABELED.write_text(json.dumps(list(items.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    done = [c for c in items.values() if c["accept"] is not None]
    print(f"已导入：审核 {len(done)}/{len(items)} 条，采纳 {sum(c['accept'] for c in done)} 条，"
          f"人工修改 {sum(bool(c.get('edited')) for c in items.values())} 条")
    for p in problems:
        print("⚠️", p)


if __name__ == "__main__":
    {"export": export, "import": import_}[sys.argv[1] if len(sys.argv) > 1 else "export"]()
