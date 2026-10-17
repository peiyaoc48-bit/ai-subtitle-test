"""④ 人工审核：逐条查看大模型生成的候选用例，决定是否采纳。

运行：python scripts/review.py
按键：y 采纳 / n 不采纳 / e 修改翻译要点后采纳 / c 修改分类后采纳 / q 保存并退出
每审一条自动保存，中途退出下次会从未审核的地方继续。
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
LABELED = ROOT / "data" / "corpus" / "candidates_labeled.json"
CATEGORIES = ["网络梗", "方言", "成语俗语", "人名地名", "语气情绪", "需要上下文", "普通句子"]


def save(items):
    LABELED.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    items = json.loads(LABELED.read_text(encoding="utf-8"))
    todo = [c for c in items if c["accept"] is None]
    print(f"共 {len(items)} 条，待审核 {len(todo)} 条\n")

    for n, c in enumerate(todo, 1):
        print(f"[{n}/{len(todo)}] {c['id']}  分类：{c['category']}  难度：{c['difficulty']}")
        print(f"  原文：{c['source']}")
        print(f"  要点：{c['requirement']}")
        while True:
            key = input("  y 采纳 / n 不采纳 / e 改要点 / c 改分类 / q 退出 > ").strip().lower()
            if key == "y":
                c["accept"] = True
            elif key == "n":
                c["accept"] = False
                c["reject_reason"] = input("  不采纳的原因（可直接回车跳过）> ").strip()
            elif key == "e":
                c["requirement"] = input("  新的翻译要点 > ").strip() or c["requirement"]
                c["accept"], c["edited"] = True, True
            elif key == "c":
                print("  " + "  ".join(f"{i}.{name}" for i, name in enumerate(CATEGORIES, 1)))
                idx = input("  选择分类编号 > ").strip()
                if idx.isdigit() and 1 <= int(idx) <= len(CATEGORIES):
                    c["category"], c["accept"], c["edited"] = CATEGORIES[int(idx) - 1], True, True
                else:
                    print("  编号无效，请重新选择")
                    continue
            elif key == "q":
                save(items)
                print("已保存，下次运行会继续")
                return
            else:
                continue
            break
        save(items)
        print()

    accepted = sum(c["accept"] is True for c in items)
    edited = sum(bool(c.get("edited")) for c in items)
    print(f"审核完成：采纳 {accepted}/{len(items)}，其中人工修改 {edited} 条")


if __name__ == "__main__":
    main()
