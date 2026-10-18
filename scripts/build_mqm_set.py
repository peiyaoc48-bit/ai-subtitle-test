"""构建评测集 2.0：原有 45 条单句 + 7 条上下文用例 + 从语料训练集新抽样的句子。

MQM 评测只需要原文，不需要手写翻译要点，所以可以低成本扩大规模。
新抽样的句子用大模型批量标注分类（只用于分组统计，不影响打分）。

运行：python scripts/build_mqm_set.py              # 从头构建
      python scripts/build_mqm_set.py extend 800   # 保留已有条目，再追加 800 条（已有评测结果可复用）
输出：data/eval/mqm_set.json
"""
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "corpus" / "raw" / "zh_meme_train.jsonl"
OUT = ROOT / "data" / "eval" / "mqm_set.json"
NEW_SAMPLES, SEED = 150, 2026
CATEGORIES = ["网络梗", "方言", "成语俗语", "人名地名", "语气情绪", "需要上下文", "普通句子"]


def sentences():
    if not RAW.exists():
        sys.exit("缺少语料文件，请先运行 ./scripts/download_corpus.sh")
    out = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        for msg in json.loads(line)["messages"]:
            out += [s.strip() for s in re.findall(r"[^。！？!?]+[。！？!?]?", msg["content"])]
    return sorted({s for s in out if 6 <= len(s) <= 40})


def categorize(batch):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    prompt = (f"给每个中文句子分类，类别只能从 {CATEGORIES} 中选一个。"
              '只输出 JSON：{"categories": ["...", ...]}，顺序、数量与输入一致。\n\n'
              + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(batch)))
    resp = client.chat.completions.create(model=os.getenv("LLM_MODEL", "deepseek-chat"), temperature=0,
                                          response_format={"type": "json_object"},
                                          messages=[{"role": "user", "content": prompt}])
    cats = json.loads(resp.choices[0].message.content).get("categories", [])
    return [c if c in CATEGORIES else "普通句子" for c in cats] if len(cats) == len(batch) else ["普通句子"] * len(batch)


def label(sample, first_id):
    items = []
    for start in range(0, len(sample), 25):
        batch = sample[start:start + 25]
        for j, (s, cat) in enumerate(zip(batch, categorize(batch))):
            items.append({"id": f"m-{first_id + start + j:03d}", "source": s, "before": [], "category": cat,
                          "origin": "语料训练集抽样"})
        print(f"已分类 {min(start + 25, len(sample))}/{len(sample)}", file=sys.stderr)
    return items


def extend(n):
    items = json.loads(OUT.read_text(encoding="utf-8"))
    used = {i["source"] for i in items}
    first = max(int(i["id"][2:]) for i in items if i["id"].startswith("m-")) + 1
    random.seed(SEED + len(items))   # 换一个种子，避免和第一次抽样重复
    items += label(random.sample([s for s in sentences() if s not in used], n), first)
    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"评测集共 {len(items)} 条：{dict(Counter(i['category'] for i in items))}")


def main():
    old = json.loads((ROOT / "data/eval/cases.json").read_text(encoding="utf-8"))
    ctx = json.loads((ROOT / "data/eval/context_cases.json").read_text(encoding="utf-8"))
    items = [{"id": c["id"], "source": c["source"], "before": [], "category": c["category"], "origin": "v1 评测集"} for c in old]
    items += [{"id": c["id"], "source": c["source"], "before": c["before"], "category": c["category"], "origin": "上下文评测集"} for c in ctx]

    used = {i["source"] for i in items}
    pool = [s for s in sentences() if s not in used]
    random.seed(SEED)
    sample = random.sample(pool, NEW_SAMPLES)
    items += label(sample, 1)

    OUT.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"评测集 2.0 共 {len(items)} 条：{dict(Counter(i['category'] for i in items))}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extend":
        extend(int(sys.argv[2]) if len(sys.argv) > 2 else 800)
    else:
        main()
