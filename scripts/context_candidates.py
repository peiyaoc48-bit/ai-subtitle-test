"""从对话语料中生成"需要上下文"的候选用例：抽取 → 大模型筛选并写预期 → 交给人工审核。

语料里每条都是"一人说 → 另一人回"，回复那句单独看常常意思不明，配上前一句才能理解。

运行：python scripts/context_candidates.py（需要 .env 中的 LLM_API_KEY）
输出：data/corpus/context_labeled.json（在审核网页里审核：REVIEW_FILE=context_labeled.json）
"""
import json
import os
import random
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "corpus" / "raw" / "zh_meme_test.jsonl"
OUT = ROOT / "data" / "corpus" / "context_labeled.json"

MAX_BEFORE, MAX_SOURCE = 60, 40   # 上文、当前句的最大字数（字幕不宜过长）
SAMPLE_SIZE, BATCH, SEED = 60, 10, 42

PROMPT = """你是字幕翻译测试专家。下面每组是一段两人对话：A 先说，B 回复。
判断：如果只看 B 这一句、不看 A，把 B 翻译成英文时是否容易翻错或翻得含糊？
典型情况：B 省略了主语或宾语、用了指代不明的代词、是"我也是/看情况"这类回答型短句、
用了多义词、或者是反讽，必须结合 A 才能确定意思。

对每组输出：
- need_context：true / false
- reason：一句话说明不看上文会在哪里翻错（need_context 为 false 时写"单独可译"）
- requirement：结合上文后，B 的正确英文翻译应满足什么（写具体，比如应补出的主语、应体现的含义）
只输出 JSON：{"items": [{"need_context": ..., "reason": ..., "requirement": ...}]}，顺序与输入一致。"""


def load_pairs():
    pairs = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        msgs = json.loads(line)["messages"]
        before, source = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if len(before) <= MAX_BEFORE and len(source) <= MAX_SOURCE:
            pairs.append((before, source))
    return sorted(set(pairs))


def main():
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    pairs = load_pairs()
    random.seed(SEED)
    sample = random.sample(pairs, min(SAMPLE_SIZE, len(pairs)))
    print(f"长度合适的对话 {len(pairs)} 组，抽样 {len(sample)} 组交给大模型筛选")

    labeled, kept = [], 0
    for i in range(0, len(sample), BATCH):
        batch = sample[i : i + BATCH]
        resp = client.chat.completions.create(
            model=model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": PROMPT},
                      {"role": "user", "content": "\n\n".join(f"第{j + 1}组\nA：{b}\nB：{s}" for j, (b, s) in enumerate(batch))}],
        )
        items = json.loads(resp.choices[0].message.content)["items"]
        if len(items) != len(batch):  # 条数不对时丢弃整批，避免错位
            print(f"第 {i // BATCH + 1} 批返回 {len(items)} 条，期望 {len(batch)} 条，已跳过")
            continue
        for (before, source), item in zip(batch, items):
            if not item.get("need_context"):
                continue
            kept += 1
            labeled.append({
                "id": f"ctx-c{kept:02d}", "category": "需要上下文",
                "before": [before], "source": source, "after": [],
                "requirement": item.get("requirement", ""), "reason": item.get("reason", ""),
                "difficulty": 2, "accept": None,
            })
    OUT.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"大模型判定需要上下文的 {len(labeled)}/{len(sample)} 组，输出到 {OUT.relative_to(ROOT)}，请人工审核")


if __name__ == "__main__":
    main()
