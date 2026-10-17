"""②③ 分类 + 生成预期：让大模型给每条候选句打标签，并写出翻译要点。

运行：python scripts/label.py（需要 .env 中的 LLM_API_KEY）
输入：data/corpus/candidates_raw.json
输出：data/corpus/candidates_labeled.json（交给人工审核）
"""
import json
import os
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
IN = ROOT / "data" / "corpus" / "candidates_raw.json"
OUT = ROOT / "data" / "corpus" / "candidates_labeled.json"

CATEGORIES = ["网络梗", "方言", "成语俗语", "人名地名", "语气情绪", "需要上下文", "普通句子"]
BATCH = 10

PROMPT = f"""你是字幕翻译测试专家。下面是一批中文句子，请为每一句生成英文翻译的测试要点。
对每一句输出：
- category：从 {CATEGORIES} 中选一个最贴切的
- requirement：一句话说明"什么样的英文翻译才算对"，重点写容易翻错的地方（梗的真实含义、不能字面直译的词等）
- difficulty：1-3，翻译难度
只输出 JSON：{{"items": [{{"source": 原句, "category": ..., "requirement": ..., "difficulty": ...}}]}}，顺序与输入一致。"""


def label_batch(client, model, sentences):
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))},
        ],
    )
    return json.loads(resp.choices[0].message.content)["items"]


def main():
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    sentences = json.loads(IN.read_text(encoding="utf-8"))

    labeled = []
    for i in range(0, len(sentences), BATCH):
        batch = sentences[i : i + BATCH]
        items = label_batch(client, model, batch)
        if len(items) != len(batch):  # 大模型输出条数不对时，丢弃这一批，避免错位
            print(f"第 {i // BATCH + 1} 批返回 {len(items)} 条，期望 {len(batch)} 条，已跳过")
            continue
        for src, item in zip(batch, items):
            labeled.append({
                "id": f"corpus-{len(labeled) + 1:02d}",
                "category": item.get("category", "普通句子"),
                "source": src,  # 用原句而不是模型复述的句子，防止被改写
                "requirement": item.get("requirement", ""),
                "difficulty": item.get("difficulty", 1),
                "accept": None,  # 人工审核：改成 true 或 false
            })
        print(f"已标注 {len(labeled)} 条")

    OUT.write_text(json.dumps(labeled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成，输出到 {OUT.relative_to(ROOT)}，请人工审核 accept 字段")


if __name__ == "__main__":
    main()
