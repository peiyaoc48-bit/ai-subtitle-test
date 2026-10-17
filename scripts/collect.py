"""① 采集：从开放语料中抽取适合做字幕的短句。

运行：python scripts/collect.py
输出：data/corpus/candidates_raw.json
"""
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "corpus" / "raw" / "zh_meme_test.jsonl"
OUT = ROOT / "data" / "corpus" / "candidates_raw.json"

MIN_LEN, MAX_LEN = 6, 40  # 字幕一般不超过 40 字
SAMPLE_SIZE = 50          # 抽多少条交给大模型分类
SEED = 42                 # 固定随机种子，保证每次抽到的一样（结果可复现）


def load_sentences() -> list[str]:
    sentences = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for msg in record["messages"]:
            # 按中文句末标点切成短句，保留标点
            for s in re.findall(r"[^。！？!?]+[。！？!?]?", msg["content"]):
                sentences.append(s.strip())
    return sentences


def main():
    sentences = load_sentences()

    # TODO 1：只保留长度在 MIN_LEN 到 MAX_LEN 之间的句子
    #   提示：列表推导式 [s for s in sentences if ____]
    filtered = [s for s in sentences if MIN_LEN<=len(s)<= MAX_LEN]

    # TODO 2：去重。提示：先转成集合 set() 再转回列表，最后 sorted() 让顺序固定
    unique = sorted(set(filtered))

    random.seed(SEED)
    sample = random.sample(unique, min(SAMPLE_SIZE, len(unique)))

    OUT.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"共 {len(sentences)} 句 → 长度合适 {len(filtered)} 句 → 去重后 {len(unique)} 句 → 抽样 {len(sample)} 句")


if __name__ == "__main__":
    main()
