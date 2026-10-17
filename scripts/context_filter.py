"""差分测试筛选上下文用例：同一句话"不给上文"和"给上文"各翻译一次，
只有两次译文存在实质差异（主语、指代、含义不同）的，才算真正需要上下文。

背景：context_candidates.py 让大模型直接判断"是否需要上下文"，结果 60 组判了 58 组，
它把"含网络梗"也当成了"需要上下文"，区分度太低。

运行：python scripts/context_filter.py（需要 .env 中的 LLM_API_KEY）
输入：data/corpus/context_labeled.json（大模型初筛结果，先备份为 context_labeled_all.json）
输出：data/corpus/context_labeled.json（只保留差分有差异的，交给人工审核）
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.srt import Cue  # noqa: E402
from app.translator import LLMTranslator  # noqa: E402

LABELED = ROOT / "data" / "corpus" / "context_labeled.json"
ALL = ROOT / "data" / "corpus" / "context_labeled_all.json"

JUDGE_PROMPT = """下面是同一句中文字幕的两个英文译文：译文 A 翻译时没有看上文，译文 B 看了上文。
判断两者在意思上是否存在实质差异：主语或人称不同、代词指代不同、词义理解不同、语气含义相反等算实质差异；
只是措辞、同义词、语序不同不算。
只输出 JSON：{"different": true/false, "diff": "一句话说明差异在哪（没有差异写'无'）"}"""


def cue_list(lines):
    return [Cue(i + 1, "00:00:00,000", "00:00:01,000", t) for i, t in enumerate(lines)]


def main():
    if not ALL.exists():
        shutil.copy(LABELED, ALL)
    candidates = json.loads(ALL.read_text(encoding="utf-8"))
    t = LLMTranslator("v5")

    kept = []
    for n, c in enumerate(candidates, 1):
        alone = t.translate_cue(cue_list([c["source"]]), 0, "en")
        with_ctx = t.translate_cue(cue_list([*c["before"], c["source"]]), len(c["before"]), "en")
        resp = t.client.chat.completions.create(
            model=t.model, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": JUDGE_PROMPT},
                      {"role": "user", "content": f"上文：{' / '.join(c['before'])}\n原句：{c['source']}\n译文 A：{alone}\n译文 B：{with_ctx}"}],
        )
        verdict = json.loads(resp.choices[0].message.content)
        c.update(alone=alone, with_context=with_ctx, diff=verdict.get("diff", ""))
        if verdict.get("different"):
            kept.append(c)
        print(f"[{n}/{len(candidates)}] {'保留' if verdict.get('different') else '淘汰'} {c['source'][:20]}")

    # 差分结果也写回 ALL，留作分析记录
    ALL.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in kept:  # 审核页面里展示两次译文的差异，作为"为什么需要上下文"
        c["reason"] = f"不看上文：{c['alone']}｜看了上文：{c['with_context']}｜差异：{c['diff']}"
    LABELED.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n差分测试：{len(kept)}/{len(candidates)} 组两次译文存在实质差异，已输出到 {LABELED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
