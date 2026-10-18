"""术语一致性评测：检索质量 + 端到端翻译，对比不检索 / 关键词检索 / 混合检索 / 标准答案术语表 / Agent。

1. 检索评测（python scripts/rag_eval.py retrieval）
   用 data/glossary/cases.json 的标准答案，计算每种检索器的精确率、召回率，以及同形词、无术语句子的误检率。
2. 端到端评测（python scripts/rag_eval.py translate none keyword oracle agent）
   - 术语准确率：当前句里的每个专有名词，译文是否用了设定表里的译法（不区分大小写，开头的 the 可省略）
   - 同形词误用率：普通含义的词（"一世长安"）被套用了译名（Chang'an）的句子比例
   - 成本：每句平均输入 / 输出 token、耗时；Agent 另计步数、查询次数、是否查到了该查的术语
   结果按条写入 reports/rag/<方式>.json，断点续跑。
3. 报告（python scripts/rag_eval.py report）→ reports/rag_summary.md
   打分规则修改后用 python scripts/rag_eval.py rescore 按已保存的译文重新打分
"""
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.glossary import EmbeddingRetriever, Glossary, HybridRetriever, KeywordRetriever, openai_embedder  # noqa: E402
from eval2 import stats  # noqa: E402

G = Glossary.load()
CASES = json.loads((ROOT / "data" / "glossary" / "cases.json").read_text(encoding="utf-8"))
# 端到端评测不含"未登记叫法"：台词用"萧家那位储君"之类的说法时，忠实的译文本来就不该出现人名，
# 按"译文里有没有译名"打分会奖励模型擅自添加信息。这类用例只用于检索评测。
E2E = [c for c in CASES if c["kind"] != "未登记叫法"]
OUT = ROOT / "reports" / "rag"
KINDS = ["全名", "别名", "称谓", "多术语", "带上文", "同形词", "无术语"]
WORKERS = int(os.getenv("WORKERS", "8"))


# ---------- 检索 ----------
# 向量检索需要一个相似度门槛（否则每句都固定返回 k 个条目）。门槛在开发集（编号为偶数的用例）上选，
# 在测试集（编号为奇数）上报告，避免"用测试数据调参"。
DEV = [c for c in CASES if int(c["id"][2:]) % 2 == 0]
TEST = [c for c in CASES if int(c["id"][2:]) % 2 == 1]
RETRIEVAL_KINDS = ["全名", "别名", "称谓", "多术语", "带上文", "未登记叫法"]


def prf(retriever, cases):
    tp = fp = fn = 0
    for c in cases:
        got, gold = {e["id"] for e in retriever(c["text"])}, set(c["terms"])
        tp, fp, fn = tp + len(got & gold), fp + len(got - gold), fn + len(gold - got)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def tune_threshold(make, cases):
    grid = [round(0.30 + 0.02 * i, 2) for i in range(16)]
    return max(grid, key=lambda t: prf(make(t), cases)[2])


def retrieval():
    embed = openai_embedder()
    kw = KeywordRetriever(G)
    t1 = tune_threshold(lambda t: EmbeddingRetriever(G, embed, k=1, threshold=t), DEV)
    t3 = tune_threshold(lambda t: EmbeddingRetriever(G, embed, k=3, threshold=t), DEV)
    th = tune_threshold(lambda t: HybridRetriever(kw, EmbeddingRetriever(G, embed, k=1, threshold=t)), DEV)
    rs = {"关键词": kw,
          f"向量 top1（门槛 {t1}）": EmbeddingRetriever(G, embed, k=1, threshold=t1),
          f"向量 top3（门槛 {t3}）": EmbeddingRetriever(G, embed, k=3, threshold=t3),
          f"混合：关键词 + 向量 top1（门槛 {th}）": HybridRetriever(kw, EmbeddingRetriever(G, embed, k=1, threshold=th))}
    n = Counter(c["kind"] for c in TEST)
    lines = ["## 检索质量", "",
             f"测试集 {len(TEST)} 条（开发集 {len(DEV)} 条只用来选向量检索的相似度门槛）。"
             "精确率、召回率按条目计；误检率 = 同形词 / 无术语句子中检索出了不该出的条目的比例。", "",
             "| 检索器 | 精确率 | 召回率 | F1 | 同形词误检率 | 无术语误检率 |", "|---|---|---|---|---|---|"]
    for name, r in rs.items():
        p, rc, f1 = prf(r, TEST)
        bad = Counter(c["kind"] for c in TEST if c["kind"] in ("同形词", "无术语") and r(c["text"]))
        lines.append(f"| {name} | {p:.0%} | {rc:.0%} | {f1:.2f} | {bad['同形词'] / n['同形词']:.0%} | {bad['无术语'] / n['无术语']:.0%} |")
    lines += ["", "按类型的召回率：", "", "| 类型 | 条数 | " + " | ".join(rs) + " |", "|---|---|" + "---|" * len(rs)]
    for k in RETRIEVAL_KINDS:
        ks = [c for c in TEST if c["kind"] == k]
        lines.append(f"| {k} | {len(ks)} | " + " | ".join(f"{prf(r, ks)[1]:.0%}" for r in rs.values()) + " |")
    return lines


# ---------- 翻译 ----------
def term_used(en: str, text: str) -> bool:
    core = re.sub(r"^the ", "", en.lower()).replace("’", "'")
    return re.search(rf"(?<![a-z]){re.escape(core)}(?![a-z])", text.lower().replace("’", "'")) is not None


def accepted(e):
    """设定表允许的译法：en，加上说明里规定的称呼译法（accept，如“二皇子”译 the Second Prince）。"""
    return [e["en"], *e.get("accept", [])]


def score(case, translation):
    gold = [G.by_id[i] for i in case["terms"]]
    hits = [any(term_used(t, translation or "") for t in accepted(e)) for e in gold]
    # 句子里出现了写法、但不是标准答案的条目（同形词等）：译文里出现它的译名就算误用
    spurious = [e for e in G.match(case["text"]) if e["id"] not in case["terms"]]
    misused = [e["id"] for e in spurious if any(term_used(t, translation or "") for t in accepted(e))]
    return {"term_hits": hits, "misused": misused}


class OracleRetriever:
    """直接给出标准答案里的条目：相当于检索 100% 准确，用来看术语表本身能带来多少提升（上限）。"""

    def __init__(self):
        self.gold = {c["text"]: [G.by_id[i] for i in c["terms"]] for c in E2E}

    def __call__(self, text):
        return self.gold.get(text, [])


def make_worker(mode):
    from app.srt import Cue
    from app.translator import LLMTranslator

    local = threading.local()

    def translate(case):
        if mode == "agent":
            from openai import OpenAI

            from app.agent import TranslationAgent

            if not hasattr(local, "a"):
                client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
                local.a = TranslationAgent(client, os.getenv("LLM_MODEL", "deepseek-chat"), G)
            r = local.a.run(case["text"], case["before"])
            # 每个应查的条目，是否至少有一次查询命中了它（查询词能匹配到该条目）
            looked = {e["id"] for q in r.lookups for e in G.match(q)}
            return {"translation": r.translation, "status": r.status, "steps": r.steps, "lookups": r.lookups,
                    "looked_up_all": set(case["terms"]) <= looked,
                    "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "seconds": round(r.seconds, 2)}
        if not hasattr(local, "t"):
            retriever = {"none": None, "keyword": KeywordRetriever(G), "oracle": OracleRetriever()}[mode]
            local.t = LLMTranslator("v6" if mode == "none" else "v7", retriever=retriever)
        t = local.t
        before = dict(t.stats)
        lines = [*case["before"], case["text"]]
        cues = [Cue(i + 1, "00:00:00,000", "00:00:01,000", s) for i, s in enumerate(lines)]
        start = time.perf_counter()
        out = t.translate_cue(cues, len(case["before"]), "English")
        return {"translation": out, "retrieved": [e["id"] for e in t.last_terms],
                "prompt_tokens": t.stats["prompt_tokens"] - before["prompt_tokens"],
                "completion_tokens": t.stats["completion_tokens"] - before["completion_tokens"],
                "seconds": round(time.perf_counter() - start, 2)}

    return translate


def translate(modes):
    OUT.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        path = OUT / f"{mode}.json"
        done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        todo = [c for c in E2E if c["id"] not in done]
        print(f"== {mode}：待翻译 {len(todo)} / {len(E2E)}", file=sys.stderr)
        work = make_worker(mode)
        lock = threading.Lock()
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(work, c): c for c in todo}
            for n, f in enumerate(as_completed(futures), 1):
                c = futures[f]
                try:
                    r = f.result()
                except Exception as e:
                    print(f"  {c['id']} 失败，重跑会补上：{e}", file=sys.stderr)
                    continue
                with lock:
                    done[c["id"]] = {**r, **score(c, r["translation"])}
                    if n % 20 == 0:
                        path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {mode} 完成 {len(done)}/{len(E2E)}", file=sys.stderr)


# ---------- 报告 ----------
NAMES = {"none": "不给术语表（v6）", "keyword": "关键词检索 + v7", "agent_v1": "Agent（修复前）",
         "oracle": "标准答案术语表 + v7（上限）", "agent": "Agent 自主查询"}


def report(modes):
    data = {m: json.loads((OUT / f"{m}.json").read_text(encoding="utf-8")) for m in modes if (OUT / f"{m}.json").exists()}
    modes = list(data)
    ids = [c["id"] for c in E2E if all(c["id"] in data[m] for m in modes)]
    by_id = {c["id"]: c for c in CASES}
    with_terms = [i for i in ids if by_id[i]["terms"]]
    homo = [i for i in ids if G.match(by_id[i]["text"]) and set(e["id"] for e in G.match(by_id[i]["text"])) - set(by_id[i]["terms"])]

    def acc(m, i):
        h = data[m][i]["term_hits"]
        return sum(h) / len(h)

    lines = ["# 术语一致性评测：RAG 与 Agent", "",
             f"虚构古装剧《长安夜行》的剧集设定表 {len(G.entries)} 个条目，台词 {len(ids)} 条（其中 {len(with_terms)} 条含专有名词，"
             f"{len(homo)} 条含普通含义的同形词）。译文由 DeepSeek 生成；术语是否用对按设定表译名做字符串匹配，不依赖裁判模型。", ""]
    lines += ["## 端到端结果", "",
              "| 方式 | 术语准确率（95% CI）| 整句术语全对 | 同形词误用率 | 每句输入 token | 每句输出 token | 每句耗时 |",
              "|---|---|---|---|---|---|---|"]
    for m in modes:
        a = [acc(m, i) for i in with_terms]
        mu, lo, hi = stats.bootstrap_ci(a)
        full = sum(all(data[m][i]["term_hits"]) for i in with_terms) / len(with_terms)
        mis = sum(bool(data[m][i]["misused"]) for i in homo) / max(len(homo), 1)
        tok_in = mean(data[m][i]["prompt_tokens"] for i in ids)
        tok_out = mean(data[m][i]["completion_tokens"] for i in ids)
        sec = mean(data[m][i]["seconds"] for i in ids)
        lines.append(f"| {NAMES.get(m, m)} | **{mu:.1%}**（{lo:.1%} ～ {hi:.1%}）| {full:.0%} | {mis:.0%} | "
                     f"{tok_in:.0f} | {tok_out:.0f} | {sec:.1f}s |")

    if "none" in modes:
        lines += ["", "## 与不给术语表相比（配对 bootstrap）", "", "| 方式 | 术语准确率变化 | p 值 |", "|---|---|---|"]
        base = [acc("none", i) for i in with_terms]
        for m in modes:
            if m == "none":
                continue
            cur = [acc(m, i) for i in with_terms]
            p = stats.paired_bootstrap(base, cur)
            lines.append(f"| {NAMES.get(m, m)} | {mean(cur) - mean(base):+.1%} | {'< 0.001' if p < 0.001 else f'{p:.3f}'} |")

    lines += ["", "## 按难点类型的术语准确率", "", "| 类型 | 条数 | " + " | ".join(NAMES.get(m, m) for m in modes) + " |",
              "|---|---|" + "---|" * len(modes)]
    for k in KINDS:
        ks = [i for i in with_terms if by_id[i]["kind"] == k]
        if ks:
            lines.append(f"| {k} | {len(ks)} | " + " | ".join(f"{mean(acc(m, i) for i in ks):.0%}" for m in modes) + " |")

    # 哪些条目最容易错
    miss = defaultdict(Counter)
    for m in modes:
        for i in with_terms:
            for tid, ok in zip(by_id[i]["terms"], data[m][i]["term_hits"]):
                miss[m][tid] += not ok
    lines += ["", "## 最常译错的条目（各方式出错次数）", "", "| 条目 | 译名 | " + " | ".join(NAMES.get(m, m) for m in modes) + " |",
              "|---|---|" + "---|" * len(modes)]
    worst = sorted(G.by_id, key=lambda t: -sum(miss[m][t] for m in modes))[:8]
    for t in worst:
        lines.append(f"| {G.by_id[t]['zh']} | {G.by_id[t]['en']} | " + " | ".join(str(miss[m][t]) for m in modes) + " |")

    if "agent" in modes:
        a = data["agent"]
        st = Counter(a[i]["status"] for i in ids)
        lines += ["", "## Agent 行为", "",
                  "| 指标 | 数值 |", "|---|---|",
                  f"| 平均步数（模型调用次数）| {mean(a[i]['steps'] for i in ids):.2f} |",
                  f"| 平均查询次数 | {mean(len(a[i]['lookups']) for i in ids):.2f} |",
                  f"| 含专有名词的句子中，所有术语都查过的比例 | {sum(a[i]['looked_up_all'] for i in with_terms) / len(with_terms):.0%} |",
                  f"| 无术语句子中仍去查询的比例 | {sum(bool(a[i]['lookups']) for i in ids if by_id[i]['kind'] == '无术语') / max(1, sum(by_id[i]['kind'] == '无术语' for i in ids)):.0%} |",
                  f"| 结束方式 | " + "，".join(f"{k} {v}" for k, v in st.most_common()) + " |"]

    lines += ["", "## 检索质量（不调用翻译模型）", ""] + [l for l in retrieval_lines()]
    text = "\n".join(lines) + "\n"
    print(text)
    (ROOT / "reports" / "rag_summary.md").write_text(text, encoding="utf-8")


def retrieval_lines():
    cache = ROOT / "reports" / "rag" / "retrieval.md"
    if cache.exists():
        return cache.read_text(encoding="utf-8").splitlines()[2:]
    return ["（先运行 python scripts/rag_eval.py retrieval）"]


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "report"), sys.argv[2:]
    if cmd == "retrieval":
        OUT.mkdir(parents=True, exist_ok=True)
        text = "\n".join(retrieval()) + "\n"
        print(text)
        (OUT / "retrieval.md").write_text(text, encoding="utf-8")
    elif cmd == "rescore":   # 打分规则改动后，用已保存的译文重新打分，不重新翻译
        for path in OUT.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            by_id = {c["id"]: c for c in CASES}
            data = {k: {**v, **score(by_id[k], v["translation"])} for k, v in data.items()}
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print("rescored", path.name)
    elif cmd == "translate":
        translate(args or ["none", "keyword", "oracle", "agent"])
    else:
        report(args or ["none", "keyword", "oracle", "agent_v1", "agent"])
