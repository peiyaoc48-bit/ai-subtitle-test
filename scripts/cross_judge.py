"""跨模型裁判交叉验证：换一个不同家族的模型当第二裁判，检验 MQM 结论是否依赖 DeepSeek 裁判。

- 译文不重新生成：直接复用 reports/mqm/<版本>.json 里 DeepSeek 的译文，只换裁判，保证唯一变量是裁判
- 第二裁判用同一套 MQM 提示词（eval2/mqm.py），任何兼容 OpenAI 接口的模型都可以
- 断点续跑、多线程，结果写入 reports/mqm_cross/<模型>/<版本>.json

配置（写在 .env，不要提交）：
  JUDGE2_API_KEY=...
  JUDGE2_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1   # 阿里云百炼
  JUDGE2_MODEL=qwen-plus

运行：
  python scripts/cross_judge.py run v1 v3 v4 v5 v6
  python scripts/cross_judge.py report v1 v3 v4 v5 v6     # 输出 reports/cross_judge.md
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval2 import mqm, stats  # noqa: E402

LEVELS = ["none", "minor", "major", "critical"]
SET = {i["id"]: i for i in json.loads((ROOT / "data" / "eval" / "mqm_set.json").read_text(encoding="utf-8"))}
MODEL = os.getenv("JUDGE2_MODEL", "qwen-plus")
OUT_DIR = ROOT / "reports" / "mqm_cross" / MODEL
WORKERS = int(os.getenv("WORKERS", "8"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run(versions):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["JUDGE2_API_KEY"], base_url=os.environ["JUDGE2_BASE_URL"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for v in versions:
        base = load(ROOT / "reports" / "mqm" / f"{v}.json")
        out = OUT_DIR / f"{v}.json"
        done = load(out)
        todo = [k for k in base if k not in done]
        print(f"== {v}：共 {len(base)} 条，已完成 {len(done)}，待评 {len(todo)}", file=sys.stderr)
        lock = threading.Lock()

        def work(k):
            it = SET[k]
            result, usage = mqm.judge(client, MODEL, it["source"], base[k]["translation"], it["before"] or None)
            return k, {"errors": result.errors, "score": result.score, "worst": result.worst,
                       "parse_error": result.parse_error, "judge_tokens": usage}

        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(work, k) for k in todo]
            for n, f in enumerate(as_completed(futures), 1):
                try:
                    k, value = f.result()
                except Exception as e:  # 单条失败不影响其他条，重跑会补上
                    print(f"  一条失败，稍后重跑会补上：{e}", file=sys.stderr)
                    continue
                with lock:
                    done[k] = value
                    if n % 10 == 0:
                        out.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
                if n % 50 == 0:
                    print(f"  {v} 进度 {n}/{len(todo)}", file=sys.stderr)
        out.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  {v} 完成：{len(done)}/{len(base)}", file=sys.stderr)


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def report(versions):
    a = {v: load(ROOT / "reports" / "mqm" / f"{v}.json") for v in versions}
    b = {v: load(OUT_DIR / f"{v}.json") for v in versions}
    ids = sorted(set.intersection(*(set(a[v]) & set(b[v]) for v in versions)))
    lines = [f"# 跨模型裁判交叉验证：DeepSeek vs {MODEL}", "",
             f"同一批译文（{len(ids)} 条 × {len(versions)} 个版本，均由 DeepSeek 翻译），分别由两个裁判按同一套 MQM 标准评分。", "",
             "## 各版本 MQM 均分", "",
             f"| 版本 | DeepSeek 裁判（95% CI）| {MODEL} 裁判（95% CI）| 含 major 及以上（DS / {MODEL}）|", "|---|---|---|---|"]
    sa, sb = {}, {}
    for v in versions:
        sa[v] = [a[v][i]["score"] for i in ids]
        sb[v] = [b[v][i]["score"] for i in ids]
        ma, la, ha = stats.bootstrap_ci(sa[v])
        mb, lb, hb = stats.bootstrap_ci(sb[v])
        maj = lambda d: sum(d[v][i]["worst"] in ("major", "critical") for i in ids) / len(ids)
        lines.append(f"| {v} | {ma:.2f}（{la:.2f} ～ {ha:.2f}）| {mb:.2f}（{lb:.2f} ～ {hb:.2f}）| {maj(a):.0%} / {maj(b):.0%} |")

    rank = lambda s: " < ".join(sorted(versions, key=lambda v: mean(s[v])))
    rho = stats.spearman([mean(sa[v]) for v in versions], [mean(sb[v]) for v in versions])
    lines += ["", f"版本排名（从差到好）：DeepSeek 裁判 {rank(sa)}；{MODEL} 裁判 {rank(sb)}；排名 Spearman 相关 {rho:.2f}。"]

    lines += ["", "## 配对显著性检验（两个裁判各自的结论）", "",
              f"| 对比 | DeepSeek：变化 / p | {MODEL}：变化 / p | 结论是否一致 |", "|---|---|---|---|"]
    pairs = [(versions[i - 1], versions[i]) for i in range(1, len(versions))] + [(versions[0], versions[-1])]
    verdict = lambda p: "更好" if p < 0.05 else "更差" if p > 0.95 else "不显著"
    for x, y in pairs:
        pa, pb = stats.paired_bootstrap(sa[x], sa[y]), stats.paired_bootstrap(sb[x], sb[y])
        lines.append(f"| {x} → {y} | {mean(sa[y]) - mean(sa[x]):+.2f} / {fmt_p(pa)} | {mean(sb[y]) - mean(sb[x]):+.2f} / {fmt_p(pb)} | "
                     f"{'一致' if verdict(pa) == verdict(pb) else '不一致'}（{verdict(pa)} / {verdict(pb)}）|")

    la_ = [LEVELS.index(a[v][i]["worst"]) for v in versions for i in ids]
    lb_ = [LEVELS.index(b[v][i]["worst"]) for v in versions for i in ids]
    ship = lambda x: x >= 2
    lines += ["", "## 裁判之间的一致性（逐句）", "",
              "| 指标 | 数值 |", "|---|---|",
              f"| 最严重级别加权 kappa | **{stats.weighted_kappa(la_, lb_, len(LEVELS)):.2f}** |",
              f"| 级别完全一致 | {sum(x == y for x, y in zip(la_, lb_)) / len(la_):.0%} |",
              f"| 能否上线（严重及以上）判断一致 | {sum(ship(x) == ship(y) for x, y in zip(la_, lb_)) / len(la_):.0%} |",
              f"| DeepSeek 更严 / {MODEL} 更严 | {sum(x > y for x, y in zip(la_, lb_)) / len(la_):.0%} / {sum(x < y for x, y in zip(la_, lb_)) / len(la_):.0%} |"]

    human = ROOT / "data" / "eval" / "human_mqm.json"
    if human.exists():
        from scripts.human_mqm import gold

        rows = [r for r in json.loads(human.read_text(encoding="utf-8")) if r.get("human") and r["id"] in b.get(r["version"], {})]
        if rows:
            g = [LEVELS.index(gold(r)) for r in rows]
            hb = [LEVELS.index(r["human"]["worst"]) for r in rows]
            dj = [LEVELS.index(r["judge_worst"]) for r in rows]
            qj = [LEVELS.index(b[r["version"]][r["id"]]["worst"]) for r in rows]
            lines += ["", f"## 与人工标注对比（{len(rows)} 条）", "",
                      "复核参考答案由看过 DeepSeek 裁判理由的复核者给出，对 DeepSeek 裁判偏有利；"
                      f"{MODEL} 没有参与复核，用它检验这一偏差。", "",
                      "| 裁判 | 与人工盲评 kappa | 与复核参考答案 kappa | 能否上线判断准确率（对复核答案）|", "|---|---|---|---|"]
            for name, pred in (("DeepSeek", dj), (MODEL, qj)):
                lines.append(f"| {name} | {stats.weighted_kappa(pred, hb, len(LEVELS)):.2f} | "
                             f"{stats.weighted_kappa(pred, g, len(LEVELS)):.2f} | "
                             f"{sum(ship(x) == ship(y) for x, y in zip(pred, g)) / len(g):.0%} |")

    bad = {v: sum(bool(b[v][i].get("parse_error")) for i in ids) for v in versions}
    tok = sum(t for v in versions for i in ids for t in b[v][i].get("judge_tokens", {}).values())
    lines += ["", f"{MODEL} 裁判输出解析失败（按 0 分计）：{bad}；共消耗 {tok:,} token。", ""]
    text = "\n".join(lines)
    print(text)
    (ROOT / "reports" / "cross_judge.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    cmd, versions = (sys.argv[1] if len(sys.argv) > 1 else "report"), (sys.argv[2:] or ["v1", "v3", "v4", "v5", "v6"])
    {"run": run, "report": report}[cmd](versions)
