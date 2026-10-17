"""评测体系 2.0 报告：MQM 均分与 95% 置信区间、错误分布、分组表现、成本、版本间配对显著性检验。

运行：python scripts/mqm_report.py v1 v3 v4 v5 v6
输出：终端 + reports/mqm_summary.md
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval2 import stats  # noqa: E402

SET = {i["id"]: i for i in json.loads((ROOT / "data/eval/mqm_set.json").read_text(encoding="utf-8"))}


def load(v):
    return json.loads((ROOT / "reports" / "mqm" / f"{v}.json").read_text(encoding="utf-8"))


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def main(versions):
    data = {v: load(v) for v in versions}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))  # 只比较所有版本都评完的条目
    lines = [f"# 评测体系 2.0：MQM 错误标注评测", "",
             f"评测集 {len(ids)} 条（单句 + 上下文），裁判模型标注错误类别与严重度，"
             f"按 critical 25 / major 5 / minor 1 扣分；分数越接近 0 越好。置信区间与显著性均为 1000 次 bootstrap。", ""]

    lines += ["## 总体", "", "| 版本 | MQM 均分（95% CI）| 无错误比例 | 含 major 及以上 | 每 100 句错误数 | 译文平均输入 token | 平均耗时 |",
              "|---|---|---|---|---|---|---|"]
    scores = {}
    for v, d in data.items():
        rows = [d[i] for i in ids]
        scores[v] = [r["score"] for r in rows]
        m, lo, hi = stats.bootstrap_ci(scores[v])
        clean = sum(r["score"] == 0 for r in rows) / len(rows)
        major = sum(r["worst"] in ("major", "critical") for r in rows) / len(rows)
        n_err = sum(len(r["errors"]) for r in rows) / len(rows) * 100
        tok = mean(r["translate_tokens"]["prompt_tokens"] for r in rows)
        sec = mean(r["translate_seconds"] for r in rows)
        lines.append(f"| {v} | **{m:.2f}**（{lo:.2f} ～ {hi:.2f}）| {clean:.0%} | {major:.0%} | {n_err:.0f} | {tok:.0f} | {sec:.2f}s |")

    lines += ["", "## 版本间配对显著性检验", "",
              "p 值为新版本**不优于**旧版本的 bootstrap 抽样比例；p < 0.05 说明新版本显著更好，p > 0.95 说明显著更差。", "",
              "| 对比 | 均分变化 | p 值 | 结论 |", "|---|---|---|---|"]
    pairs = [(versions[i - 1], versions[i]) for i in range(1, len(versions))]
    pairs += [(versions[0], v) for v in versions[2:]]
    for a, b in pairs:
        p = stats.paired_bootstrap(scores[a], scores[b])
        verdict = "显著更好" if p < 0.05 else "显著更差" if p > 0.95 else "差异不显著"
        lines.append(f"| {a} → {b} | {mean(scores[b]) - mean(scores[a]):+.2f} | {fmt_p(p)} | {verdict} |")

    lines += ["", "## 错误类型分布（每 100 句）", "", "| 版本 | " + " | ".join(
        f"{c}" for c in ["accuracy", "fluency", "style", "terminology", "other"]) + " | critical | major | minor |",
              "|---|" + "---|" * 8]
    for v, d in data.items():
        errs = [e for i in ids for e in d[i]["errors"]]
        cat, sev = Counter(e["category"] for e in errs), Counter(e["severity"] for e in errs)
        k = 100 / len(ids)
        lines.append(f"| {v} | " + " | ".join(f"{cat[c] * k:.0f}" for c in ["accuracy", "fluency", "style", "terminology", "other"])
                     + " | " + " | ".join(f"{sev[s] * k:.0f}" for s in ["critical", "major", "minor"]) + " |")

    lines += ["", "## 分组 MQM 均分", "", "| 分组 | 条数 | " + " | ".join(versions) + " |", "|---|---|" + "---|" * len(versions)]
    groups = defaultdict(list)
    for i in ids:
        groups[SET[i]["category"]].append(i)
    for g, gids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"| {g} | {len(gids)} | " + " | ".join(f"{mean(data[v][i]['score'] for i in gids):.2f}" for v in versions) + " |")

    bad = {v: sum(bool(d[i].get("parse_error")) for i in ids) for v, d in data.items()}
    lines += ["", f"裁判输出解析失败的条目（按 0 分计）：{bad}", ""]
    report = "\n".join(lines)
    print(report)
    (ROOT / "reports" / "mqm_summary.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["v1", "v3", "v4", "v5", "v6"])
