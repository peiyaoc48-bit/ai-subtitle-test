"""评测统计：bootstrap 置信区间、配对 bootstrap 显著性检验、评分一致性。

- bootstrap 置信区间：从评测集有放回抽样 1000 次，取均值的 2.5% / 97.5% 分位数。
- 配对 bootstrap（Koehn, 2004；WMT 至今仍在用）：每次抽样用同一批用例给两个系统打分，
  统计 B 不比 A 好的比例作为 p 值，p < 0.05 认为差异显著。
- Cohen's kappa（加权）：裁判和人工在"最严重错误级别"上的一致性，排除了碰巧一致的部分。
- Spearman 相关：裁判和人工的分数排序是否一致。

只用标准库，便于审查和在任何环境运行。
"""
from __future__ import annotations

import random
from statistics import mean


def bootstrap_ci(values: list[float], n: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """返回 (均值, 下限, 上限)。"""
    rng = random.Random(seed)
    k = len(values)
    means = sorted(mean(values[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return mean(values), means[int(n * alpha / 2)], means[int(n * (1 - alpha / 2)) - 1]


def paired_bootstrap(a: list[float], b: list[float], n: int = 1000, seed: int = 0) -> float:
    """检验 B 是否比 A 好（分数越高越好）。返回 p 值：B 的均值不高于 A 的抽样比例。"""
    if len(a) != len(b):
        raise ValueError("配对检验要求两组用例一一对应")
    rng = random.Random(seed)
    k = len(a)
    not_better = 0
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        if sum(b[i] for i in idx) <= sum(a[i] for i in idx):
            not_better += 1
    return not_better / n


def weighted_kappa(x: list[int], y: list[int], levels: int) -> float:
    """二次加权 Cohen's kappa：等级相差越远惩罚越重。x、y 为 0..levels-1 的等级。"""
    if len(x) != len(y) or not x:
        raise ValueError("两组标注长度必须一致且非空")
    n = len(x)
    obs = [[0] * levels for _ in range(levels)]
    for i, j in zip(x, y):
        obs[i][j] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(levels)) for j in range(levels)]
    w = [[(i - j) ** 2 / (levels - 1) ** 2 for j in range(levels)] for i in range(levels)]
    observed = sum(w[i][j] * obs[i][j] for i in range(levels) for j in range(levels)) / n
    expected = sum(w[i][j] * row[i] * col[j] for i in range(levels) for j in range(levels)) / n / n
    return 1.0 if expected == 0 else 1 - observed / expected


def _ranks(v: list[float]) -> list[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1  # 并列取平均名次
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    mx, my = mean(rx), mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return 0.0 if sx == 0 or sy == 0 else cov / (sx * sy)
