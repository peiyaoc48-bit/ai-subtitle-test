"""评测体系 2.0：对每个 Prompt 版本，翻译评测集并用 MQM 裁判标注错误。

- 断点续跑：结果按条写入 reports/mqm/<版本>.json，已完成的条目不会重复调用（不重复花钱）
- 并发：多线程同时请求，缩短整体耗时
- 余额保护：每个版本开始前查询 DeepSeek 余额，低于阈值就停止

运行：python scripts/run_mqm.py v1 v3 v4 v5 v6
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.srt import Cue  # noqa: E402
from app.translator import LLMTranslator  # noqa: E402
from eval2 import mqm  # noqa: E402

SET = ROOT / "data" / "eval" / "mqm_set.json"
OUT_DIR = ROOT / "reports" / "mqm"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-chat")
WORKERS = int(os.getenv("WORKERS", "8"))
MIN_BALANCE = float(os.getenv("MIN_BALANCE", "1.5"))


def balance() -> float | None:
    req = urllib.request.Request("https://api.deepseek.com/user/balance",
                                 headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"})
    try:
        import ssl
        import certifi  # openai 依赖里自带，用它的证书避免 macOS 上的 SSL 报错
        ctx = ssl.create_default_context(cafile=certifi.where())
        data = json.load(urllib.request.urlopen(req, timeout=20, context=ctx))
        return sum(float(b["total_balance"]) for b in data.get("balance_infos", []) if b["currency"] == "CNY")
    except Exception as e:  # 查不到余额时不阻塞评测，只提示
        print(f"  查询余额失败：{e}", file=sys.stderr)
        return None


def run_version(version: str, items: list[dict]) -> None:
    out = OUT_DIR / f"{version}.json"
    done = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    todo = [it for it in items if it["id"] not in done]
    print(f"== {version}：共 {len(items)} 条，已完成 {len(done)}，待评 {len(todo)}", file=sys.stderr)
    if not todo:
        return

    local = threading.local()
    lock = threading.Lock()

    def work(it):
        if not hasattr(local, "t"):
            local.t = LLMTranslator(version)
        t = local.t
        before_stats = dict(t.stats)
        lines = [*it["before"], it["source"]]
        cues = [Cue(i + 1, "00:00:00,000", "00:00:01,000", s) for i, s in enumerate(lines)]
        start = time.perf_counter()
        translation = t.translate_cue(cues, len(it["before"]), "en")
        seconds = time.perf_counter() - start
        result, judge_usage = mqm.judge(t.client, JUDGE_MODEL, it["source"], translation, it["before"] or None)
        return it["id"], {
            "translation": translation,
            "errors": result.errors, "score": result.score, "worst": result.worst, "parse_error": result.parse_error,
            "translate_tokens": {k: t.stats[k] - before_stats[k] for k in ("prompt_tokens", "completion_tokens")},
            "translate_seconds": round(seconds, 2), "judge_tokens": judge_usage,
        }

    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(work, it) for it in todo]
        for n, f in enumerate(as_completed(futures), 1):
            try:
                key, value = f.result()
            except Exception as e:  # 单条失败（网络超时等）不影响其他条，下次运行会补上
                print(f"  一条失败，稍后重跑会补上：{e}", file=sys.stderr)
                continue
            with lock:
                done[key] = value
                if n % 10 == 0 or n == len(todo):
                    out.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
            if n % 50 == 0:
                print(f"  {version} 进度 {n}/{len(todo)}", file=sys.stderr)
    out.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {version} 完成：{len(done)}/{len(items)}", file=sys.stderr)


def main(versions: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = json.loads(SET.read_text(encoding="utf-8"))
    for v in versions:
        b = balance()
        if b is not None:
            print(f"当前余额 ¥{b:.2f}", file=sys.stderr)
            if b < MIN_BALANCE:
                sys.exit(f"余额低于 ¥{MIN_BALANCE}，为避免欠费，停止在 {v} 之前")
        run_version(v, items)


if __name__ == "__main__":
    main(sys.argv[1:] or ["v1", "v3", "v4", "v5", "v6"])
