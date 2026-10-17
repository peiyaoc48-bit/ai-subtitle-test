"""人工标注：验证 MQM 裁判是否可信。

1. 抽样：python scripts/human_mqm.py sample
   从 v1（较差）和 v6（较好）的译文中各抽一半，按裁判判定的最严重级别分层，
   保证"无错误 / 轻微 / 严重 / 致命"都有覆盖，一致性系数才有意义。
2. 标注：.venv/bin/uvicorn scripts.human_mqm:app --port 8003，浏览器打开 http://127.0.0.1:8003
   盲评：页面上看不到裁判的结论，避免被影响。
3. 计算一致性：python scripts/human_mqm.py agreement
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SET = ROOT / "data" / "eval" / "mqm_set.json"
SAMPLE = ROOT / "data" / "eval" / "human_mqm.json"
LEVELS = ["none", "minor", "major", "critical"]
LEVEL_ZH = {"none": "无错误", "minor": "轻微", "major": "严重", "critical": "致命"}
PENALTY = {"none": 0, "minor": -1, "major": -5, "critical": -25}
SIZE, SEED = 50, 7


def sample():
    items = {i["id"]: i for i in json.loads(SET.read_text(encoding="utf-8"))}
    rng = random.Random(SEED)
    picked = []
    for version in ("v1", "v6"):
        results = json.loads((ROOT / "reports" / "mqm" / f"{version}.json").read_text(encoding="utf-8"))
        by_level = {lv: [k for k, r in results.items() if r["worst"] == lv] for lv in LEVELS}
        quota = SIZE // 2
        # 每个级别尽量平均抽，某级别不够时由其他级别补足
        for lv in LEVELS:
            rng.shuffle(by_level[lv])
        while quota > 0 and any(by_level.values()):
            for lv in LEVELS:
                if quota and by_level[lv]:
                    k = by_level[lv].pop()
                    r = results[k]
                    picked.append({"key": f"{version}:{k}", "id": k, "version": version,
                                   "source": items[k]["source"], "before": items[k]["before"],
                                   "translation": r["translation"], "judge_worst": r["worst"],
                                   "judge_score": r["score"], "judge_errors": r["errors"], "human": None})
                    quota -= 1
    rng.shuffle(picked)  # 打乱顺序，标注时看不出来自哪个版本
    SAMPLE.write_text(json.dumps(picked, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已抽样 {len(picked)} 条，裁判判定分布：{dict(Counter(p['judge_worst'] for p in picked))}")


def agreement():
    from eval2 import stats

    rows = [r for r in json.loads(SAMPLE.read_text(encoding="utf-8")) if r.get("human")]
    if len(rows) < 10:
        sys.exit(f"已标注 {len(rows)} 条，至少需要 10 条才能计算一致性")
    h = [LEVELS.index(r["human"]["worst"]) for r in rows]
    j = [LEVELS.index(r["judge_worst"]) for r in rows]
    kappa = stats.weighted_kappa(h, j, len(LEVELS))
    rho = stats.spearman([PENALTY[r["human"]["worst"]] for r in rows], [r["judge_score"] for r in rows])
    exact = sum(a == b for a, b in zip(h, j)) / len(rows)
    stricter = sum(b > a for a, b in zip(h, j)) / len(rows)
    looser = sum(b < a for a, b in zip(h, j)) / len(rows)
    has_err_h = [a > 0 for a in h]
    has_err_j = [b > 0 for b in j]
    tp = sum(a and b for a, b in zip(has_err_h, has_err_j))
    precision = tp / max(sum(has_err_j), 1)
    recall = tp / max(sum(has_err_h), 1)
    verdict = ("几乎完全一致" if kappa > 0.8 else "高度一致" if kappa > 0.6 else "中等一致" if kappa > 0.4
               else "一般" if kappa > 0.2 else "较差")
    lines = [
        "# MQM 裁判与人工标注的一致性", "",
        f"人工标注 {len(rows)} 条（v1、v6 各约一半，按裁判级别分层抽样，盲评）。", "",
        "| 指标 | 数值 | 说明 |", "|---|---|---|",
        f"| 加权 Cohen's kappa | **{kappa:.2f}**（{verdict}）| 最严重错误级别（无 / 轻微 / 严重 / 致命）的一致性，已排除碰巧一致 |",
        f"| Spearman 相关 | {rho:.2f} | 裁判分数与人工级别的排序一致性 |",
        f"| 级别完全一致 | {exact:.0%} | |",
        f"| 裁判比人工严格 | {stricter:.0%} | 裁判判的级别更重 |",
        f"| 裁判比人工宽松 | {looser:.0%} | 裁判判的级别更轻 |",
        f"| 发现错误的精确率 | {precision:.0%} | 裁判说有错的句子里，人工也认为有错的比例 |",
        f"| 发现错误的召回率 | {recall:.0%} | 人工认为有错的句子里，裁判也发现了的比例 |",
        "", "## 级别混淆矩阵（行：人工，列：裁判）", "",
        "| 人工 \\ 裁判 | " + " | ".join(LEVEL_ZH[l] for l in LEVELS) + " |", "|---|" + "---|" * len(LEVELS),
    ]
    for a in range(len(LEVELS)):
        lines.append(f"| {LEVEL_ZH[LEVELS[a]]} | " + " | ".join(str(sum(1 for x, y in zip(h, j) if x == a and y == b)) for b in range(len(LEVELS))) + " |")
    report = "\n".join(lines)
    print(report)
    (ROOT / "reports" / "judge_agreement.md").write_text(report + "\n", encoding="utf-8")


# ---------- 标注网页 ----------
try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    app = FastAPI(title="MQM 人工标注")

    class Label(BaseModel):
        worst: str
        categories: list[str] = []
        note: str = ""

    @app.get("/api/items")
    def api_items():
        # 盲评：不把裁判结论发给前端
        return [{k: r[k] for k in ("key", "source", "before", "translation", "human")}
                for r in json.loads(SAMPLE.read_text(encoding="utf-8"))]

    @app.post("/api/items/{key}")
    def api_label(key: str, label: Label):
        if label.worst not in LEVELS:
            return {"ok": False, "error": "级别无效"}
        rows = json.loads(SAMPLE.read_text(encoding="utf-8"))
        for r in rows:
            if r["key"] == key:
                r["human"] = label.model_dump()
        SAMPLE.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE
except ImportError:  # 只做抽样、计算一致性时不需要 fastapi
    pass

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MQM 人工标注</title><style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--accent:#0969da;--ok:#1a7f37}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--card:#161b22;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--accent:#4493f8;--ok:#3fb950}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"PingFang SC",sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 60px}
.top{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.guide{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;font-size:14px;color:var(--muted);margin:12px 0}
.guide b{color:var(--fg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;display:grid;gap:12px}
.ctx{color:var(--muted);font-size:14px}.src{font-size:18px;font-weight:600}.tgt{font-size:18px}
.k{font-size:12px;color:var(--muted);letter-spacing:.05em}
.levels{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.levels button,.cats button{font:inherit;padding:10px 6px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);cursor:pointer}
.levels button[aria-pressed=true],.cats button[aria-pressed=true]{border-color:var(--accent);color:var(--accent);font-weight:600}
.cats{display:flex;gap:6px;flex-wrap:wrap}.cats button{padding:4px 10px;font-size:13px}
input{font:inherit;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--fg)}
.nav{display:flex;justify-content:space-between;gap:8px}.nav button{font:inherit;padding:8px 16px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.nav .save{background:var(--accent);color:#fff;border-color:var(--accent)}
.done{color:var(--ok);font-size:13px}
</style></head><body><main>
<div class="top"><h2 style="margin:0">MQM 人工标注</h2><span id="prog"></span></div>
<div class="guide">判断<b>译文最严重的错误</b>是哪一级：<b>致命</b>＝观众得到错误信息或完全看不懂；<b>严重</b>＝意思明显偏差或缺失；
<b>轻微</b>＝不影响理解的小问题（不够自然、语气稍偏）；<b>无错误</b>＝可以直接上屏。快捷键：1 2 3 4 选级别，回车保存并下一条。</div>
<div class="card" id="card"></div></main>
<script>
const LV=[['none','无错误'],['minor','轻微'],['major','严重'],['critical','致命']];
const CAT=[['accuracy','准确（漏译/误译）'],['fluency','流畅'],['style','风格/语气'],['terminology','术语/人名']];
let items=[],i=0,sel=null,cats=new Set();
const $=s=>document.querySelector(s);
async function load(){items=await (await fetch('/api/items')).json();i=Math.max(0,items.findIndex(x=>!x.human));render()}
function render(){
  const it=items[i];const n=items.filter(x=>x.human).length;$('#prog').textContent=`已标注 ${n}/${items.length}`;
  sel=it.human?.worst||null;cats=new Set(it.human?.categories||[]);
  const c=$('#card');c.innerHTML='';
  if(it.before.length){const d=document.createElement('div');d.className='ctx';d.textContent='上文：'+it.before.join(' / ');c.append(d)}
  for(const [k,v,cls] of [['原文',it.source,'src'],['译文',it.translation,'tgt']]){const w=document.createElement('div');
    w.innerHTML=`<div class="k">${k}</div><div class="${cls}"></div>`;w.lastChild.textContent=v;c.append(w)}
  const lv=document.createElement('div');lv.className='levels';
  LV.forEach(([k,v],n)=>{const b=document.createElement('button');b.textContent=`${n+1} ${v}`;b.setAttribute('aria-pressed',sel===k);b.onclick=()=>{sel=k;render2()};lv.append(b)});
  c.append(lv);
  const ct=document.createElement('div');ct.className='cats';
  CAT.forEach(([k,v])=>{const b=document.createElement('button');b.textContent=v;b.setAttribute('aria-pressed',cats.has(k));b.onclick=()=>{cats.has(k)?cats.delete(k):cats.add(k);b.setAttribute('aria-pressed',cats.has(k))};ct.append(b)});
  c.append(ct);
  const note=document.createElement('input');note.id='note';note.placeholder='备注（可选）';note.value=it.human?.note||'';c.append(note);
  const nav=document.createElement('div');nav.className='nav';
  nav.innerHTML='<button id="prev">上一条</button><span class="done" id="st"></span><button class="save" id="save">保存并下一条</button>';
  c.append(nav);$('#prev').onclick=()=>{if(i>0){i--;render()}};$('#save').onclick=save;
}
function render2(){document.querySelectorAll('.levels button').forEach((b,n)=>b.setAttribute('aria-pressed',LV[n][0]===sel))}
async function save(){
  if(!sel){$('#st').textContent='先选一个级别';return}
  const body={worst:sel,categories:[...cats],note:$('#note').value};
  await fetch('/api/items/'+encodeURIComponent(items[i].key),{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});
  items[i].human=body;if(i<items.length-1)i++;render();
  if(items.every(x=>x.human))$('#prog').textContent='全部标注完成，可以计算一致性了';
}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
  if('1234'.includes(e.key)){sel=LV[+e.key-1][0];render2()}else if(e.key==='Enter')save()});
load();
</script></body></html>"""

if __name__ == "__main__":
    {"sample": sample, "agreement": agreement}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
