"""④ 人工审核（网页版）：每条用例一张卡片，点按钮决定是否采纳。

启动：.venv/bin/uvicorn scripts.review_web:app --port 8001
打开：http://127.0.0.1:8001
每次点按钮都会立即保存到 data/corpus/candidates_labeled.json。
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
LABELED = ROOT / "data" / "corpus" / "candidates_labeled.json"
CATEGORIES = ["网络梗", "方言", "成语俗语", "人名地名", "语气情绪", "需要上下文", "普通句子"]

app = FastAPI(title="测试用例审核")


class Decision(BaseModel):
    accept: bool | None
    requirement: str
    category: str
    reject_reason: str = ""


def load():
    return json.loads(LABELED.read_text(encoding="utf-8"))


@app.get("/api/items")
def items():
    return {"items": load(), "categories": CATEGORIES}


@app.post("/api/items/{item_id}")
def decide(item_id: str, d: Decision):
    data = load()
    for c in data:
        if c["id"] == item_id:
            if d.requirement != c["requirement"] or d.category != c["category"]:
                c["edited"] = True
            c.update(accept=d.accept, requirement=d.requirement, category=d.category,
                     reject_reason=d.reject_reason)
            LABELED.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return {"ok": True}
    raise HTTPException(404, f"找不到 {item_id}")


@app.get("/", response_class=HTMLResponse)
def page():
    return PAGE


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>测试用例审核</title>
<style>
  :root { --bg:#f6f7f9; --card:#fff; --text:#1f2328; --muted:#656d76; --line:#d0d7de;
          --ok:#1a7f37; --ok-bg:#dafbe1; --no:#cf222e; --no-bg:#ffebe9; --accent:#0969da; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.6 -apple-system,"PingFang SC",sans-serif; }
  header { position:sticky; top:0; z-index:1; background:var(--card); border-bottom:1px solid var(--line);
           padding:12px 16px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  header h1 { font-size:17px; margin:0; }
  .bar { flex:1; min-width:160px; height:8px; background:#eaeef2; border-radius:4px; overflow:hidden; }
  .bar i { display:block; height:100%; background:var(--ok); transition:width .2s; }
  .stat { color:var(--muted); font-size:14px; }
  label.f { font-size:14px; color:var(--muted); display:flex; gap:4px; align-items:center; }
  main { max-width:820px; margin:0 auto; padding:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line);
          border-radius:8px; padding:14px 16px; margin-bottom:12px; }
  .card.yes { border-left-color:var(--ok); } .card.no { border-left-color:var(--no); opacity:.7; }
  .meta { display:flex; gap:8px; align-items:center; font-size:13px; color:var(--muted); margin-bottom:6px; flex-wrap:wrap; }
  .src { font-size:18px; font-weight:600; margin:2px 0 10px; }
  textarea { width:100%; min-height:60px; font:inherit; padding:8px; border:1px solid var(--line); border-radius:6px; resize:vertical; }
  select, input[type=text] { font:inherit; padding:4px 6px; border:1px solid var(--line); border-radius:6px; }
  .actions { display:flex; gap:8px; margin-top:10px; flex-wrap:wrap; align-items:center; }
  button { font:inherit; padding:6px 16px; border-radius:6px; border:1px solid var(--line); background:var(--card); cursor:pointer; }
  button.yes.on { background:var(--ok-bg); border-color:var(--ok); color:var(--ok); font-weight:600; }
  button.no.on { background:var(--no-bg); border-color:var(--no); color:var(--no); font-weight:600; }
  .reason { flex:1; min-width:180px; }
  .saved { font-size:13px; color:var(--ok); }
</style></head><body>
<header>
  <h1>测试用例审核</h1>
  <div class="bar"><i id="bar"></i></div>
  <span class="stat" id="stat"></span>
  <label class="f"><input type="checkbox" id="pendingOnly"> 只看未审核</label>
</header>
<main id="list"></main>
<script>
let items = [], categories = [];
const $ = (s, el = document) => el.querySelector(s);

async function load() {
  const r = await fetch('/api/items').then(r => r.json());
  items = r.items; categories = r.categories; render();
}

function stats() {
  const done = items.filter(c => c.accept !== null).length;
  const yes = items.filter(c => c.accept === true).length;
  $('#bar').style.width = (done / items.length * 100) + '%';
  $('#stat').textContent = `已审核 ${done}/${items.length}，采纳 ${yes}`;
}

function render() {
  const only = $('#pendingOnly').checked;
  $('#list').innerHTML = '';
  for (const c of items) {
    if (only && c.accept !== null) continue;
    const card = document.createElement('div');
    card.className = 'card ' + (c.accept === true ? 'yes' : c.accept === false ? 'no' : '');
    card.innerHTML = `
      <div class="meta"><b>${c.id}</b>
        <select class="cat">${categories.map(x => `<option ${x === c.category ? 'selected' : ''}>${x}</option>`).join('')}</select>
        <span>难度 ${'★'.repeat(c.difficulty || 1)}</span></div>
      <div class="src"></div>
      <textarea class="req"></textarea>
      <div class="actions">
        <button class="yes ${c.accept === true ? 'on' : ''}">✓ 采纳</button>
        <button class="no ${c.accept === false ? 'on' : ''}">✗ 不采纳</button>
        <input type="text" class="reason" placeholder="不采纳原因（可选）">
        <span class="saved"></span>
      </div>`;
    $('.src', card).textContent = c.source;
    $('.req', card).value = c.requirement;
    $('.reason', card).value = c.reject_reason || '';
    $('.yes', card).onclick = () => save(c, card, true);
    $('.no', card).onclick = () => save(c, card, false);
    $('#list').appendChild(card);
  }
  stats();
}

async function save(c, card, accept) {
  const body = { accept, requirement: $('.req', card).value.trim(), category: $('.cat', card).value,
                 reject_reason: $('.reason', card).value.trim() };
  const r = await fetch('/api/items/' + c.id, { method: 'POST', headers: { 'content-type': 'application/json' },
                                                body: JSON.stringify(body) });
  if (!r.ok) { $('.saved', card).textContent = '保存失败'; return; }
  Object.assign(c, body);
  card.className = 'card ' + (accept ? 'yes' : 'no');
  $('.yes', card).classList.toggle('on', accept);
  $('.no', card).classList.toggle('on', !accept);
  $('.saved', card).textContent = '已保存';
  stats();
}

$('#pendingOnly').onchange = render;
load();
</script></body></html>"""
