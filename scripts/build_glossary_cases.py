"""生成术语一致性评测用例：按难点类型让大模型写剧中台词并标注涉及的术语，再用规则校验标注。

难点类型（字幕本地化中的真实问题）：
- 全名：直接出现标准写法
- 别名：用昵称、尊称指代（"婉儿""太子殿下"）
- 称谓：当面称呼（"殿下""师兄"），译法要统一
- 多术语：一句里有两三个术语
- 同形词：设定表里的词以普通含义出现（"一世长安"），不应套用译名
- 无术语：普通台词，检索应该返回空
- 依赖上文：当前句只用简称，上文给出是谁
- 未登记叫法：用设定表里没有列出的叫法指代条目（"林家丫头""萧太子"），关键词检索查不到，是检索的真正难点

校验规则：标注的每个条目，其写法（中文名或别名）必须出现在台词里；同形词和无术语用例不能有标注；
台词里出现了某条目的写法却没标注的，丢弃（标注不完整）。只保留通过校验的用例。

人工复核（data/glossary/review.json）：规则查不出“写法出现了但是普通含义”的情况，生成模型自己也常把
“白玉簪子”“天机不可泄露”误标成术语。复核后标准答案统一由规则重算（当前句中出现的条目），再去掉人工标出的
同形词用法、删除有歧义的用例；上文里的条目不计入（不出现在当前句译文里）。

运行：python scripts/build_glossary_cases.py          # 生成 + 规则校验 + 应用人工复核
      python scripts/build_glossary_cases.py review   # 只重新应用人工复核
      python scripts/build_glossary_cases.py unlisted # 追加"未登记叫法"用例
输出：data/glossary/cases.json
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.glossary import Glossary  # noqa: E402

OUT = ROOT / "data" / "glossary" / "cases.json"
KINDS = {
    "全名": "台词里直接出现 1 个条目的中文名（zh 字段）",
    "别名": "台词里用条目的别名（aliases）指代人物或事物，不出现中文全名",
    "称谓": "台词是当面称呼对方，用到称谓类条目（师兄、殿下、娘娘、姑娘）",
    "多术语": "台词里出现 2～3 个不同条目",
    "同形词": "台词里出现某个条目 ambiguous 字段所说的普通含义用法（如“一世长安”“天机不可泄露”“白玉簪子”），此时不是剧中专有名词，terms 必须为空",
    "无术语": "普通生活台词，不出现设定表里的任何词，terms 为空",
    "依赖上文": "给出 1 句上文（before）点明人物，当前台词只用简称或称谓指代他/她",
}
UNLISTED = ("未登记叫法", "用设定表里没有列出的叫法指代某个条目（如“林家丫头”“萧太子”“沈老七”“那位丞相千金”），"
            "台词里不能出现该条目的中文名或任何别名；terms 只填被指代的那个条目 id（其他条目的写法也尽量不要出现）")
PER_KIND = 32


def generate(client, bible, kind, desc):
    entries = "\n".join(
        f"- {e['id']}｜{e['zh']}｜别名：{'、'.join(e['aliases']) or '无'}｜{e['note']}"
        + (f"｜同形词：{e['ambiguous']}" if e.get("ambiguous") else "") for e in bible["entries"])
    prompt = f"""你在为古装剧《长安夜行》写字幕测试数据。剧集设定表：
{entries}

请写 {PER_KIND} 句不同的剧中台词，类型：{kind}——{desc}。
要求：口语化、像真实剧集台词、每句 8～30 字、互不重复、人物和场景多样。
terms 列出这句台词涉及的条目 id（专有名词意义上的），before 是上文台词列表（不需要上文时为空列表）。
只输出 JSON：{{"cases": [{{"text": "...", "before": [], "terms": ["id", ...]}}]}}"""
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-chat"), temperature=0.9,
        response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
    return json.loads(resp.choices[0].message.content).get("cases", [])


def validate(case, kind, g):
    """返回 (是否保留, 原因)。"""
    ids = set(case.get("terms", []))
    if not all(i in g.by_id for i in ids):
        return False, "标注了不存在的条目"
    if kind == "未登记叫法":
        if len(ids) != 1:
            return False, "未登记叫法应只标注一个条目"
        e = g.by_id[next(iter(ids))]
        if any(s in case["text"] for s in [e["zh"], *e["aliases"]]):
            return False, "出现了已登记的写法"
        return True, ""
    found = {e["id"] for e in g.match(case["text"])}
    if kind in ("同形词", "无术语"):
        if ids:
            return False, "该类型不应有标注"
        if kind == "同形词" and not found:
            return False, "没有出现同形词"
        if kind == "无术语" and found:
            return False, "无术语用例里出现了术语写法"
        return True, ""
    if not ids:
        return False, "缺少标注"
    if kind == "依赖上文":
        context_found = {e["id"] for e in g.match(" ".join(case.get("before", [])) + case["text"])}
        if not ids <= context_found:
            return False, "标注的条目在上文和台词里都没出现"
        return True, ""
    if not ids <= found:
        return False, "标注的条目没有出现在台词里"
    if found - ids:
        return False, "台词里有未标注的条目"
    return True, ""


REVIEW = ROOT / "data" / "glossary" / "review.json"


def apply_review(cases, g):
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    out = []
    for c in cases:
        if c["id"] in review["drop"]:
            continue
        terms = {e["id"] for e in g.match(c["text"])} - set(review["remove"].get(c["id"], []))
        terms |= set(review.get("add", {}).get(c["id"], []))   # 未登记叫法：规则匹配不到，按人工标注补上
        c = {**c, "terms": sorted(terms)}
        if not terms:
            c["kind"] = "同形词" if g.match(c["text"]) else "无术语"
        elif c["kind"] == "同形词":     # 生成时归为同形词，复核发现其实是专有名词
            c["kind"] = "全名"
        elif c["kind"] == "依赖上文":
            c["kind"] = "带上文"
        out.append(c)
    return out


def main():
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    g = Glossary.load()
    cases, dropped = [], {}
    for kind, desc in KINDS.items():
        raw = generate(client, g.bible, kind, desc)
        kept = 0
        for c in raw:
            ok, why = validate(c, kind, g)
            if ok and c["text"] not in {x["text"] for x in cases}:
                kept += 1
                cases.append({"id": f"g-{len(cases) + 1:03d}", "kind": kind, "text": c["text"],
                              "before": c.get("before", []) if kind == "依赖上文" else [],
                              "terms": sorted(set(c.get("terms", [])))})
            elif not ok:
                dropped[why] = dropped.get(why, 0) + 1
        print(f"{kind}：生成 {len(raw)}，保留 {kept}", file=sys.stderr)
    cases = apply_review(cases, g)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"共 {len(cases)} 条；丢弃原因：{dropped}")


def add_unlisted():
    """追加"未登记叫法"用例：标注写入 review.json 的 add（规则无法从字面验证，靠人工复核）。"""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    g = Glossary.load()
    cases = json.loads(OUT.read_text(encoding="utf-8"))
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    kind, desc = UNLISTED
    raw = generate(client, g.bible, kind, desc)
    n = int(cases[-1]["id"][2:])
    for c in raw:
        ok, why = validate(c, kind, g)
        if ok and c["text"] not in {x["text"] for x in cases}:
            n += 1
            cid = f"g-{n:03d}"
            cases.append({"id": cid, "kind": kind, "text": c["text"], "before": [], "terms": []})
            review.setdefault("add", {})[cid] = c["terms"]
    REVIEW.write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT.write_text(json.dumps(apply_review(cases, g), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"追加后共 {len(cases)} 条，请人工复核 review.json 里的 add")


def review_only():
    g = Glossary.load()
    cases = apply_review(json.loads(OUT.read_text(encoding="utf-8")), g)
    OUT.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"共 {len(cases)} 条：{dict(Counter(c['kind'] for c in cases))}")


if __name__ == "__main__":
    {"review": review_only, "unlisted": add_unlisted}.get((sys.argv[1:] or [""])[0], main)()
