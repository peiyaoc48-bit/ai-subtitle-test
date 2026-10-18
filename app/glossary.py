"""剧集术语表（show bible）与检索。

字幕本地化团队会为每部剧维护一份设定表：人名、称谓、地名等统一怎么译。翻译每句字幕前，
先从表里检索出这句可能涉及的条目，再连同译法一起交给模型（检索增强，RAG）。

三种检索方式，用 scripts/rag_eval.py 对比：
- KeywordRetriever：按中文名和别名做字符串匹配，最长匹配优先（"太子殿下"优先于"殿下"）。
  召回高、速度快，但同形词（"一世长安"）也会被检索出来。
- EmbeddingRetriever：把句子和条目说明转成向量，按相似度取前 k 个。能理解语义，但可能漏掉字面出现的人名。
- HybridRetriever：关键词命中的条目全部保留，再用向量相似度补充。

只检索当前句：上文里提到的人物不会出现在当前句的译文里（"殿下"译 Your Highness，不需要知道是谁），
带进来只会增加干扰。
"""
import json
import math
import os
from pathlib import Path

BIBLE = Path(__file__).parent.parent / "data" / "glossary" / "bible.json"


class Glossary:
    def __init__(self, bible: dict):
        self.bible = bible
        self.entries = bible["entries"]
        self.by_id = {e["id"]: e for e in self.entries}
        # 每种写法对应的条目；按长度降序，匹配时长的优先
        self.surfaces = sorted(((s, e) for e in self.entries for s in [e["zh"], *e["aliases"]]),
                               key=lambda x: -len(x[0]))

    @classmethod
    def load(cls, path: Path = BIBLE) -> "Glossary":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def match(self, text: str) -> list[dict]:
        """最长匹配：已被长写法占用的位置，短写法不再匹配（"白玉关"里不再匹配"白玉"）。按在句中出现的先后排序。"""
        taken = [False] * len(text)
        first = {}   # 条目 id -> (首次出现位置, 条目)
        for surface, e in self.surfaces:
            start = text.find(surface)
            while start != -1:
                end = start + len(surface)
                if not any(taken[start:end]):
                    taken[start:end] = [True] * len(surface)
                    if e["id"] not in first or start < first[e["id"]][0]:
                        first[e["id"]] = (start, e)
                start = text.find(surface, start + 1)
        return [e for _, e in sorted(first.values(), key=lambda x: x[0])]


def format_entries(entries: list[dict]) -> str:
    """给模型看的术语表：写法、译法、说明，有同形词时一并提示。"""
    lines = []
    for e in entries:
        names = "、".join([e["zh"], *e["aliases"]])
        line = f"- {names} → {e['en']}（{e['type']}；{e['note']}）"
        if e.get("ambiguous"):
            line += f"【注意：{e['ambiguous']}，这种普通含义不要套用译名】"
        lines.append(line)
    return "\n".join(lines)


class KeywordRetriever:
    name = "keyword"

    def __init__(self, glossary: Glossary):
        self.g = glossary

    def __call__(self, text: str) -> list[dict]:
        return self.g.match(text)


class EmbeddingRetriever:
    """向量检索。embed 是一个函数：list[str] -> list[list[float]]，便于测试时替换成假的。"""
    name = "embedding"

    def __init__(self, glossary: Glossary, embed, k: int = 3, threshold: float = 0.0):
        self.g, self.embed, self.k, self.threshold = glossary, embed, k, threshold
        docs = [f"{e['zh']}（{'、'.join(e['aliases'])}）：{e['type']}，{e['note']}" for e in glossary.entries]
        self.vectors = embed(docs)

    def __call__(self, text: str) -> list[dict]:
        q = self.embed([text])[0]
        scored = sorted(((cosine(q, v), e) for v, e in zip(self.vectors, self.g.entries)), key=lambda x: -x[0])
        return [e for s, e in scored[:self.k] if s >= self.threshold]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, keyword: KeywordRetriever, embedding: EmbeddingRetriever):
        self.keyword, self.embedding = keyword, embedding

    def __call__(self, text: str) -> list[dict]:
        found = self.keyword(text)
        ids = {e["id"] for e in found}
        return found + [e for e in self.embedding(text) if e["id"] not in ids]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def openai_embedder(model: str | None = None):
    """调用兼容 OpenAI 的向量接口（默认用第二裁判的阿里云百炼 Key）。带缓存，同一文本只算一次。"""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["JUDGE2_API_KEY"], base_url=os.environ["JUDGE2_BASE_URL"])
    model = model or os.getenv("EMBED_MODEL", "text-embedding-v4")
    cache: dict[str, list[float]] = {}

    def embed(texts: list[str]) -> list[list[float]]:
        todo = [t for t in dict.fromkeys(texts) if t not in cache]
        for i in range(0, len(todo), 10):   # 百炼一次最多 10 条
            resp = client.embeddings.create(model=model, input=todo[i:i + 10])
            for t, d in zip(todo[i:i + 10], resp.data):
                cache[t] = d.embedding
        return [cache[t] for t in texts]

    return embed
