"""Retrieval over the lab documents: chunk, index, recall, rerank, generate.

What this replaces was a term-frequency count over whole files -- `score =
title.count(term) * 3 + body.count(term)`. With three documents that returned
either everything or nothing: "Lissajous" scored zero because the word does not
appear in any of them, while any query mentioning the lab returned all three in
full. Nothing about it was retrieval; it was a grep with a sort.

The five stages here are separate on purpose, because they fail differently:

  1. CHUNK    split each document at its headings, then by size with overlap.
              A whole file is the wrong unit -- half of experiment_config.md is
              irrelevant to any given question, and it crowds out the half that
              is not.
  2. INDEX    embed every chunk once and cache it keyed by content hash, so
              editing one paragraph re-embeds one chunk rather than the corpus.
  3. RECALL   dense cosine similarity AND lexical BM25, unioned. Dense alone
              loses exact identifiers -- `sine_frequency2`, `Neta_2`,
              `/data/traces` -- which is most of what gets asked about here.
              Lexical alone is what we already had.
  4. RERANK   fuse the two rankings, then optionally have the model score the
              survivors. Recall is tuned to be generous; this is where the
              precision comes from.
  5. GENERATE an answer grounded in the retrieved chunks, with citations back
              to the file and heading each claim came from.

Every stage degrades to the one before it if the embedding API is unavailable,
and says so in the result rather than silently returning worse answers.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from superradiant_assistant.config import CONFIG

# `text-embedding-004` is retired; the config still names it. These are what the
# API actually offers, best first.
EMBED_MODELS = ("gemini-embedding-001", "gemini-embedding-2")

#: Reduced from the native 3072. Retrieval quality over a corpus this size is
#: indistinguishable, the cache is a quarter the size, and cosine is faster.
#: Reduced dimensions are NOT normalised by the API, so `_normalise` must run.
EMBED_DIMS = 768

TARGET_CHARS = 700          # a chunk is about a paragraph and a half
OVERLAP_CHARS = 140         # so a fact split across a boundary survives in one
MIN_CHARS = 120             # below this a chunk is a heading with no content

CACHE_PATH = Path(CONFIG.knowledge_cache) / "rag_index.json"


# --------------------------------------------------------------------------
# 1. CHUNK
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    doc_title: str
    doc_path: str
    heading: str                 # breadcrumb, e.g. "Control stack > BLACS"
    text: str
    ordinal: int

    @property
    def uid(self) -> str:
        return hashlib.sha1(
            f"{self.doc_path}|{self.ordinal}|{self.text}".encode("utf-8")
        ).hexdigest()[:16]

    @property
    def embed_text(self) -> str:
        """What actually gets embedded.

        The heading breadcrumb is prepended so a chunk carries its context: a
        paragraph about "duration" means something different under "Blue MOT"
        than under "Cavity scan", and the body alone does not say which.
        """
        head = f"{self.doc_title} — {self.heading}" if self.heading else self.doc_title
        return f"{head}\n\n{self.text}"

    def cite(self) -> str:
        return f"{Path(self.doc_path).name}" + (f" § {self.heading}" if self.heading else "")


def _split_by_heading(markdown: str) -> List[Tuple[str, str]]:
    """[(breadcrumb, body)] — sections keyed by their heading path."""
    lines = markdown.split("\n")
    stack: List[str] = []
    out: List[Tuple[str, List[str]]] = []
    buf: List[str] = []
    crumb = ""

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if not m:
            buf.append(line)
            continue
        if buf:
            out.append((crumb, buf))
            buf = []
        depth, title = len(m.group(1)), m.group(2).strip()
        stack = stack[: depth - 1] + [title]
        crumb = " > ".join(stack)
    if buf:
        out.append((crumb, buf))
    return [(c, "\n".join(b).strip()) for c, b in out if "\n".join(b).strip()]


def _split_by_size(text: str, target: int, overlap: int) -> List[str]:
    """Fall back to paragraph packing when a section is too long to embed whole."""
    if len(text) <= target:
        return [text]
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > target:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n\n" + p) if tail else p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    # A single paragraph longer than the target still has to be cut somewhere.
    final: List[str] = []
    for ch in chunks:
        while len(ch) > target * 2:
            final.append(ch[:target])
            ch = ch[target - overlap:]
        final.append(ch)
    return final


def chunk_document(doc) -> List[Chunk]:
    """A loaded Document -> its chunks."""
    out: List[Chunk] = []
    n = 0
    for crumb, body in _split_by_heading(doc.content) or [("", doc.content)]:
        for piece in _split_by_size(body, TARGET_CHARS, OVERLAP_CHARS):
            if len(piece.strip()) < MIN_CHARS:
                continue
            out.append(Chunk(doc_title=doc.title, doc_path=str(doc.path),
                             heading=crumb, text=piece.strip(), ordinal=n))
            n += 1
    return out


def chunk_all(docs) -> List[Chunk]:
    chunks: List[Chunk] = []
    for d in docs:
        chunks.extend(chunk_document(d))
    return chunks


# --------------------------------------------------------------------------
# 2. INDEX
# --------------------------------------------------------------------------

def _normalise(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _embed(texts: Sequence[str], task_type: str,
           model: Optional[str] = None) -> Optional[List[List[float]]]:
    """Embed a batch. None means the API is unreachable — the caller degrades."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None
    key = CONFIG.gemini_api_key
    if not key:
        return None

    client = genai.Client(api_key=key)
    cfg = types.EmbedContentConfig(task_type=task_type,
                                   output_dimensionality=EMBED_DIMS)
    for candidate in ([model] if model else EMBED_MODELS):
        try:
            out: List[List[float]] = []
            for i in range(0, len(texts), 100):          # API batch ceiling
                resp = client.models.embed_content(
                    model=candidate, contents=list(texts[i:i + 100]), config=cfg)
                out.extend(_normalise(list(e.values)) for e in resp.embeddings)
            return out
        except Exception:
            continue
    return None


@dataclass
class Index:
    chunks: List[Chunk] = field(default_factory=list)
    vectors: Dict[str, List[float]] = field(default_factory=dict)
    model: str = ""
    built_at: float = 0.0

    @property
    def dense_ready(self) -> bool:
        return bool(self.vectors) and len(self.vectors) >= len(self.chunks)

    def describe(self) -> str:
        docs = len({c.doc_path for c in self.chunks})
        mode = ("dense+lexical" if self.dense_ready else "lexical only "
                "(embeddings unavailable)")
        return (f"{len(self.chunks)} chunks from {docs} documents | {mode}"
                + (f" | {self.model}" if self.model else ""))


def _load_cache() -> Dict[str, List[float]]:
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if raw.get("dims") != EMBED_DIMS:
        return {}                                    # dimensionality changed
    return raw.get("vectors", {})


def _save_cache(vectors: Dict[str, List[float]], model: str) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(
            {"dims": EMBED_DIMS, "model": model, "saved": time.time(),
             "vectors": vectors}), encoding="utf-8")
    except OSError:
        pass


def build_index(docs=None, allow_embedding: bool = True) -> Index:
    """Chunk the corpus and attach a vector to each chunk.

    Only chunks whose text has changed are embedded: the cache is keyed by a
    hash of the chunk itself, so editing one paragraph costs one embedding.
    """
    if docs is None:
        from superradiant_assistant.knowledge.loader import load_knowledge_base
        docs = load_knowledge_base()

    chunks = chunk_all(docs)
    index = Index(chunks=chunks, built_at=time.time())
    if not chunks or not allow_embedding:
        return index

    cached = _load_cache()
    index.vectors = {c.uid: cached[c.uid] for c in chunks if c.uid in cached}

    missing = [c for c in chunks if c.uid not in index.vectors]
    if missing:
        got = _embed([c.embed_text for c in missing], "RETRIEVAL_DOCUMENT")
        if got:
            for c, v in zip(missing, got):
                index.vectors[c.uid] = v
            index.model = EMBED_MODELS[0]
            _save_cache(index.vectors, index.model)
    elif index.vectors:
        index.model = EMBED_MODELS[0]
    return index


# --------------------------------------------------------------------------
# 3. RECALL
# --------------------------------------------------------------------------

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")


def _terms(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s) if len(t) > 1]


class _BM25:
    """Okapi BM25 over the chunks.

    Kept alongside the dense scores rather than replaced by them: the questions
    asked here are full of exact tokens -- `sine_frequency2`, `Neta_2`,
    `/data/traces`, `RigolDG1022` -- and an embedding of a rare identifier is
    close to an embedding of any other rare identifier.
    """

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [_terms(c.embed_text) for c in chunks]
        self.len = [len(d) or 1 for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        self.df: Dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = max(1, len(self.docs))

    def score(self, query: str) -> List[float]:
        q = _terms(query)
        out = [0.0] * len(self.docs)
        for t in set(q):
            df = self.df.get(t, 0)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, d in enumerate(self.docs):
                f = d.count(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))          # both are unit vectors


def recall(index: Index, query: str, k: int = 20) -> Dict[str, object]:
    """Candidate chunks from both retrievers, each with its own ranking."""
    bm = _BM25(index.chunks)
    lex_scores = bm.score(query)
    lex_rank = sorted(range(len(index.chunks)), key=lambda i: -lex_scores[i])
    lex_rank = [i for i in lex_rank if lex_scores[i] > 0][:k]

    dense_rank: List[int] = []
    dense_scores = [0.0] * len(index.chunks)
    if index.dense_ready:
        qv = _embed([query], "RETRIEVAL_QUERY")
        if qv:
            q = qv[0]
            for i, c in enumerate(index.chunks):
                v = index.vectors.get(c.uid)
                if v:
                    dense_scores[i] = _cosine(q, v)
            dense_rank = sorted(range(len(index.chunks)),
                                key=lambda i: -dense_scores[i])[:k]

    return {"lexical": lex_rank, "dense": dense_rank,
            "lex_scores": lex_scores, "dense_scores": dense_scores,
            "dense_used": bool(dense_rank)}


# --------------------------------------------------------------------------
# 4. RERANK
# --------------------------------------------------------------------------

RRF_K = 60          # the standard damping constant for reciprocal rank fusion


def rerank_fusion(index: Index, recalled: Dict[str, object], query: str,
                  top_k: int = 5) -> List[Tuple[float, Chunk]]:
    """Reciprocal rank fusion, plus a boost for exact identifier matches.

    RRF combines the two rankings without needing their scores to be on the
    same scale, which they are not: cosine sits in [0,1] and BM25 is unbounded.
    """
    scores: Dict[int, float] = {}
    for key in ("lexical", "dense"):
        for rank, i in enumerate(recalled[key]):        # type: ignore[index]
            scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)

    # An exact occurrence of a rare token from the query is strong evidence that
    # embeddings do not capture. Identifiers are what this corpus is made of.
    rare = [t for t in _terms(query) if len(t) > 4 and "_" in t or t.isupper()]
    if rare:
        for i, c in enumerate(index.chunks):
            body = c.embed_text.lower()
            hits = sum(1 for t in rare if t.lower() in body)
            if hits:
                scores[i] = scores.get(i, 0.0) + 0.02 * hits

    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    return [(s, index.chunks[i]) for i, s in ordered]


_RERANK_SYSTEM = """You score how well each passage answers a question.

Return one line per passage, `id: score`, score 0-10, nothing else.
10 = contains the answer. 5 = related background. 0 = unrelated.
Judge only what is in the passage; do not use outside knowledge."""


def rerank_llm(query: str, candidates: List[Tuple[float, Chunk]],
               top_k: int = 5, model: str = "gemini-3.6-flash",
               cost_tracker=None) -> Optional[List[Tuple[float, Chunk]]]:
    """Have the model judge each candidate against the query.

    This is the cross-encoder stage: the fused ranking scores query and passage
    separately and never compares them directly. Returns None if the call
    fails, so the fused order stands.
    """
    if not candidates:
        return []
    listing = "\n\n".join(
        f"[{i}] ({c.cite()})\n{c.text[:900]}" for i, (_, c) in enumerate(candidates))
    prompt = f"QUESTION\n{query}\n\nPASSAGES\n{listing}"
    try:
        from superradiant_assistant.llm.client import GeminiClient
        resp = GeminiClient(model=model, cost_tracker=cost_tracker).generate(
            prompt, system=_RERANK_SYSTEM, temperature=0.0, max_retries=2)
    except Exception:
        return None

    scored: Dict[int, float] = {}
    for line in (resp.text or "").splitlines():
        m = re.match(r"\s*\[?(\d+)\]?\s*[:.\-]\s*([0-9.]+)", line)
        if m:
            i = int(m.group(1))
            if 0 <= i < len(candidates):
                scored[i] = float(m.group(2))
    if not scored:
        return None
    order = sorted(scored.items(), key=lambda kv: -kv[1])
    return [(s, candidates[i][1]) for i, s in order if s > 0][:top_k]


# --------------------------------------------------------------------------
# 5. GENERATE
# --------------------------------------------------------------------------

_ANSWER_SYSTEM = """Answer the question from the numbered passages given.

Cite the passage number in square brackets after each claim, e.g. [2]. A
citation means that passage says it — never attach one to a claim it does not
make.

If the passages do not cover part of the question, say which part is missing.
You may then answer it from your own knowledge, prefixed "not in the documents:".
Be brief."""


def generate(query: str, hits: List[Tuple[float, Chunk]],
             model: str = "gemini-3.6-flash", cost_tracker=None) -> Optional[str]:
    """A grounded answer with citations, or None if the call fails."""
    if not hits:
        return None
    passages = "\n\n".join(
        f"[{i + 1}] ({c.cite()})\n{c.text}" for i, (_, c) in enumerate(hits))
    try:
        from superradiant_assistant.llm.client import GeminiClient
        resp = GeminiClient(model=model, cost_tracker=cost_tracker).generate(
            f"QUESTION\n{query}\n\nPASSAGES\n{passages}",
            system=_ANSWER_SYSTEM, temperature=0.1, max_retries=2)
        return (resp.text or "").strip() or None
    except Exception:
        return None


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

#: Appended to every retrieval result.
#:
#: The rule is about ATTRIBUTION, not about what may be said. Asked how the
#: atoms are cooled, the model answered with 556 nm, the 1S0->3P1 transition,
#: microkelvin temperatures and a quantum non-demolition measurement, wrote
#: "based on the lab documentation", and cited lab_overview.md. None of those
#: four things appear anywhere in the corpus. They are also all true of this
#: apparatus -- that is what makes the citation dangerous rather than merely
#: wrong: it is indistinguishable from the parts that were retrieved.
#:
#: Forbidding everything outside the passages would be the wrong fix. Physics
#: reasoning, interpretation of a result, and suggesting the next measurement
#: are the job. What has to stop is the two being merged under one attribution.
GROUNDING_RULE = """\
---
Attribution rule for the passages above.

Cite a file or section ONLY for something that passage actually states. Quote or
closely paraphrase; do not attach a citation to a claim the passage does not
make, even a claim you are confident is true.

You may add your own physics knowledge, interpretation and recommendations —
that is wanted, especially when discussing results. Mark it as yours: "the
documents do not say, but ...", "from general Yb physics ...", "my reading of
this is ...". The operator needs to be able to tell which parts they can check
against a file and which parts rest on you.

If the passages do not answer the question, say so first, then answer from your
own knowledge if you can — clearly labelled as such."""


_INDEX: Optional[Index] = None


def get_index(rebuild: bool = False) -> Index:
    global _INDEX
    if _INDEX is None or rebuild:
        _INDEX = build_index()
    return _INDEX


def search(query: str, top_k: int = 5, recall_k: int = 20,
           rerank: str = "fusion", synthesise: bool = False,
           cost_tracker=None) -> str:
    """The whole pipeline, rendered for the model.

    `rerank='llm'` adds a model call and roughly a second; `fusion` is free.
    The default is fusion because this tool is called several times per turn and
    latency is already the loudest complaint about the agent.
    """
    index = get_index()
    if not index.chunks:
        return ("found 0 chunks — the knowledge base is empty. Documents live "
                f"in {Path(CONFIG.knowledge_root) / 'handwritten'}")

    recalled = recall(index, query, k=recall_k)
    fused = rerank_fusion(index, recalled, query, top_k=max(top_k, 8))

    stage = "fusion(dense+bm25)" if recalled["dense_used"] else "fusion(bm25 only)"
    rerank_note = ""
    if rerank == "llm":
        judged = rerank_llm(query, fused, top_k=top_k, cost_tracker=cost_tracker)
        if judged is not None:
            fused, stage = judged, stage + " -> llm rerank"
        else:
            # Silently keeping the fused order looked identical to a successful
            # rerank: the only difference was four words in a header the caller
            # never sees in full. A stage that did not run has to say so.
            stage += " -> llm rerank FAILED"
            rerank_note = ("NOTE: the llm rerank did not run (the call failed) — "
                           "these passages are in fused order, not judged order. "
                           "Do not report this as a reranked search.")
    hits = fused[:top_k]

    if not hits:
        return (f"found 0 passages for '{query}' | {index.describe()} | "
                f"retrieval={stage}\nNothing in the knowledge base matched. Say "
                f"that the documents do not cover this rather than answering "
                f"from general knowledge.")

    lines = [f"found {len(hits)} passages | {index.describe()} | retrieval={stage}"]
    if not recalled["dense_used"]:
        lines.append("NOTE: embeddings unavailable, so this was keyword-only "
                     "retrieval. Exact wording matters more than usual.")
    if rerank_note:
        lines.append(rerank_note)
    if synthesise:
        grounded = generate(query, hits, cost_tracker=cost_tracker)
        if grounded:
            lines += ["", "GROUNDED ANSWER (cites the passages below)", grounded]
    lines.append("")
    for i, (score, c) in enumerate(hits, 1):
        lines.append(f"[{i}] {c.cite()}   (score {score:.3f})")
        lines.append(c.text)
        lines.append("")
    lines.append(GROUNDING_RULE)
    return "\n".join(lines)
