---
name: lab-knowledge-search
description: How retrieval over the lab documents works, how to cite it honestly, and when to use a different tool instead.
apparatus: cesium
tags: search, rag, documentation, retrieval, citation
---

# Searching lab knowledge

`search_lab_knowledge` retrieves passages from the handwritten lab documents.

## Check what the corpus actually contains

The corpus is whatever markdown this lab has put in `knowledge/handwritten/`. A
fresh checkout has none, and the tool says so rather than inventing passages.

**It is not guaranteed to describe the bench you are running.** A lab that
inherited its documents from another apparatus has a corpus about that one. For
anything about *this* apparatus — its sequences, its analyses, its wiring, what
a shot file contains — use `list_scripts`, `read_lab_file` and `inspect_shot`.
Retrieval always returns the nearest passage it has, and a passage about the
wrong apparatus is worse than nothing because it looks like an answer.

## How retrieval works

Ask in your own words. Retrieval is hybrid:

- **dense** — the query and every passage are embedded, and matched by meaning.
  A paraphrase works; you do not have to guess the document's wording.
- **BM25** — lexical scoring, which is what catches an exact identifier: a
  global's or a metric's literal name. An embedding cannot tell two rare
  identifiers apart; this can.

Both run on every query and their rankings are fused. Documents are split into
passages at their headings, so a result is a section, not a whole file.

Read the first line of the result. It states what actually ran:

    found 5 passages | 21 chunks from 3 documents | dense+lexical |
    gemini-embedding-001 | retrieval=fusion(dense+bm25)

- `bm25 only` means embeddings were unavailable — exact wording matters more.
- `llm rerank FAILED` means that stage did not run; the order is the fused one.

`rerank='llm'` adds a stage that judges each passage against the question. It
costs about half a minute, so use it only when the fast path answered badly.
`answer=true` returns a synthesised, cited answer alongside the passages.

## Citing it honestly

Every passage comes with the file and heading it came from. Cite those — but
only for what the passage actually says.

This matters more than it sounds, and the failure has happened here. Asked how
a stage of the experiment works, it is very easy to write "based on the lab
documentation" and then supply the transition, the wavelength and the
measurement scheme — every one of them true of the experiment, and **none of
them anywhere in the corpus.** A citation attached to a claim the file does not
make is worse than no citation, because the operator cannot tell it apart from
the parts that were actually retrieved.

You are welcome — encouraged — to add physics knowledge, interpretation and
recommendations. Mark them as yours: "the documents do not say, but ...",
"from general atomic physics ...". The operator needs to know which half they
can check against a file.

If the passages do not answer the question, say so first.

## When not to search

Retrieval covers what is written down. It does not cover:

- **the current state of the apparatus** — `get_runmanager_globals`
- **what recent shots measured** — `read_shot_results`, `inspect_shot`
- **what code exists here and what it does** — `list_scripts`, `read_lab_file`
- **why the operator wants something** — ask them

Never take a parameter value from documentation when you can read the live one.
A documented "typical" value describes what was typical when the note was
written.
