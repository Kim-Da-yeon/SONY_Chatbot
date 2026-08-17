# SONY Product Manual RAG Chatbot

**A Korean-language retrieval-augmented chatbot that answers product questions from Sony manuals, with page-level source attribution and a web-search fallback**

> Coursework project · Retrieval: BAAI/bge-m3 + Chroma · Generation: KoAlpaca-llama-1-7b · UI: Streamlit

---

## Overview

Sony ships a separate PDF manual per product, and a user with a question ("how do
I unpair the Bluetooth on my speaker?") has to find the right manual and the right
page. This project indexes a set of manuals and answers such questions in Korean,
citing the file and page the answer came from.

Two things distinguish it from a plain vector-search RAG:

- **Product scoping.** The user picks their product from a dropdown. Retrieved
  chunks are then filtered by comparing that product name against the source
  filename, so a question about the LSPX-S3 speaker is not answered from the
  ILCE-7CM2 camera manual.
- **Web-search fallback.** If nothing survives the product filter, the query is
  forwarded to Tavily web search instead of letting the model answer from
  nothing.

## How it works

```
manual PDFs ─► PyMuPDFLoader ─► RecursiveCharacterTextSplitter ─► bge-m3 ─► Chroma
                (per-page          (chunk 500, overlap 50)        embeddings   (persisted)
                 metadata:
                 file_name,
                 page_number)

question + product ─► Chroma top-5 ─► product filter ─► KoAlpaca-7b ─► answer + source
                                       (cosine ≥ 0.7)       │
                                       └─ nothing passes ───┴─► Tavily web search
```

| Stage | Component | Setting |
|---|---|---|
| Chunking | `RecursiveCharacterTextSplitter` | size 500, overlap 50 |
| Embedding | `BAAI/bge-m3` | normalized |
| Vector store | Chroma | persisted to disk |
| Retrieval | similarity | top `k = 5` |
| Product filter | `all-MiniLM-L6-v2` cosine | threshold `0.7` |
| Generation | `KoAlpaca-llama-1-7b` | `max_new_tokens = 200`, fp16 on GPU |
| Fallback | Tavily search | `k = 2` |

Every answer is returned with the `file_name` and `page_number` of each chunk
that fed it, so a claim can be traced back to a specific manual page.

## Repository Structure

```
.
├── build_vectorstore.py   # Ingest: PDFs -> chunks -> embeddings -> Chroma
├── app.py                 # Streamlit app (product dropdown + question box)
├── query_test.py          # Headless single-query test of the same pipeline
└── requirements.txt
```

## Setup

The manuals are **not** included in this repository — they are Sony's
copyrighted documents. Download the ones you want from
[Sony's official support site](https://www.sony.com/electronics/support) and put
them in `data/`.

```bash
pip install -r requirements.txt

# Tavily key for the web-search fallback (https://tavily.com)
export TAVILY_API_KEY="your-key-here"

# Optional — defaults shown
export SONY_DATA_DIR="data"                          # where the PDFs live
export SONY_VECTORSTORE_DIR="data/vectorstore"       # where Chroma persists
export SONY_MODEL_PATH="beomi/KoAlpaca-llama-1-7b"   # or a local model directory
```

```bash
python build_vectorstore.py     # index every *.pdf in SONY_DATA_DIR
streamlit run app.py            # launch the UI
python query_test.py            # or run one hard-coded query headlessly
```

`build_vectorstore.py` indexes whatever PDFs it finds, but `app.py`'s product
dropdown is a hard-coded list. If you index a different set of manuals, update
that list to match the filenames.

A GPU is strongly recommended: the 7B generator runs in fp16 with
`device_map="auto"` and will offload to CPU otherwise.

## Limitations

Reported honestly — this was built as a working prototype, not an evaluated system.

- **No evaluation.** Neither retrieval quality (recall@k, MRR) nor answer quality
  was measured. There is no test set.
- **The product filter is on shaky ground.** It embeds a Korean product name and a
  Korean filename with `all-MiniLM-L6-v2`, which is an English-centric model, and
  thresholds the cosine at 0.7. That threshold was not tuned, and the similarity
  it measures between two Korean strings is not reliable. Using `bge-m3` here —
  already loaded, and multilingual — or matching on filename directly would be
  sounder.
- **The web-search fallback reads a field that may not exist.** It calls
  `result.get('snippet', ...)` on Tavily results, but `TavilySearchResults` returns
  each hit under `content` in current LangChain versions. If so, the fallback
  silently yields placeholder text rather than search results. Verify against the
  version you install before relying on it.
- **Retrieval happens before filtering.** Only 5 chunks are fetched and *then*
  narrowed by product, so a query can end up with nothing to work from even when
  the right manual is indexed. Filtering at the vector-store level (Chroma
  metadata filter on `file_name`) would avoid this.

## References

- **bge-m3** — Chen et al. (2024). "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation."
- **KoAlpaca** — [beomi/KoAlpaca](https://github.com/Beomi/KoAlpaca), a Korean instruction-tuned LLaMA.
- **all-MiniLM-L6-v2** — [sentence-transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- **Chroma** — [chroma-core/chroma](https://github.com/chroma-core/chroma).
- **Tavily** — [tavily.com](https://tavily.com), search API for LLM agents.
