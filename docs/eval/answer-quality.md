# Answer-Quality Evaluation (Multimodal RAG)

This report template aligns with the evaluation criteria in "Building a Multimodal RAG System v. 2" and uses the gold set generated from `data/processed/artifacts/chunks.jsonl`.

## 1) Evaluation set summary
- File: `eval/golden_redesign.json`
- Total queries: 30
- Docs covered: 2 (TM-9-254, GOVPUB-D101-PURL-LPS45154)
- Query mix: 20 text, 6 table, 4 image
- Intent mix: reference (20), specs (6), identification (4)
- Difficulty mix: easy (12), medium (9), hard (9)

## 2) Retrieval metrics (required)
Run:
```bash
python eval/run_retrieval_eval.py \
  --queries eval/golden_redesign.json \
  --k 1 3 5 10 \
  --out eval/retrieval_metrics.json
```

The report separates `asset_retrieval` (actual CLIP text-to-asset search) from
`page_linked_assets` (a diagnostic inferred from text-retrieved pages). Do not
use the latter as a multimodal retrieval score.

Record results here:
- Recall@k (k=1/3/5/10): TODO
- Precision@k (k=1/3/5/10): TODO
- MRR (k=1/3/5/10): TODO
- Image/table hit-rate (k=1/3/5/10): TODO

## 3) Answer-quality metrics (required)
Use a manual rubric across a 10-15 query sample (covering text, table, and image queries). Score each from 1–5.

Rubric dimensions:
1. Faithfulness / groundedness
   - 1: Hallucinated or contradicted by sources
   - 3: Mostly grounded, minor unsupported claims
   - 5: Fully grounded in cited sources
2. Completeness
   - 1: Misses key details present in sources
   - 3: Covers some key points, misses others
   - 5: Covers all key points in sources
3. Citation accuracy
   - 1: Citations do not support claims
   - 3: Partial support or missing ties
   - 5: Each claim is backed by a cited chunk
4. Multimodal grounding (for image/table queries)
   - 1: Ignores visual asset or contradicts it
   - 3: Mentions asset but shallow
   - 5: Correctly interprets the asset

Record results here:
- Mean faithfulness: TODO
- Mean completeness: TODO
- Mean citation accuracy: TODO
- Mean multimodal grounding: TODO

## 4) Human preference scoring (optional but recommended)
Compare two prompt variants or reranker settings:
- Variant A: baseline prompt + current reranker
- Variant B: modified prompt or reranker

Method:
- Blindly score 10 queries for preference (A vs B)
- Record win rate and short justification per query

## 5) Error analysis (required)
Log at least 5 failures with categories:
- OCR noise / bad extraction
- Chunking boundary or overlap issues
- Embedding mismatch (semantic drift)
- Missing or wrong asset linking
- Prompt too permissive (hallucinations)

Template:
- Query ID:
- Failure type:
- Symptom:
- Likely cause:
- Fix:

## 6) Concrete improvements (required)
1. Improve chunking for tables: split large tables into logical sections and preserve headers.
2. Re-caption low-quality images and store captions as separate chunks for better retrieval.
3. Add a lightweight reranker for image/table queries (caption-only rerank) to boost hit rate.
4. Add query rewriting for ambiguous requests (short or low-entropy queries).
