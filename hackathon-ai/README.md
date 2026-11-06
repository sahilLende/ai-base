# Hackathon AI Skeleton
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. `cp .env.example .env` -> fill API_KEY / MODEL (+ BASE_URL, EMBED_MODEL if given)
4. `python -m src.llm` -> must print a reply (API check; also tests embeddings)
5. `streamlit run app.py`

## Tabs
- Similar cases: upload Excel (e.g. incidents + resolutions) -> pick problem/resolution columns -> Build index -> describe new incident.
  One row = one item; only the problem columns are embedded; the resolution comes back with the match.
- Ask documents: chunked RAG over txt/pdf/csv/xlsx.  Analyze: prompt tasks.  Analytics: pandas charts.  Business impact: ROI.

## Embeddings
Set EMBED_MODEL in .env (e.g. text-embedding-3-small). Empty or failing -> automatic TF-IDF fallback.
Vectors are cached in data/.cache so re-runs are instant (delete it if the data changes a lot).

Edit `src/prompts.py` (the brain) and the title in `app.py`. Sample data: `data/incidents_sample.xlsx`.

## Timeline (1pm-5pm)
- 1:00-1:20 Understand case, write problem/user/value in 3 lines
- 1:20-2:00 Prompts working on sample data
- 2:00-3:30 Wire to UI, one happy path end-to-end
- 3:30-4:15 Polish, edge cases, business-impact numbers
- 4:15-5:00 Pitch + rehearsal (freeze code at 4:15)
