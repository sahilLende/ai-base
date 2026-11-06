"""Run: streamlit run app.py"""
import pandas as pd
import streamlit as st
from src import llm
from src.llm import chat, ask_json
from src.ingest import load_file, load_table
from src.rag import RAG
from src.prompts import TASKS, SYSTEM, CASE_SYSTEM, CASE_JSON

st.set_page_config(page_title="AI Hackathon Demo", layout="wide")
st.title("🚀 [Project Name] - [one-line value proposition]")

with st.sidebar:
    st.header("Documents (for tabs 2-3)")
    files = st.file_uploader("Upload txt/pdf/csv/xlsx", accept_multiple_files=True)
    pasted = st.text_area("...or paste text", height=150)
    st.caption(f"Chat model: {llm.MODEL} | Embeddings: {llm.EMBED_MODEL or 'off (TF-IDF)'}")
texts = [load_file(f) for f in files] + ([pasted] if pasted.strip() else [])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🎫 Similar cases (Excel)", "💬 Ask documents", "⚙️ Analyze", "📈 Analytics", "📊 Business impact"])

# ---------- 1. Similar cases: Excel of past cases -> resolve a new one ----------
with tab1:
    tf = st.file_uploader("Excel/CSV of past cases (problem + resolution)",
                          type=["xlsx", "xls", "csv"], key="table")
    if tf:
        df = load_table(tf)
        st.dataframe(df.head(5), use_container_width=True)
        cols = list(df.columns)
        c1, c2, c3 = st.columns(3)
        prob = c1.multiselect("Problem column(s) - what is matched", cols, default=cols[:1])
        sol = c2.selectbox("Resolution column", cols, index=min(1, len(cols) - 1))
        filt = c3.selectbox("Optional filter column", ["(none)"] + cols)
        if st.button("Build index") and prob:
            with st.spinner(f"Indexing {len(df)} rows..."):
                st.session_state.trag = RAG.from_table(df, prob, sol)
                st.session_state.tdf = df
                st.session_state.tcols = (prob, sol)

    if "trag" in st.session_state:
        rag, df = st.session_state.trag, st.session_state.tdf
        st.success(f"Index ready ({len(rag.items)} cases) - search mode: {rag.mode}")
        new = st.text_area("Describe the new incident / case", height=100)
        c1, c2, c3 = st.columns(3)
        k = c1.slider("Similar cases", 1, 8, 4)
        min_score = c2.slider("Min similarity", 0.0, 1.0, 0.3)
        fval = None
        if filt != "(none)":
            fval = c3.selectbox("Filter value", ["(all)"] + sorted(df[filt].astype(str).unique()))
        as_json = st.checkbox("Also return structured JSON", value=True)

        if st.button("Find resolution") and new.strip():
            where = (lambda m: m.get(filt) == fval) if fval and fval != "(all)" else None
            with st.spinner("Searching and reasoning..."):
                ans, hits = rag.answer(new, k=k, min_score=min_score, where=where, system=CASE_SYSTEM)
            st.subheader("Suggested resolution")
            st.write(ans)
            if hits and hits[0]["score"] >= min_score and as_json:
                ctx = "\n\n".join(f"[{h['idx']}] {h['item']['doc']}" for h in hits)
                st.json(ask_json(CASE_JSON.format(q=new, ctx=ctx)))
            st.subheader("Similar past cases")
            for h in hits:
                with st.expander(f"[{h['idx']}] similarity {h['score']:.2f} - {h['item']['text'][:90]}"):
                    st.text(h["item"]["doc"])

# ---------- 2. Ask documents (chunked RAG) ----------
with tab2:
    if not texts:
        st.info("Add documents in the sidebar.")
    else:
        if "rag" not in st.session_state or st.session_state.get("n") != len(texts):
            with st.spinner("Indexing..."):
                st.session_state.rag, st.session_state.n = RAG.from_texts(texts), len(texts)
            st.session_state.msgs = []
        for m in st.session_state.msgs:
            st.chat_message(m["role"]).write(m["content"])
        if q := st.chat_input("Ask a question"):
            st.chat_message("user").write(q)
            with st.spinner("Thinking..."):
                a, hits = st.session_state.rag.answer(q, st.session_state.msgs[-6:])
            st.chat_message("assistant").write(a)
            with st.expander("Sources"):
                for h in hits:
                    st.caption(f"[{h['idx']}] ({h['score']:.2f}) {h['item']['doc'][:300]}...")
            st.session_state.msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]

# ---------- 3. Analyze (prompt tasks) ----------
with tab3:
    task = st.selectbox("Task", list(TASKS))
    prompt_t = st.text_area("Prompt template (editable)", TASKS[task], height=120)
    if st.button("Run") and texts:
        prompt = prompt_t.replace("{text}", "\n\n".join(texts)[:12000])
        with st.spinner("Running..."):
            if "JSON" in task or "risk" in task:
                st.json(ask_json(prompt))
            else:
                st.write(chat(prompt, system=SYSTEM))

# ---------- 4. Analytics (plain pandas, no AI) ----------
with tab4:
    if "tdf" not in st.session_state:
        st.info("Build the index in tab 1 first.")
    else:
        d = st.session_state.tdf
        col = st.selectbox("Count by column", list(d.columns), key="an")
        st.bar_chart(d[col].astype(str).value_counts().head(15))
        st.caption("Tip: top recurring issues = automation candidates.")

# ---------- 5. Business impact ----------
with tab5:
    st.subheader("Show the judges the numbers")
    c1, c2, c3 = st.columns(3)
    hours = c1.number_input("Hours saved per week", 0.0, 1000.0, 10.0)
    rate = c2.number_input("Cost per hour", 0.0, 1000.0, 30.0)
    people = c3.number_input("People using it", 1, 10000, 5)
    st.metric("Estimated yearly saving", f"{hours * rate * people * 52:,.0f}")
