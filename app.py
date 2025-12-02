# app.py
import streamlit as st
from rag_pipeline import rag_answer

st.set_page_config(page_title="🎬 Movie Review RAG + KG", layout="wide")
st.title("🎬 Movie Review RAG Assistant with Knowledge Graph (DBpedia)")

# Input
question = st.text_input("Ask something about any movie review:")

if st.button("Ask") and question.strip():
    # Get RAG answer, top chunks, and KG info
    answer, sources, kg_info = rag_answer(question)

    # 1️⃣ Show RAG answer
    st.subheader("💡 RAG Answer:")
    st.write(answer)

    # 2️⃣ Show top retrieved review chunks
    st.subheader("📌 Top Retrieved Review Evidences:")
    for src in sources:
        st.write("- " + src)

    # 3️⃣ Show Knowledge Graph info
    st.subheader("🎬 Knowledge Graph Lookup (DBpedia)")

    if kg_info is None:
        st.write("⚠ No structured knowledge found for this movie.")
    elif "error" in kg_info:
        st.error("KG ERROR: " + kg_info["error"])
    else:
        st.write(f"**Title:** {kg_info['movie']}")
        st.write(f"**Director:** {kg_info['director']}")
        st.write(f"**Genre:** {kg_info['genre']}")
        st.write(f"**Release Year:** {kg_info['year']}")
