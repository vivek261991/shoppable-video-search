import streamlit as st
from scripts.search import query_milvus, generate_answer_with_llm

st.set_page_config(page_title="Retail Video Q&A", layout="wide")

st.title("🛋️ Retail Video Assistant")
st.write("Ask questions about customer scenes in retail videos.")

video_name = "video_name"
query = st.text_input("Ask a question", value="Why did they skip the delivery for the couch?")

if st.button("Ask"):
    with st.spinner("Querying Milvus and reasoning with OpenAI..."):
        docs = query_milvus(video_name, query)

        st.subheader("🔎 Top Relevant Chunks")
        for doc in docs:
            st.markdown(f"**[{doc.metadata.get('start', 0.0):.2f}s - {doc.metadata.get('end', 0.0):.2f}s]** {doc.page_content}")

        context = "\n".join([
            f"[{doc.metadata.get('start', 0.0):.2f}s - {doc.metadata.get('end', 0.0):.2f}s] {doc.page_content}"
            for doc in docs
        ])

        answer = generate_answer_with_llm(context, query)
        st.subheader("LLM Answer")
        st.success(answer)