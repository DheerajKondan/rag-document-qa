import streamlit as st
from src.pdf_loader import load_pdf
from src.text_splitter import split_text
from src.embeddings import create_embeddings, model as embedding_model
from src.vector_store import create_faiss_index, search
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="PDF Q&A", layout="wide")

st.title("📄 AI Document Q&A (RAG)")
st.write("Upload a PDF and ask questions")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # Load model (only once)
    @st.cache_resource
    def load_llm():
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        return tokenizer, model

    tokenizer, llm_model = load_llm()

    # Process PDF
    text = load_pdf("temp.pdf")
    chunks = split_text(text)
    embeddings = create_embeddings(chunks)
    index = create_faiss_index(embeddings)

    # User question
    query = st.text_input("Ask a question")

    if query:
        query_embedding = embedding_model.encode([query])[0]
        results = search(index, query_embedding, chunks)

        context = results[0]

        prompt = f"""
        You are a precise AI assistant.

        Check the complete context of the pdf, check the comlpete pdf
        Answer ONLY using the given context.
        If the answer is not clearly present, say: "Not found in document".


        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=120,
            num_beams=4
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("🔍Retrieved Context:")
        st.write(context)