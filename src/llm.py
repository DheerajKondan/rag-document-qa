from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pdf_loader import load_pdf
from text_splitter import split_text
from embeddings import create_embeddings, model as embedding_model
from vector_store import create_faiss_index, search

# Load LLM
print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Load PDF
text = load_pdf("data/sample.pdf")

# Split into chunks
chunks = split_text(text)

# Create embeddings
embeddings = create_embeddings(chunks)

# Create FAISS index
index = create_faiss_index(embeddings)

# Query
query = input("Enter your question: ")

# Convert query to embedding
query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]

# Retrieve relevant chunks
results = search(index, query_embedding, chunks)

# Debug (optional)
print("\nRetrieved Chunks:\n")
for r in results:
    print(r)
    print("-" * 50)

# Clean + limit context
clean_results = [r.strip() for r in results if len(r.strip()) > 50]
context = clean_results[0]   # top 3 chunks

# Prompt (GENERIC)
prompt = f"""
You are an intelligent AI assistant.

Answer the question clearly and completely using ONLY the context below.

Instructions:
- Give a clear and structured answer
- Use bullet points if possible
- Do NOT repeat the question
- Do NOT copy raw text
- Do NOT give partial sentences
- Summarize properly
- If multiple points exist, present them as bullet points
- Do NOT include irrelevant text
- Be concise and accurate

Context:
{context}

Question:
{query}

Answer:
"""

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

# Generate answer
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    num_beams=5,
    repetition_penalty=2.0,
    do_sample=False,
    early_stopping=True
)

# Decode output
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nFinal Answer:\n")
print(answer)