from sentence_transformers import SentenceTransformer

# load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
    embeddings = model.encode(chunks, normalize_embeddings=True)
    return embeddings


# testing
if __name__ == "__main__":
    from pdf_loader import load_pdf
    from text_splitter import split_text

    text = load_pdf("data/sample.pdf")
    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    print(f"Total Chunks: {len(chunks)}")
    print(f"Embedding Shape: {embeddings.shape}")