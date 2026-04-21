import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def search(index, query_embedding, chunks, top_k=3):
    distances, indices = index.search(np.array([query_embedding]), top_k)

    seen = set()
    results = [chunks[i] for i in indices[0]if i < len(chunks)]

    for i in indices[0]:
        chunk = chunks[i].strip()

        # remove duplicates + noise
        if chunk not in seen and len(chunk) > 50:
            seen.add(chunk)
            results.append(chunk)

    return results


# testing
if __name__ == "__main__":
    from pdf_loader import load_pdf
    from text_splitter import split_text
    from embeddings import create_embeddings, model

    text = load_pdf("data/sample.pdf")
    chunks = split_text(text)

    embeddings = create_embeddings(chunks)

    index = create_faiss_index(embeddings)

    query = "What skills does this person have?"
    query_embedding = model.encode([query])[0]

    results = search(index, query_embedding, chunks)

    print("\nTop Matching Chunks:\n")
    for r in results:
        print(r)
        print("-" * 50)