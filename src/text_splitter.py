import re

def split_text(text):
    chunks = []

    # Clean text
    text = text.replace("\n", " ").replace("  ", " ")

    # ✅ Extract proper Q&A pairs
    pattern = r"(Q\.?\s*\d*.*?\?)(.*?)(?=Q\.?\s*\d*|\Z)"
    matches = re.findall(pattern, text, re.IGNORECASE)

    for q, a in matches:
        chunk = f"Question: {q.strip()} Answer: {a.strip()}"
        if len(chunk) > 80:
            chunks.append(chunk)

    # ✅ Fallback (for normal PDFs)
    if not chunks:
        chunk_size = 400
        overlap = 100

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size].strip()
            if len(chunk) > 80:
                chunks.append(chunk)

    return chunks