from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


# testing purpose
if __name__ == "__main__":
    file_path = "data/sample.pdf"   # make sure this file exists
    extracted_text = load_pdf(file_path)

    print("Extracted Text:\n")
    print(extracted_text[:1000])  # print first 1000 characters