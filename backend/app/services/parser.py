from pypdf import PdfReader
from docx import Document

def extract_text(file):
    filename = file.filename.lower()

    if not filename:
        raise ValueError("No filename provided")

    if filename.endswith(".pdf"):
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if not text.strip():
            raise ValueError("PDF has no extractable text")
        return text

    elif filename.endswith(".docx"):
        doc = Document(file.file)
        text = "\n".join(p.text for p in doc.paragraphs)
        if not text.strip():
            raise ValueError("DOCX file is empty")
        return text

    elif filename.endswith(".txt"):
        text = file.file.read().decode("utf-8")
        if not text.strip():
            raise ValueError("TXT file is empty")
        return text

    else:
        raise ValueError("Only PDF, DOCX, TXT are supported")
