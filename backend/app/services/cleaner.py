import re

def clean_text(text: str) -> str:
    """
    Normalize extracted text
    """
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,()\-+/ ]", "", text)
    return text.strip().lower()
