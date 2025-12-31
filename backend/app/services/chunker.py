from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text):
    if not text or len(text.strip()) < 50:
        raise ValueError("Text too short to chunk")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return splitter.split_text(text)
