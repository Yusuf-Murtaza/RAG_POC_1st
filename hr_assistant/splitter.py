"""Step 2: Split the loaded documents into smaller chunks for better processing and retrieval."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from hr_assistant import config

def split_into_chunks(documents):
    """Split the documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    return text_splitter.split_documents(documents)
