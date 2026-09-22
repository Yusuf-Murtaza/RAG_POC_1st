"""Step 3. Turn text into numbers (Vectors) using Jina"""

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config

def get_embeddings_model(model_name: str = config.EMBEDDINGS_MODEL_NAME):
    """Return the Jina embeddings model. Reads JINA_API_KEY from environment variables.""" 
    return JinaEmbeddings(model_name=config.EMBEDDINGS_MODEL_NAME)
