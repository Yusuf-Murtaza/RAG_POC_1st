"""Step 6. Connect to the LLM (The brain of assistant)"""

from langchain_openrouter import ChatOpenRouter
from hr_assistant import config

def get_llm(model_name: str = config.LLM_MODEL_NAME):
    """Return the OpenRouter LLM model. Reads OPENROUTER_API_KEY from environment variables."""
    return ChatOpenRouter(
        model_name=config.LLM_MODEL_NAME, 
        max_tokens=2048,
        temperature=0
        )

