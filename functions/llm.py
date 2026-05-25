from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
MODEL_NAME = os.getenv("AGENT_MODEL_NAME")
DEEPINFRA_BASE_URL = os.getenv("DEEPINFRA_BASE_URL")


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url=DEEPINFRA_BASE_URL,
        model=MODEL_NAME,
        temperature=temperature,
        max_completion_tokens=8192,
    )


def get_compact_llm() -> ChatOpenAI:
    """LLM ringan untuk meringkas messages saat context terlalu besar."""
    return ChatOpenAI(
        api_key=DEEPINFRA_API_KEY,
        base_url=DEEPINFRA_BASE_URL,
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        temperature=0.1,
        max_completion_tokens=4096,
    )
