"""
config.py — ONE PLACE to control which model backend the whole app uses.

This is the "swap the backing model with one config change" requirement.
Change MODEL_BACKEND (or set the env var) and nothing else in the app
needs to change, because every backend returns a LangChain chat model
object that supports the same .invoke() / .stream() interface.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if present

# ---- THE ONE SWITCH ----
# Options: "openai" | "anthropic" | "gemini" | "ollama"
MODEL_BACKEND = os.getenv("MODEL_BACKEND", "gemini")

# Model names per backend — tweak freely
MODEL_NAMES = {
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "gemini": os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
    "ollama": os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
}

# Token budget for conversation history (context engineering knob)
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "1200"))

# How many of the most recent messages to always keep verbatim
# (never summarised away — keeps the last exchange sharp)
KEEP_LAST_N_MESSAGES = int(os.getenv("KEEP_LAST_N_MESSAGES", "4"))


def get_chat_model(streaming: bool = True):
    """
    Returns a LangChain chat model instance for whichever backend is
    configured above. This is the only function the rest of the app
    calls — it never talks to OpenAI/Anthropic/Ollama SDKs directly.
    """
    if MODEL_BACKEND == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=MODEL_NAMES["openai"],
            streaming=streaming,
            temperature=0.3,
        )

    if MODEL_BACKEND == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=MODEL_NAMES["anthropic"],
            streaming=streaming,
            temperature=0.3,
        )

    if MODEL_BACKEND == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=MODEL_NAMES["gemini"],
            temperature=0.3,
            # streaming is on by default for .stream() calls
        )

    if MODEL_BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=MODEL_NAMES["ollama"],
            temperature=0.3,
            # Ollama streams by default via .stream(), no flag needed
        )

    raise ValueError(f"Unknown MODEL_BACKEND: {MODEL_BACKEND}")
