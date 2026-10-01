"""
context_manager.py — the "context engineering" core of the project.

Job: decide what conversation history actually goes into the prompt.
- Count tokens for the current history.
- If under budget: send it all verbatim.
- If over budget: summarise the OLDER turns into one system-style note
  and keep only the last KEEP_LAST_N_MESSAGES verbatim.

This is exactly what production chat apps do — you never send an
unbounded transcript to the model.
"""

import tiktoken
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

import config


# ---------------------------------------------------------------------
# TOKEN COUNTING
# tiktoken is OpenAI's tokenizer. It's not a perfect match for
# Anthropic/Ollama token counts, but it's a reliable, fast, consistent
# proxy for budgeting purposes across backends — exact counts aren't
# the point, staying under a sane budget is.
# ---------------------------------------------------------------------
_encoding = tiktoken.get_encoding("cl100k_base")


def extract_text(content) -> str:
    """
    Gemini (and some other backends) can return .content as either a
    plain string or a list of content blocks (dicts/strings) — this
    normalises either shape into plain text for display/concatenation.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text or ""))


def count_message_tokens(messages: list[BaseMessage]) -> int:
    return sum(count_tokens(getattr(m, "content", "")) for m in messages)


# ---------------------------------------------------------------------
# SUMMARISATION
# When history exceeds the budget, we ask the model itself to compress
# the older turns into a short note. That note gets injected into the
# system prompt (see prompts.build_system_prompt) instead of the raw
# messages — this is the "trim / summarise / drop" decision in action.
# ---------------------------------------------------------------------
SUMMARY_PROMPT = (
    "Summarise the following conversation history in under 100 words. "
    "Preserve names, decisions, numbers, and anything the user asked "
    "you to remember. Do not add commentary, just the summary:\n\n"
)


def summarise_messages(messages: list[BaseMessage], model) -> str:
    transcript = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
    )
    result = model.invoke(SUMMARY_PROMPT + transcript)
    raw_content = result.content if hasattr(result, "content") else str(result)
    return extract_text(raw_content)


def manage_context(
    messages: list[BaseMessage],
    model,
    existing_summary: str = "",
) -> tuple[list[BaseMessage], str, int]:
    """
    Core budgeting decision.

    Returns: (messages_to_send_verbatim, updated_summary_text, token_count)
    """
    total_tokens = count_message_tokens(messages)

    if total_tokens <= config.MAX_HISTORY_TOKENS:
        # Under budget — send everything, no summarisation needed.
        return messages, existing_summary, total_tokens

    # Over budget: keep the last N messages verbatim, summarise the rest.
    keep_n = config.KEEP_LAST_N_MESSAGES
    older, recent = messages[:-keep_n], messages[-keep_n:]

    if not older:
        # Nothing old enough to summarise yet (history is still short) —
        # just send everything verbatim rather than asking the model
        # to summarise an empty transcript.
        return messages, existing_summary, total_tokens

    new_summary_chunk = summarise_messages(older, model)
    combined_summary = (
        f"{existing_summary}\n{new_summary_chunk}".strip()
        if existing_summary
        else new_summary_chunk
    )

    recent_tokens = count_message_tokens(recent)
    return recent, combined_summary, recent_tokens
