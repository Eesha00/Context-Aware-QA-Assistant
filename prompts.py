"""
prompts.py — dynamic system prompts (change based on session metadata)
and the structured-output schema the model must answer in.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# 1. STRUCTURED OUTPUT SCHEMA
# Instead of parsing free text, we force the model to return this shape.
# LangChain's `.with_structured_output(Schema)` handles the tool-calling
# / JSON-mode plumbing for whichever backend you're using.
# ---------------------------------------------------------------------
class QAAnswer(BaseModel):
    answer: str = Field(description="The direct answer to the user's question")
    confidence: str = Field(description="One of: high, medium, low")
    follow_up_question: str = Field(
        description="One relevant follow-up question the user might ask next"
    )


# ---------------------------------------------------------------------
# 2. DYNAMIC SYSTEM PROMPT
# The "context" here is session metadata (the user's role), not just
# conversation history. Same question, different framing of the answer.
# ---------------------------------------------------------------------
ROLE_INSTRUCTIONS = {
    "developer": (
        "The user is a software developer. Be technically precise. "
        "Use code snippets, correct terminology, and mention edge cases "
        "or implementation details where relevant. Don't oversimplify."
    ),
    "manager": (
        "The user is a non-technical manager. Avoid jargon. Focus on "
        "business impact, timelines, risk, and trade-offs. Use analogies "
        "instead of technical detail. Keep answers under 150 words."
    ),
    "student": (
        "The user is a student learning the topic. Explain concepts "
        "step by step, define any technical term the first time you use "
        "it, and prefer clarity over brevity."
    ),
}


def build_system_prompt(role: str, extra_context: str = "") -> str:
    """
    Builds the system prompt text based on session metadata (role).
    extra_context is where you'd inject retrieved documents, a
    conversation summary, etc. — see context_manager.py.
    """
    role_text = ROLE_INSTRUCTIONS.get(role, ROLE_INSTRUCTIONS["developer"])
    base = (
        "You are a helpful, context-aware Q&A assistant.\n"
        f"AUDIENCE: {role_text}\n"
    )
    if extra_context:
        base += (
            "\nRELEVANT CONTEXT (summary of earlier conversation, "
            "use only if relevant, do not repeat it verbatim):\n"
            f"{extra_context}\n"
        )
    return base


def build_chat_prompt(role: str, extra_context: str = "") -> ChatPromptTemplate:
    """
    Assembles the final prompt template used by the chain in app.py.
    Order matters for context engineering:
      1. System instructions + role framing + summarised history
      2. Placeholder for recent verbatim messages (the token-budgeted window)
      3. The new user question

    IMPORTANT: system_text contains dynamically generated content (the
    rolling summary), which may itself contain literal { } characters
    (e.g. from code snippets or JSON the model summarised). LangChain's
    default f-string template format treats { } as placeholder syntax,
    so any stray brace would crash template parsing. We escape braces
    here to treat the dynamic content as literal text, not template
    syntax.
    """
    system_text = build_system_prompt(role, extra_context)
    escaped_system_text = system_text.replace("{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages(
        [
            ("system", escaped_system_text),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
