import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. Load .env
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Create a .env file in the same folder as app.py:\n"
        "GROQ_API_KEY=your_groq_api_key"
    )


# ============================================================
# 2. Groq LLM
# ============================================================

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


# ============================================================
# 3. State
# ============================================================

class State(TypedDict):
    input: str
    draft: str
    critique: str
    revision_count: int
    status: str


# ============================================================
# 4. Generate Node
# ============================================================

def generate(state: State):

    print("\n[GENERATE] Creating answer...")

    if state["revision_count"] == 0:

        prompt = f"""
You are an expert assistant.

Answer the following question clearly and accurately.

Question:
{state["input"]}

Give a well-structured answer.
"""

    else:

        prompt = f"""
You are improving an existing answer.

Question:
{state["input"]}

Previous answer:
{state["draft"]}

Reviewer feedback:
{state["critique"]}

Improve the previous answer according to the feedback.

Return only the improved answer.
"""

    response = llm.invoke(prompt)

    return {
        "draft": response.content,
        "revision_count": state["revision_count"] + 1
    }


# ============================================================
# 5. Reflection Node
# ============================================================

def reflect(state: State):

    print("[REFLECT] Reviewing answer...")

    prompt = f"""
You are a strict reviewer.

Review this answer.

Question:
{state["input"]}

Answer:
{state["draft"]}

Check:

1. Is it correct?
2. Is anything important missing?
3. Are there logical errors?
4. Is it clear?
5. Is it relevant to the question?

If the answer is good enough, start your response with:

PASS

Otherwise start your response with:

REVISE

Then briefly explain your reasoning.
"""

    response = llm.invoke(prompt)

    critique = response.content.strip()

    if critique.upper().startswith("PASS"):
        status = "PASS"
    else:
        status = "REVISE"

    return {
        "critique": critique,
        "status": status
    }


# ============================================================
# 6. Decide whether to continue
# ============================================================

def should_continue(state: State):

    if state["status"] == "PASS":
        return "end"

    if state["revision_count"] >= 3:
        return "end"

    return "generate"


# ============================================================
# 7. Build LangGraph
# ============================================================

graph = StateGraph(State)

graph.add_node("generate", generate)
graph.add_node("reflect", reflect)

graph.add_edge(START, "generate")

graph.add_edge("generate", "reflect")

graph.add_conditional_edges(
    "reflect",
    should_continue,
    {
        "generate": "generate",
        "end": END,
    },
)


# ============================================================
# 8. Compile
# ============================================================

app = graph.compile()


# ============================================================
# 9. Run Application
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("       SELF-IMPROVING AI AGENT")
    print("       LangGraph + Groq")
    print("=" * 60)

    while True:

        user_input = input(
            "\nEnter your question (or type 'exit'): "
        )

        if user_input.lower() == "exit":
            print("\nGoodbye!")
            break

        initial_state = {
            "input": user_input,
            "draft": "",
            "critique": "",
            "revision_count": 0,
            "status": "",
        }

        try:

            result = app.invoke(initial_state)

            print("\n" + "=" * 60)
            print("FINAL ANSWER")
            print("=" * 60)

            print(result["draft"])

            print("\n" + "-" * 60)
            print("Revisions:", result["revision_count"])
            print("Status:", result["status"])
            print("-" * 60)

        except Exception as e:

            print("\nERROR:")
            print(e)