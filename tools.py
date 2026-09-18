"""
tools.py
--------
Defines the tools the agent can choose to call. Each tool is a plain Python
function that takes a string input and returns a string observation.

Keeping tools as simple string-in/string-out functions makes them easy to
plug into a ReAct-style loop, where the LLM's output is parsed as text.
"""

import ast
import operator as op
import requests


# ---------------------------------------------------------------------------
# Tool 1: Calculator
# ---------------------------------------------------------------------------
# We do NOT use eval() directly since that would let the agent execute
# arbitrary Python code (a real security risk if this were ever exposed).
# Instead we walk a parsed AST and only allow safe arithmetic operations.

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.Mod: op.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):  # numbers
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type} not allowed")
        return _ALLOWED_OPERATORS[op_type](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {node}")


def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression, e.g. '12 * (3 + 4)'."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# ---------------------------------------------------------------------------
# Tool 2: Wikipedia search
# ---------------------------------------------------------------------------
# Uses Wikipedia's public REST API directly (no API key required), so the
# project runs out of the box. Returns a short summary the agent can reason
# over.

def wikipedia_search(query: str) -> str:
    """Look up a topic on Wikipedia and return a short summary."""
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        search_resp = requests.get(search_url, params=search_params, timeout=10).json()
        results = search_resp.get("query", {}).get("search", [])
        if not results:
            return f"No Wikipedia results found for '{query}'."

        title = results[0]["title"]

        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        summary_resp = requests.get(summary_url, timeout=10).json()
        extract = summary_resp.get("extract", "No summary available.")
        return f"[{title}] {extract}"
    except Exception as e:
        return f"Error searching Wikipedia: {e}"


# ---------------------------------------------------------------------------
# Tool 3: Scratchpad (in-memory notes)
# ---------------------------------------------------------------------------
# A trivial "memory" tool. Lets the agent jot down intermediate facts across
# steps of a multi-step task -- a simple building block toward more advanced
# agent memory systems.

_notes = []


def add_note(text: str) -> str:
    _notes.append(text)
    return f"Noted: {text}"


def get_notes(_: str = "") -> str:
    if not _notes:
        return "No notes yet."
    return "\n".join(f"{i+1}. {n}" for i, n in enumerate(_notes))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
# Maps the tool name (as the LLM will refer to it in its output) to the
# actual Python function, plus a human-readable description used to build
# the system prompt.

TOOLS = {
    "calculator": {
        "fn": calculator,
        "description": "Evaluate a basic arithmetic expression. Input: a math expression like '12 * (3 + 4)'.",
    },
    "wikipedia_search": {
        "fn": wikipedia_search,
        "description": "Search Wikipedia and return a short summary. Input: a topic or question, e.g. 'Eiffel Tower'.",
    },
    "add_note": {
        "fn": add_note,
        "description": "Save a short fact to the scratchpad for later reference. Input: the text to save.",
    },
    "get_notes": {
        "fn": get_notes,
        "description": "Retrieve all notes saved so far. Input: ignored, pass an empty string.",
    },
}
