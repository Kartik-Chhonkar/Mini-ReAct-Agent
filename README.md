
# Mini ReAct Agent

A small but *real* agentic AI project: an LLM that reasons about a task,
decides which tool to call, observes the result, and repeats — until it can
give a final answer. This is the **ReAct (Reason + Act)** pattern, the same
core idea behind LangChain agents, AutoGPT, and most "agentic AI" tooling.

## How it works

1. The user gives a task (e.g. *"What is the population of Japan divided by 1000?"*).
2. The LLM is prompted with a strict format: it must output either
   - `Thought: ... / Action: tool_name[input]` — to call a tool, or
   - `Thought: ... / Final Answer: ...` — when it's done.
3. Python parses the LLM's output. If it chose an action, the matching
   Python function is called with the given input.
4. The tool's result ("Observation") is appended to the conversation, and
   the loop repeats — so the LLM can chain multiple tool calls together to
   solve multi-step problems.
5. This continues until the LLM outputs a `Final Answer`, or a safety limit
   (`MAX_STEPS`) is hit.

This is exactly the "multi-step AI workflow" pattern referenced in things
like Capgemini's Agentic AI syllabus — the LLM isn't just answering from its
own knowledge, it's actively deciding *when* to defer to an external tool.

## Tools included

| Tool | What it does |
|---|---|
| `calculator` | Safely evaluates arithmetic expressions (no `eval()` — uses a restricted AST walker, so it can't execute arbitrary code) |
| `wikipedia_search` | Looks up a topic on Wikipedia's free public API and returns a short summary (a tiny example of retrieval — the same idea RAG is built on) |
| `add_note` / `get_notes` | A simple in-memory scratchpad, so the agent can save facts partway through a multi-step task |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python agent.py "What is the population of Japan divided by 1000?"
```

You'll see each step printed: the model's Thought, its chosen Action, and
the Observation returned by the tool — so you can watch the reasoning loop
happen in real time.

## Example run

```
--- Step 1 ---
Thought: I need to find the population of Japan first.
Action: wikipedia_search[population of Japan]
Observation: [Japan] Japan is an island country in East Asia...
             population approximately 124 million...

--- Step 2 ---
Thought: Now I need to divide that population by 1000.
Action: calculator[124000000 / 1000]
Observation: 124000.0

--- Step 3 ---
Thought: I now have the final answer.
Final Answer: Japan's population divided by 1000 is approximately 124,000.
```

