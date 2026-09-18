"""
agent.py
--------
A minimal ReAct-style agent: the LLM "thinks" about the task, decides on an
action (a tool call), we execute that tool in Python, feed the result back
as an "observation," and repeat until the LLM produces a Final Answer.

This is the core pattern behind almost all "agentic AI" frameworks
(LangChain agents, AutoGPT, etc.) -- they're built on the same loop, just
with more tooling around it. Understanding this file end-to-end is enough
to explain how an agent works in an interview.

Usage:
    python agent.py "What is the population of Japan divided by 1000?"

Requires an ANTHROPIC_API_KEY environment variable.
"""

import os
import re
import sys

import anthropic

from tools import TOOLS

MODEL = "claude-sonnet-4-5"
MAX_STEPS = 6  # safety limit so the agent can't loop forever


def build_system_prompt() -> str:
    tool_descriptions = "\n".join(
        f"- {name}: {meta['description']}" for name, meta in TOOLS.items()
    )
    return f"""You are an AI agent that solves tasks step by step using tools.

Available tools:
{tool_descriptions}

You must respond using EXACTLY one of these two formats, and nothing else:

Format 1 - to use a tool:
Thought: <your reasoning about what to do next>
Action: <tool_name>[<tool input>]

Format 2 - when you have the final answer:
Thought: <your reasoning>
Final Answer: <the final answer to the user's question>

Rules:
- Only call one tool per turn.
- Wait for the Observation before deciding the next step.
- Do not make up an Observation yourself -- it will be provided to you.
- Keep Thoughts concise (1-2 sentences).
"""


def parse_action(text: str):
    """Extract tool name and input from a line like: Action: calculator[2+2]"""
    match = re.search(r"Action:\s*(\w+)\[(.*)\]", text, re.DOTALL)
    if not match:
        return None, None
    tool_name = match.group(1).strip()
    tool_input = match.group(2).strip()
    return tool_name, tool_input


def parse_final_answer(text: str):
    match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def run_agent(user_task: str, verbose: bool = True) -> str:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    system_prompt = build_system_prompt()
    conversation = [{"role": "user", "content": f"Task: {user_task}"}]

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system=system_prompt,
            messages=conversation,
        )
        model_text = response.content[0].text
        conversation.append({"role": "assistant", "content": model_text})

        if verbose:
            print(f"\n--- Step {step} ---")
            print(model_text)

        final_answer = parse_final_answer(model_text)
        if final_answer:
            return final_answer

        tool_name, tool_input = parse_action(model_text)
        if tool_name is None:
            # Model didn't follow the format -- nudge it and try again
            conversation.append({
                "role": "user",
                "content": "Please respond using the exact Thought/Action or Thought/Final Answer format.",
            })
            continue

        if tool_name not in TOOLS:
            observation = f"Error: unknown tool '{tool_name}'. Available tools: {list(TOOLS.keys())}"
        else:
            observation = TOOLS[tool_name]["fn"](tool_input)

        if verbose:
            print(f"Observation: {observation}")

        conversation.append({"role": "user", "content": f"Observation: {observation}"})

    return "Agent stopped: reached max steps without a final answer."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your question here"')
        sys.exit(1)

    task = " ".join(sys.argv[1:])
    answer = run_agent(task)
    print("\n=== FINAL ANSWER ===")
    print(answer)
