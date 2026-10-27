"""Demo: run the agent on sample tasks and print the trace.

    python run_demo.py
"""
from agentbridge.agent import Agent

SAMPLES = [
    "What is 125 * 8?",
    "Check claim #CLM-1001",
    "Hello, what can you do?",
]


def main() -> None:
    agent = Agent()
    print(f"Provider: {agent.provider.name}\n")
    for i, sample in enumerate(SAMPLES, 1):
        result = agent.run(sample)
        print(f"--- Task {i}: {sample}")
        print(f"    Answer : {result.answer}")
        print(f"    Steps  : {result.iterations}, tools used: {result.tool_names}")
        for s in result.trace:
            print(f"      [{s.kind:6}] {s.name or '-':18} {s.latency_ms:7.2f} ms  {s.detail[:70]}")
        print()


if __name__ == "__main__":
    main()
