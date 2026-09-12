"""Accumulates every LLM call made during a run and writes evaluation/usage_report.md,
per the submission requirement: model providers/names, model calls, input/output tokens,
total and average tokens per request, estimated total and per-request cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config


@dataclass
class CallRecord:
    call_type: str  # "messages_llm" | "vision_llm" | "explain_llm"
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_hit: bool
    cost: float


@dataclass
class UsageTracker:
    records: list[CallRecord] = field(default_factory=list)

    def record(self, call_type: str, provider: str, model: str, input_tokens: int, output_tokens: int, cache_hit: bool) -> None:
        cost = 0.0 if cache_hit else estimate_cost(provider, model, input_tokens, output_tokens)
        self.records.append(CallRecord(call_type, provider, model, input_tokens, output_tokens, cache_hit, cost))

    def write_report(self, num_requests: int, path=None) -> str:
        path = path or (config.EVALUATION_DIR / "usage_report.md")
        path.parent.mkdir(parents=True, exist_ok=True)

        billed = [r for r in self.records if not r.cache_hit]
        cached = [r for r in self.records if r.cache_hit]

        by_model: dict[tuple[str, str], list[CallRecord]] = {}
        for r in self.records:
            by_model.setdefault((r.provider, r.model), []).append(r)

        total_input = sum(r.input_tokens for r in self.records)
        total_output = sum(r.output_tokens for r in self.records)
        total_tokens = total_input + total_output
        total_cost = sum(r.cost for r in self.records)

        lines = [
            "# Token Usage and Cost Report",
            "",
            f"Final full-dataset run: {num_requests} requests in `dataset/requests.csv`.",
            "",
            "## Summary",
            "",
            f"- Total LLM calls: {len(self.records)} ({len(billed)} billed, {len(cached)} served from cache at $0)",
            f"- Total input tokens: {total_input:,}",
            f"- Total output tokens: {total_output:,}",
            f"- Total tokens: {total_tokens:,}",
            f"- Average tokens per request: {total_tokens / num_requests:.1f}" if num_requests else "- Average tokens per request: n/a",
            f"- Estimated total cost: ${total_cost:.4f}",
            f"- Estimated cost per request: ${total_cost / num_requests:.6f}" if num_requests else "- Estimated cost per request: n/a",
            "",
            "## By call type",
            "",
            "| call_type | calls | billed | cached | input_tokens | output_tokens | cost |",
            "|---|---|---|---|---|---|---|",
        ]
        for call_type in sorted(set(r.call_type for r in self.records)):
            rs = [r for r in self.records if r.call_type == call_type]
            lines.append(
                f"| {call_type} | {len(rs)} | {sum(1 for r in rs if not r.cache_hit)} | "
                f"{sum(1 for r in rs if r.cache_hit)} | {sum(r.input_tokens for r in rs):,} | "
                f"{sum(r.output_tokens for r in rs):,} | ${sum(r.cost for r in rs):.4f} |"
            )

        lines += [
            "",
            "## By model",
            "",
            "| provider | model | calls | input_tokens | output_tokens | cost |",
            "|---|---|---|---|---|---|",
        ]
        for (provider, model), rs in sorted(by_model.items()):
            lines.append(
                f"| {provider} | {model} | {len(rs)} | {sum(r.input_tokens for r in rs):,} | "
                f"{sum(r.output_tokens for r in rs):,} | ${sum(r.cost for r in rs):.4f} |"
            )

        lines.append("")
        report = "\n".join(lines)
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        return report


# Per-million-token rates (USD). Placeholder figures -- verify against current published
# pricing for whichever provider actually served the final submission run; update here if
# they've changed.
_RATES_PER_MILLION_TOKENS = {
    ("anthropic", "claude-sonnet-5"): (3.00, 15.00),  # (input, output)
    ("gemini", "gemini-3.6-flash"): (0.30, 2.50),  # (input, output) -- placeholder, verify actual rate
}


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    rate_in, rate_out = _RATES_PER_MILLION_TOKENS.get((provider, model), (0.0, 0.0))
    return (input_tokens / 1_000_000) * rate_in + (output_tokens / 1_000_000) * rate_out
