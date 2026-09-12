# Token Usage and Cost Report

Final full-dataset run: 250 requests in `dataset/requests.csv`.

## Summary

- Total LLM calls: 506 (250 billed, 256 served from cache at $0)
- Total input tokens: 301,536
- Total output tokens: 21,675
- Total tokens: 323,211
- Average tokens per request: 1292.8
- Estimated total cost: $0.0149
- Estimated cost per request: $0.000059

## By call type

| call_type | calls | billed | cached | input_tokens | output_tokens | cost |
|---|---|---|---|---|---|---|
| explain_llm | 250 | 250 | 0 | 111,554 | 9,255 | $0.0149 |
| messages_llm | 234 | 0 | 234 | 160,876 | 11,808 | $0.0000 |
| vision_llm | 22 | 0 | 22 | 29,106 | 612 | $0.0000 |

## By model

| provider | model | calls | input_tokens | output_tokens | cost |
|---|---|---|---|---|---|
| gemini | gemini-3.5-flash-lite | 479 | 269,006 | 20,741 | $0.0149 |
| gemini | gemini-flash-lite-latest | 27 | 32,530 | 934 | $0.0000 |
