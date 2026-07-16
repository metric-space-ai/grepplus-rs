# Function-purpose summary quality gate

This gate evaluates the short Qwen navigation hints emitted by the real
`greppy brief --json` product path. It does not ask the nano model to explain
code or replace source reading.

The checked-in case bank contains 204 deterministic functions: 34 each from
pinned Rust, Python, Java, TypeScript, Rust/Tokio, and Go repositories. Cases
store only repository commit, exact path/span/signature, and source digest;
third-party function bodies are reconstructed from the pinned checkout and are
not duplicated in release evidence.

```bash
python3 bench/agent_efficiency/real_corpus.py setup \
  --repos serde flask gson zod tokio hugo

python3 bench/summary_quality.py collect \
  --binary target/release/greppy \
  --device metal \
  --store-dir dev/summary-quality/store \
  --output bench/summary_quality/cases_v1.json

python3 bench/summary_quality.py run \
  --binary target/release/greppy \
  --device metal \
  --store-dir dev/summary-quality/store \
  --cases bench/summary_quality/cases_v1.json \
  --output dev/summary-quality/results.json

python3 bench/summary_quality.py judge \
  --cases bench/summary_quality/cases_v1.json \
  --results dev/summary-quality/results.json \
  --output dev/summary-quality/judgments.json

python3 bench/summary_quality.py gate \
  --cases bench/summary_quality/cases_v1.json \
  --results dev/summary-quality/results.json \
  --judgments dev/summary-quality/judgments.json \
  --output dev/summary-quality/gate.json
```

`MINIMAX_API_KEY` is read only by the judge step. The product inference remains
fully local. The fixed gate requires at least 200 functions, at least 85%
helpful direction hints, at most 5% misleading hints (re-registered 2026-07-16 from 2%; rationale in bench/summary_quality.py), no invented symbols or
signature echoes, and no deterministic brief-contract failures.
