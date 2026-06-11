# Day 17 Micro-GPT — Design Spec

**Date:** 2026-06-11
**Status:** Approved
**Repo:** 1000days-017-reflection-classifier

## Goal

Add a from-scratch, single-head transformer language model ("micro-GPT") to the Day 17 reflection classifier, written in plain C# with no ML libraries — same constraint as the existing Naive Bayes classifier. It trains at startup on the same 150 labelled reflections and powers two new features:

1. **Generate** — produce a new positive or negative reflection on demand.
2. **Autocomplete** — suggest the next word while the user types.

The Naive Bayes classifier is unchanged. The README gains an honest framing: this is a real (tiny) language model — the same mechanism as an LLM, roughly ten million times smaller — while the classifier remains classical ML.

## Non-goals

- No change to the classifier, its tokenizer, or `/classify`.
- No multi-head attention, no stacked blocks, no byte-pair encoding.
- No model persistence; training at startup is the design, as with the classifier.
- No new training data in this phase.

## Components

### 1. LM vocabulary + tokenizer (`src/Services/LmVocabulary.cs`)

- Word-level: lowercase, strip punctuation, split on whitespace. Natural word order — does **not** reuse the classifier's `Tokenizer`, whose `not_` negation tagging is classifier-specific.
- Built from `TrainingData.All` at startup (~400 words).
- Special tokens: `<pos>`, `<neg>`, `<end>`, `<unk>`.
- Interface: `Encode(string) -> int[]`, `Decode(int) -> string`, `Size`.

### 2. Model (`src/Services/MicroTransformer.cs`)

Architecture (decoder-only, one block):

```
token embeddings (d_model = 24)
  + learned positional embeddings (context window T = 8)
→ single-head causal self-attention (mask: position i attends to ≤ i)
→ residual + layer norm
→ feed-forward (24 → 64 → 24, tanh)
→ residual + layer norm
→ linear to vocab logits → softmax
```

~25k parameters. All forward and backward passes hand-written. Adam optimizer. Seeded RNG (`Random(seed)`) for reproducible training and sampling.

Training:
- Each reflection becomes the sequence `<pos>|<neg> w1 w2 … wn <end>`; training examples are all (context ≤ 8 tokens → next token) windows.
- Fixed epoch budget chosen so startup training stays under ~10 s on a dev machine. If it overshoots, shrink `d_model`/FFN width — do not silently lower the budget.
- Exposes final average cross-entropy loss for logging/tests.

Inference:
- `GenerateNext(int[] context) -> float[]` (probability over vocab, one forward pass).
- `Sample(sentimentToken, temperature, maxWords, rng)` — autoregressive sampling until `<end>` or cap; returns each word with the probability it was sampled at.

Sentiment control is **prompting**: generation starts from `<pos>` or `<neg>`.

### 3. API (`src/Program.cs`)

- `POST /generate` — body `{ "sentiment": "positive"|"negative", "temperature"?: number, "maxWords"?: number }`
  → `200 { "text": "...", "tokens": [ { "word": "...", "prob": 0.31 }, … ] }`
  → `400` on missing/invalid sentiment.
  Defaults: temperature 1.0 (clamped to [0.2, 2.0]), maxWords 20 (capped at 30).
- `POST /suggest` — body `{ "text": "..." }`
  → `200 { "suggestions": [ { "word": "...", "prob": 0.42 }, ×3 ] }`
  Uses the last ≤ 8 tokens of the input; unknown words map to `<unk>`; empty input is prompted from `<pos>` (sentence starters). Never 400s on empty text.
- Model registered as a singleton, trained in the same startup path as the classifier.

### 4. UI (`src/wwwroot/index.html`)

- **Generate section:** positive/negative toggle + "Generate" button. Output renders word by word; each word's background opacity reflects its sampled probability. A "classify this" button feeds the generated text into the existing classifier flow.
- **Autocomplete:** three suggestion chips under the input box, refreshed via debounced (~250 ms) calls to `/suggest`; clicking a chip appends the word.
- Match the existing page's styling and vanilla-JS approach.

## Data flow

Startup: `TrainingData.All` → build `LmVocabulary` → train `MicroTransformer` (seeded, epoch-capped) → DI singletons. Requests are pure forward passes; the model is immutable after training and safe for concurrent reads.

## Error handling

- Unknown input words → `<unk>` (never throw).
- `temperature` and `maxWords` clamped server-side.
- `/generate` with invalid sentiment → 400 with an error message, matching `/classify`'s style.

## Testing (xUnit, TDD)

1. **Gradient check (the critical test):** analytical gradients from hand-written backprop vs central finite differences on a tiny model (e.g., vocab 12, d_model 6, T 4) — max relative error below 1e-3. This is the only reliable verification of hand-rolled attention gradients.
2. Training on a tiny corpus reduces cross-entropy loss (final < initial).
3. Seeded generation emits only known vocabulary words and terminates within maxWords.
4. `<pos>` vs `<neg>` prompts yield measurably different next-token distributions.
5. Vocabulary round-trip: encode → decode is identity for in-vocab words; OOV → `<unk>`.
6. API contract tests for `/generate` and `/suggest` (status codes + response shape) via `WebApplicationFactory`.

Tests use reduced dimensions/epochs so CI stays fast. Existing classifier tests must keep passing.

## Risks

- **Output quality:** 150 short texts is a tiny corpus; generations will be short and sometimes garbled. Expected — the per-word probabilities in the UI make the model's uncertainty visible instead of hiding it.
- **Startup time:** epoch cap + small dims bound it; verified before deploy.
- **Backprop bugs:** mitigated by the gradient-check test written first.

## Delivery

Implement → all tests green → run locally and smoke-test `/generate`, `/suggest`, UI → update README → push to GitHub (CI) → redeploy to Azure (`day017-reflection-classifier`, zip via `tar`, SCM basic auth already enabled).
