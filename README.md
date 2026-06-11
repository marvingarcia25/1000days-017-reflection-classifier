# 1000days-017-reflection-classifier

Day 017 of 1000 Days Challenge — a sentiment classifier with the AI **built from scratch**.

`day17_ReflectionClassifier` reads a short daily reflection ("how was your day?") and labels it **positive** or **negative** — then tells you which words drove the decision. The model is a multinomial Naive Bayes classifier written in plain C#. No ML.NET, no external model, no API. It trains in memory at startup from a labelled dataset baked into the build.

## Why this one

Days so far have been apps with hand-rolled logic (affirmations, habit streaks). This is the first day the *logic itself is the AI* — implemented from first principles rather than called from a library. Once Naive Bayes is in your hands, every heavier model is a variation on the same counting-and-probabilities idea.

## What it does

- **Multinomial Naive Bayes**, ~90 lines of model code, no ML dependencies.
- **Add-one (Laplace) smoothing** so unseen words never zero out a class.
- **Negation handling** — a word after *not / never / didn't* is tagged separately, so "not happy" reads negative.
- **Explains itself** — every prediction returns the words that pushed hardest and which way.
- Trained on 150 hand-labelled reflections embedded in the build.

## Accuracy

Measured, not asserted: **5-fold cross-validation ≈ 87%**, and the held-out test in the suite (every 5th example) scores **28/30 ≈ 93%**. The same algorithm and split were verified independently before being ported to C#, so the number in the test is real, not aspirational.

## Run it

```bash
dotnet run --project src
```

Then open the URL it prints (default **http://localhost:5000**) in a browser — the UI lets you type a reflection and watch each word pull the verdict positive or negative. Prefer the API directly?

```bash
curl -s -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"i kept my streak and i feel proud"}'
```

```json
{ "label": "positive", "confidence": 0.93,
  "topSignals": [ { "word": "proud", "lean": 2.1 }, ... ] }
```

## Tests

```bash
dotnet test
```

Covers the tokenizer (punctuation, negation scope), smoothing on unseen words, clear-cut positive/negative cases, the negation flip, and the held-out accuracy threshold. CI runs build + test on every push (`.github/workflows/ci.yml`).

## Layout

```
src/
  Program.cs                       minimal API (POST /classify) + static UI
  wwwroot/index.html               the UI — talks to /classify, draws the leanings
  Services/Tokenizer.cs            text -> tokens, with negation
  Services/NaiveBayesClassifier.cs the model, from scratch
  Data/TrainingData.cs             150 labelled reflections
tests/
  ClassifierTests.cs               xUnit suite
```

## What I'd reach for next

- Add a third **neutral** class and re-measure.
- Weight tokens by TF-IDF instead of raw counts.
- Add bigrams so phrases like "not bad" are learned directly.
- Feed it the journal entries from the habit tracker (Day 002) and score trends over time.

## Stack

C# / .NET 8, ASP.NET Core minimal API, xUnit. No machine-learning libraries — that's the point.

---

Day 17 of building a small thing every day.
