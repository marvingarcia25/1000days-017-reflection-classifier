# day17-AIbyAI

Day 017 of the 1000 Days Challenge: an AI app created by AI.

This project is a small reflection reader where the application, UI, API, and toy AI models were assembled by an AI coding agent. The result is intentionally transparent: the AI inside the app is not a hidden API call. It is C# code in this repository, trained at startup from 150 labelled reflections.

Live app: http://day017-reflection-classifier.azurewebsites.net

## What It Does

- **Reads your reflection** and classifies it as positive or negative.
- **Explains the decision** by showing which words pulled the result positive or negative.
- **Suggests next words** while you type using a tiny from-scratch micro-GPT.
- **Generates a reflection** from a positive or negative prompt.
- **Runs without ML libraries**: no ML.NET, no external model API, no hosted LLM inference.

## AI Created By AI

The app is deliberately meta: an AI coding agent built a tiny AI system.

The classifier is classical machine learning: a multinomial Naive Bayes model with Laplace smoothing and simple negation handling.

The generator is a toy language model: a single-head, decoder-style transformer with token embeddings, positional embeddings, causal self-attention, a feed-forward layer, and next-token sampling. It is tiny and data-starved, so its writing can be odd. The point is the mechanism: the model code is visible and runs inside this app.

## Run Locally

```bash
dotnet run --project src
```

Then open the printed local URL, usually:

```text
http://localhost:5000
```

## API

Classify text:

```bash
curl -s -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"i kept my streak and i feel proud"}'
```

Generate a reflection:

```bash
curl -s -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"sentiment":"positive","maxWords":20}'
```

Suggest next words:

```bash
curl -s -X POST http://localhost:5000/suggest \
  -H "Content-Type: application/json" \
  -d '{"text":"i feel"}'
```

## Tests

```bash
dotnet test
```

The suite covers tokenizer behavior, classifier accuracy, vocabulary behavior, transformer forward-pass behavior, and the language-model service.

## Layout

```text
src/
  Program.cs                       minimal API: /classify, /generate, /suggest
  wwwroot/index.html               static UI for classify, generate, autocomplete
  Data/TrainingData.cs             150 labelled reflections
  Services/Tokenizer.cs            classifier tokenizer with negation handling
  Services/NaiveBayesClassifier.cs from-scratch sentiment classifier
  Services/LmVocabulary.cs         word-level vocabulary for generation
  Services/Tensor.cs               small matrix parameter helper
  Services/MicroTransformer.cs     tiny single-head transformer
  Services/LanguageModelService.cs startup training, sampling, suggestions
tests/
  *.cs                             xUnit test suites
```

## Stack

C# / .NET 8, ASP.NET Core minimal API, vanilla HTML/CSS/JS, xUnit.

No machine-learning libraries. No external AI model calls. The AI in this app was created by AI, then runs as ordinary code.

---

Day 17 of building a small thing every day.
