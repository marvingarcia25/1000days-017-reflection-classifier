# Day 17 Micro-GPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a from-scratch single-head transformer language model (micro-GPT) to the Day 17 app, powering `POST /generate` (write a positive/negative reflection) and `POST /suggest` (next-word autocomplete), trained at startup on the existing 150 reflections.

**Architecture:** Decoder-only transformer, one block: token + learned positional embeddings (context window 8, d_model 24) → single-head causal self-attention → residual + layer norm → tanh FFN (24→64→24) → residual + layer norm → linear to vocab logits. Hand-written forward/backward in plain C# `double` math, Adam optimizer, seeded RNG. Sentiment control via `<pos>`/`<neg>` prompt tokens. The Naive Bayes classifier is untouched.

**Tech Stack:** C# / .NET 8, ASP.NET Core minimal API, xUnit, `Microsoft.AspNetCore.Mvc.Testing` (new test dep). No ML libraries — that's the point.

**Spec:** `docs/superpowers/specs/2026-06-11-day17-micro-gpt-design.md`

**Conventions for all tasks:**
- Repo root: `C:\Users\marvi\source\repos\1000\day17_AICreatedAI` — all paths below are relative to it. Run all commands from the repo root.
- Commit author email must be `8763989+marvingarcia25@users.noreply.github.com` (GitHub blocks the default email). Use `git -c user.email="8763989+marvingarcia25@users.noreply.github.com" commit …`.
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Existing namespaces: `Day17ReflectionClassifier.Services`, `.Data`. Match them.
- `TrainingData.All` is `IReadOnlyList<(string Text, string Label)>`, labels are `"pos"` / `"neg"`.

---

### Task 1: LM vocabulary

**Files:**
- Create: `src/Services/LmVocabulary.cs`
- Test: `tests/LmVocabularyTests.cs`

- [ ] **Step 1: Write the failing tests**

Create `tests/LmVocabularyTests.cs`:

```csharp
using Day17ReflectionClassifier.Services;
using Xunit;

namespace Day17ReflectionClassifier.Tests;

public class LmVocabularyTests
{
    [Fact]
    public void Specials_come_first_and_have_stable_ids()
    {
        var v = new LmVocabulary(new[] { "a happy day" });
        Assert.Equal(0, v.PosId);
        Assert.Equal(1, v.NegId);
        Assert.Equal(2, v.EndId);
        Assert.Equal(3, v.UnkId);
        Assert.Equal("<pos>", v.Decode(v.PosId));
    }

    [Fact]
    public void Encode_decode_round_trips_known_words()
    {
        var v = new LmVocabulary(new[] { "I felt Great today!" });
        var ids = v.Encode("great TODAY");
        Assert.Equal(2, ids.Length);
        Assert.Equal("great", v.Decode(ids[0]));
        Assert.Equal("today", v.Decode(ids[1]));
    }

    [Fact]
    public void Unknown_words_map_to_unk()
    {
        var v = new LmVocabulary(new[] { "a happy day" });
        var ids = v.Encode("zebra happy");
        Assert.Equal(v.UnkId, ids[0]);
        Assert.NotEqual(v.UnkId, ids[1]);
    }

    [Fact]
    public void Punctuation_is_stripped_and_apostrophes_kept()
    {
        var v = new LmVocabulary(new[] { "i didn't stop, really." });
        var ids = v.Encode("didn't really");
        Assert.Equal("didn't", v.Decode(ids[0]));
        Assert.Equal("really", v.Decode(ids[1]));
    }

    [Fact]
    public void Size_counts_specials_plus_unique_words()
    {
        var v = new LmVocabulary(new[] { "good day", "good night" });
        Assert.Equal(4 + 3, v.Size); // specials + good, day, night
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test --filter "FullyQualifiedName~LmVocabularyTests"`
Expected: build error — `LmVocabulary` does not exist.

- [ ] **Step 3: Write the implementation**

Create `src/Services/LmVocabulary.cs`:

```csharp
using System.Text;

namespace Day17ReflectionClassifier.Services;

/// <summary>
/// Word-level vocabulary for the language model. Natural word order —
/// deliberately separate from the classifier's Tokenizer, whose not_ tagging
/// is classifier-specific. Specials occupy ids 0..3.
/// </summary>
public sealed class LmVocabulary
{
    public const string Pos = "<pos>", Neg = "<neg>", End = "<end>", Unk = "<unk>";

    private readonly Dictionary<string, int> _ids = new();
    private readonly List<string> _words = new();

    public LmVocabulary(IEnumerable<string> texts)
    {
        foreach (var special in new[] { Pos, Neg, End, Unk })
            Add(special);
        foreach (var text in texts)
            foreach (var word in Split(text))
                Add(word);
    }

    public int Size => _words.Count;
    public int PosId => 0;
    public int NegId => 1;
    public int EndId => 2;
    public int UnkId => 3;

    public int[] Encode(string text)
    {
        var result = new List<int>();
        foreach (var word in Split(text))
            result.Add(_ids.TryGetValue(word, out var id) ? id : UnkId);
        return result.ToArray();
    }

    public string Decode(int id) => _words[id];

    private void Add(string word)
    {
        if (!_ids.ContainsKey(word))
        {
            _ids[word] = _words.Count;
            _words.Add(word);
        }
    }

    private static IEnumerable<string> Split(string text)
    {
        var sb = new StringBuilder();
        foreach (var ch in text.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(ch) || ch == '\'')
            {
                sb.Append(ch);
            }
            else if (sb.Length > 0)
            {
                yield return sb.ToString();
                sb.Clear();
            }
        }
        if (sb.Length > 0)
            yield return sb.ToString();
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test --filter "FullyQualifiedName~LmVocabularyTests"`
Expected: 5 passed. Also run the full suite (`dotnet test`) — the 9 existing classifier tests must still pass.

- [ ] **Step 5: Commit**

```
git add src/Services/LmVocabulary.cs tests/LmVocabularyTests.cs
git -c user.email="8763989+marvingarcia25@users.noreply.github.com" commit -m "feat: LM word vocabulary with special tokens"  (append the Co-Authored-By line)
```

---

### Task 2: Tensor + transformer forward pass

**Files:**
- Create: `src/Services/Tensor.cs`
- Create: `src/Services/MicroTransformer.cs`
- Test: `tests/MicroTransformerTests.cs`

- [ ] **Step 1: Write the failing tests**

Create `tests/MicroTransformerTests.cs`:

```csharp
using Day17ReflectionClassifier.Services;
using Xunit;

namespace Day17ReflectionClassifier.Tests;

public class MicroTransformerTests
{
    [Fact]
    public void Forward_probabilities_are_valid_distribution()
    {
        var model = new MicroTransformer(vocabSize: 12, dModel: 6, hidden: 10, seed: 42);
        var logits = model.LastLogits(new[] { 1, 5, 7 });
        Assert.Equal(12, logits.Length);
        // softmax of logits must be a valid distribution
        double max = logits.Max();
        double sum = logits.Sum(z => Math.Exp(z - max));
        Assert.True(double.IsFinite(sum) && sum > 0);
    }

    [Fact]
    public void Forward_is_deterministic_for_same_seed()
    {
        var a = new MicroTransformer(12, 6, 10, seed: 7).LastLogits(new[] { 2, 3 });
        var b = new MicroTransformer(12, 6, 10, seed: 7).LastLogits(new[] { 2, 3 });
        Assert.Equal(a, b);
    }

    [Fact]
    public void Context_longer_than_window_is_clamped_to_last_tokens()
    {
        var model = new MicroTransformer(12, 6, 10, seed: 42);
        var longCtx = new[] { 9, 9, 9, 1, 2, 3, 4, 5, 6, 7, 8 }; // 11 tokens
        var clamped = new[] { 1, 2, 3, 4, 5, 6, 7, 8 };           // last 8
        Assert.Equal(model.LastLogits(clamped), model.LastLogits(longCtx));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test --filter "FullyQualifiedName~MicroTransformerTests"`
Expected: build error — `MicroTransformer` does not exist.

- [ ] **Step 3: Write Tensor and the forward pass**

Create `src/Services/Tensor.cs`:

```csharp
namespace Day17ReflectionClassifier.Services;

/// <summary>Row-major matrix parameter with gradient and Adam moment buffers.</summary>
public sealed class Tensor
{
    public readonly int Rows;
    public readonly int Cols;
    public readonly double[] Data;
    public readonly double[] Grad;
    internal readonly double[] M; // Adam first moment
    internal readonly double[] V; // Adam second moment

    public Tensor(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        Data = new double[rows * cols];
        Grad = new double[rows * cols];
        M = new double[rows * cols];
        V = new double[rows * cols];
    }

    public double this[int r, int c]
    {
        get => Data[r * Cols + c];
        set => Data[r * Cols + c] = value;
    }
}
```

Create `src/Services/MicroTransformer.cs` with the forward pass. Backward (`ForwardBackward`), `ZeroGrads`, `AdamStep`, and `Train` are added in Tasks 3–4; for now the file contains exactly this:

```csharp
namespace Day17ReflectionClassifier.Services;

/// <summary>
/// A micro-GPT: decoder-only transformer with one single-head causal
/// self-attention block, written from scratch. All forward and backward
/// passes are hand-derived — no ML libraries.
/// </summary>
public sealed class MicroTransformer
{
    public const int ContextWindow = 8;
    private const double LnEps = 1e-5;

    private readonly int _vocab;
    private readonly int _d;
    private readonly int _h;

    private readonly Tensor _emb;   // vocab × d   token embeddings
    private readonly Tensor _pos;   // T × d       learned positional embeddings
    private readonly Tensor _wq, _wk, _wv, _wo; // d × d attention projections
    private readonly Tensor _lg1, _lb1;         // 1 × d layer norm 1 gain/bias
    private readonly Tensor _w1, _b1;           // d × h, 1 × h FFN in
    private readonly Tensor _w2, _b2;           // h × d, 1 × d FFN out
    private readonly Tensor _lg2, _lb2;         // 1 × d layer norm 2 gain/bias
    private readonly Tensor _wout, _bout;       // d × vocab, 1 × vocab output head
    private readonly Tensor[] _params;
    private int _adamStep;

    public MicroTransformer(int vocabSize, int dModel, int hidden, int seed)
    {
        _vocab = vocabSize;
        _d = dModel;
        _h = hidden;
        var rng = new Random(seed);
        _emb = Init(vocabSize, _d, rng);
        _pos = Init(ContextWindow, _d, rng);
        _wq = Init(_d, _d, rng);
        _wk = Init(_d, _d, rng);
        _wv = Init(_d, _d, rng);
        _wo = Init(_d, _d, rng);
        _lg1 = Ones(1, _d);
        _lb1 = new Tensor(1, _d);
        _w1 = Init(_d, _h, rng);
        _b1 = new Tensor(1, _h);
        _w2 = Init(_h, _d, rng);
        _b2 = new Tensor(1, _d);
        _lg2 = Ones(1, _d);
        _lb2 = new Tensor(1, _d);
        _wout = Init(_d, vocabSize, rng);
        _bout = new Tensor(1, vocabSize);
        _params = new[] { _emb, _pos, _wq, _wk, _wv, _wo, _lg1, _lb1, _w1, _b1, _w2, _b2, _lg2, _lb2, _wout, _bout };
    }

    public IReadOnlyList<Tensor> Parameters => _params;

    /// <summary>Logits for the next token after the given context (last ≤8 tokens used).</summary>
    public double[] LastLogits(int[] tokens)
    {
        var clamped = tokens.Length <= ContextWindow ? tokens : tokens[^ContextWindow..];
        var c = Forward(clamped);
        return c.Logits[c.L - 1];
    }

    /// <summary>Average cross-entropy at positions where targets[i] >= 0 (forward only).</summary>
    public double ComputeLoss(int[] tokens, int[] targets)
    {
        var c = Forward(tokens);
        double loss = 0;
        int n = 0;
        for (int i = 0; i < c.L; i++)
        {
            if (targets[i] < 0) continue;
            n++;
            loss += -Math.Log(c.Probs[i][targets[i]] + 1e-12);
        }
        return n == 0 ? 0 : loss / n;
    }

    private sealed class Cache
    {
        public int L;
        public int[] Tokens = Array.Empty<int>();
        public double[][] X = default!, Q = default!, K = default!, Vv = default!,
            Alpha = default!, A = default!, U1Hat = default!, X1 = default!,
            Act = default!, U2Hat = default!, X2 = default!, Logits = default!, Probs = default!;
        public double[] Sig1 = default!, Sig2 = default!;
    }

    private Cache Forward(int[] tokens)
    {
        int L = tokens.Length;
        var c = new Cache
        {
            L = L,
            Tokens = tokens,
            X = NewMat(L, _d), Q = NewMat(L, _d), K = NewMat(L, _d), Vv = NewMat(L, _d),
            Alpha = NewMat(L, L), A = NewMat(L, _d),
            U1Hat = NewMat(L, _d), X1 = NewMat(L, _d),
            Act = NewMat(L, _h),
            U2Hat = NewMat(L, _d), X2 = NewMat(L, _d),
            Logits = NewMat(L, _vocab), Probs = NewMat(L, _vocab),
            Sig1 = new double[L], Sig2 = new double[L],
        };

        // embeddings + positions
        for (int i = 0; i < L; i++)
            for (int e = 0; e < _d; e++)
                c.X[i][e] = _emb[tokens[i], e] + _pos[i, e];

        // attention projections
        for (int i = 0; i < L; i++)
        {
            MatVecRow(c.X[i], _wq, c.Q[i]);
            MatVecRow(c.X[i], _wk, c.K[i]);
            MatVecRow(c.X[i], _wv, c.Vv[i]);
        }

        // causal single-head attention
        double scale = 1.0 / Math.Sqrt(_d);
        for (int i = 0; i < L; i++)
        {
            var scores = new double[i + 1];
            double max = double.NegativeInfinity;
            for (int j = 0; j <= i; j++)
            {
                double s = 0;
                for (int e = 0; e < _d; e++) s += c.Q[i][e] * c.K[j][e];
                scores[j] = s * scale;
                if (scores[j] > max) max = scores[j];
            }
            double sum = 0;
            for (int j = 0; j <= i; j++)
            {
                scores[j] = Math.Exp(scores[j] - max);
                sum += scores[j];
            }
            for (int j = 0; j <= i; j++) c.Alpha[i][j] = scores[j] / sum;

            for (int e = 0; e < _d; e++)
            {
                double a = 0;
                for (int j = 0; j <= i; j++) a += c.Alpha[i][j] * c.Vv[j][e];
                c.A[i][e] = a;
            }
        }

        var attnOut = new double[_d];
        var ffnOut = new double[_d];
        var u = new double[_d];
        for (int i = 0; i < L; i++)
        {
            // residual + layer norm 1
            MatVecRow(c.A[i], _wo, attnOut);
            for (int e = 0; e < _d; e++) u[e] = c.X[i][e] + attnOut[e];
            LayerNormForward(u, _lg1, _lb1, c.U1Hat[i], c.X1[i], out c.Sig1[i]);

            // FFN: tanh(x1·W1 + b1)·W2 + b2
            for (int k = 0; k < _h; k++)
            {
                double s = _b1[0, k];
                for (int e = 0; e < _d; e++) s += c.X1[i][e] * _w1[e, k];
                c.Act[i][k] = Math.Tanh(s);
            }
            for (int e = 0; e < _d; e++)
            {
                double f = _b2[0, e];
                for (int k = 0; k < _h; k++) f += c.Act[i][k] * _w2[k, e];
                ffnOut[e] = f;
            }

            // residual + layer norm 2
            for (int e = 0; e < _d; e++) u[e] = c.X1[i][e] + ffnOut[e];
            LayerNormForward(u, _lg2, _lb2, c.U2Hat[i], c.X2[i], out c.Sig2[i]);

            // output head + softmax
            double maxz = double.NegativeInfinity;
            for (int v = 0; v < _vocab; v++)
            {
                double z = _bout[0, v];
                for (int e = 0; e < _d; e++) z += c.X2[i][e] * _wout[e, v];
                c.Logits[i][v] = z;
                if (z > maxz) maxz = z;
            }
            double zsum = 0;
            for (int v = 0; v < _vocab; v++)
            {
                c.Probs[i][v] = Math.Exp(c.Logits[i][v] - maxz);
                zsum += c.Probs[i][v];
            }
            for (int v = 0; v < _vocab; v++) c.Probs[i][v] /= zsum;
        }
        return c;
    }

    private static void LayerNormForward(double[] u, Tensor g, Tensor b, double[] uHat, double[] y, out double sigma)
    {
        int d = u.Length;
        double mu = 0;
        for (int e = 0; e < d; e++) mu += u[e];
        mu /= d;
        double variance = 0;
        for (int e = 0; e < d; e++)
        {
            double t = u[e] - mu;
            variance += t * t;
        }
        variance /= d;
        sigma = Math.Sqrt(variance + LnEps);
        for (int e = 0; e < d; e++)
        {
            uHat[e] = (u[e] - mu) / sigma;
            y[e] = g[0, e] * uHat[e] + b[0, e];
        }
    }

    private static void MatVecRow(double[] x, Tensor w, double[] outRow)
    {
        // outRow = x · w   (x: 1×Rows, w: Rows×Cols)
        for (int col = 0; col < w.Cols; col++)
        {
            double s = 0;
            for (int r = 0; r < w.Rows; r++) s += x[r] * w[r, col];
            outRow[col] = s;
        }
    }

    private static double[][] NewMat(int rows, int cols)
    {
        var m = new double[rows][];
        for (int i = 0; i < rows; i++) m[i] = new double[cols];
        return m;
    }

    private static Tensor Init(int rows, int cols, Random rng)
    {
        var t = new Tensor(rows, cols);
        for (int i = 0; i < t.Data.Length; i++)
            t.Data[i] = Gaussian(rng) * 0.08;
        return t;
    }

    private static Tensor Ones(int rows, int cols)
    {
        var t = new Tensor(rows, cols);
        Array.Fill(t.Data, 1.0);
        return t;
    }

    private static double Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test --filter "FullyQualifiedName~MicroTransformerTests"`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```
git add src/Services/Tensor.cs src/Services/MicroTransformer.cs tests/MicroTransformerTests.cs
git -c user.email="8763989+marvingarcia25@users.noreply.github.com" commit -m "feat: micro-transformer forward pass (single-head causal attention)"  (append the Co-Authored-By line)
```

---

### Task 3: Backward pass + gradient check

This is the correctness-critical task. The gradient check compares every analytical gradient against central finite differences — if the backward code below is transcribed exactly, it passes.

**Files:**
- Modify: `src/Services/MicroTransformer.cs` (add `ZeroGrads`, `ForwardBackward`, two private helpers)
- Modify: `tests/MicroTransformerTests.cs` (add one test)

- [ ] **Step 1: Write the failing test**

Add to `tests/MicroTransformerTests.cs` inside the class:

```csharp
    [Fact]
    public void Gradients_match_finite_differences()
    {
        var model = new MicroTransformer(vocabSize: 12, dModel: 6, hidden: 10, seed: 42);
        int[] tokens = { 1, 5, 7, 3 };
        int[] targets = { 5, 7, 3, 2 };

        model.ZeroGrads();
        model.ForwardBackward(tokens, targets);

        const double eps = 1e-5;
        foreach (var p in model.Parameters)
        {
            for (int i = 0; i < p.Data.Length; i++)
            {
                double orig = p.Data[i];
                p.Data[i] = orig + eps;
                double lossPlus = model.ComputeLoss(tokens, targets);
                p.Data[i] = orig - eps;
                double lossMinus = model.ComputeLoss(tokens, targets);
                p.Data[i] = orig;

                double numeric = (lossPlus - lossMinus) / (2 * eps);
                double analytic = p.Grad[i];
                double denom = Math.Max(1e-6, Math.Abs(numeric) + Math.Abs(analytic));
                double relErr = Math.Abs(numeric - analytic) / denom;
                Assert.True(relErr < 1e-3,
                    $"grad mismatch in {p.Rows}x{p.Cols} tensor at {i}: numeric={numeric:G6} analytic={analytic:G6}");
            }
        }
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test --filter "FullyQualifiedName~Gradients_match_finite_differences"`
Expected: build error — `ZeroGrads` / `ForwardBackward` do not exist.

- [ ] **Step 3: Write the backward pass**

Add these members to `MicroTransformer` (after `ComputeLoss`):

```csharp
    public void ZeroGrads()
    {
        foreach (var p in _params)
            Array.Clear(p.Grad, 0, p.Grad.Length);
    }

    /// <summary>
    /// Forward + backward for one sequence. targets[i] is the expected token
    /// at position i+1 (or -1 for "no loss here"). Gradients of the MEAN loss
    /// are ACCUMULATED into Parameters[..].Grad. Returns the mean loss.
    /// </summary>
    public double ForwardBackward(int[] tokens, int[] targets)
    {
        var c = Forward(tokens);
        int L = c.L;

        double loss = 0;
        int n = 0;
        for (int i = 0; i < L; i++)
        {
            if (targets[i] < 0) continue;
            n++;
            loss += -Math.Log(c.Probs[i][targets[i]] + 1e-12);
        }
        if (n == 0) return 0;
        double invN = 1.0 / n;

        // ----- output head: dz = (p - onehot)/n ; accumulate Wout, bout; build dX2
        var dX2 = NewMat(L, _d);
        for (int i = 0; i < L; i++)
        {
            if (targets[i] < 0) continue;
            for (int v = 0; v < _vocab; v++)
            {
                double dz = c.Probs[i][v] * invN;
                if (v == targets[i]) dz -= invN;
                _bout.Grad[v] += dz;
                for (int e = 0; e < _d; e++)
                {
                    _wout.Grad[e * _vocab + v] += c.X2[i][e] * dz;
                    dX2[i][e] += dz * _wout[e, v];
                }
            }
        }

        // ----- layer norm 2 + FFN (residual: U2 = X1 + ffn)
        var dX1 = NewMat(L, _d);
        var dU = new double[_d];
        var dAct = new double[_h];
        for (int i = 0; i < L; i++)
        {
            LayerNormBackward(dX2[i], c.U2Hat[i], c.Sig2[i], _lg2, _lb2, dU);

            // FFN backward: ffn = tanh(X1·W1 + b1)·W2 + b2, with dFfn = dU
            for (int k = 0; k < _h; k++)
            {
                double da = 0;
                for (int e = 0; e < _d; e++)
                {
                    _w2.Grad[k * _d + e] += c.Act[i][k] * dU[e];
                    da += dU[e] * _w2[k, e];
                }
                dAct[k] = da;
            }
            for (int e = 0; e < _d; e++) _b2.Grad[e] += dU[e];
            for (int k = 0; k < _h; k++)
            {
                double ds = dAct[k] * (1 - c.Act[i][k] * c.Act[i][k]); // tanh'
                _b1.Grad[k] += ds;
                for (int e = 0; e < _d; e++)
                {
                    _w1.Grad[e * _h + k] += c.X1[i][e] * ds;
                    dX1[i][e] += ds * _w1[e, k];
                }
            }
            for (int e = 0; e < _d; e++) dX1[i][e] += dU[e]; // residual branch
        }

        // ----- layer norm 1 + attention output projection (residual: U1 = X + A·Wo)
        var dX = NewMat(L, _d);
        var dA = NewMat(L, _d);
        for (int i = 0; i < L; i++)
        {
            LayerNormBackward(dX1[i], c.U1Hat[i], c.Sig1[i], _lg1, _lb1, dU);
            for (int e = 0; e < _d; e++) dX[i][e] += dU[e]; // residual branch
            // attnOut = A[i]·Wo ; dAttnOut = dU
            for (int f = 0; f < _d; f++)
            {
                double s = 0;
                for (int e = 0; e < _d; e++)
                {
                    _wo.Grad[f * _d + e] += c.A[i][f] * dU[e];
                    s += dU[e] * _wo[f, e];
                }
                dA[i][f] += s;
            }
        }

        // ----- attention backward: A[i] = sum_j alpha_ij V[j], scores scaled by 1/sqrt(d)
        var dQ = NewMat(L, _d);
        var dK = NewMat(L, _d);
        var dV = NewMat(L, _d);
        double scale = 1.0 / Math.Sqrt(_d);
        for (int i = 0; i < L; i++)
        {
            var dAlpha = new double[i + 1];
            for (int j = 0; j <= i; j++)
            {
                double da = 0;
                for (int e = 0; e < _d; e++)
                {
                    dV[j][e] += c.Alpha[i][j] * dA[i][e];
                    da += dA[i][e] * c.Vv[j][e];
                }
                dAlpha[j] = da;
            }
            // softmax backward over row i
            double dot = 0;
            for (int j = 0; j <= i; j++) dot += dAlpha[j] * c.Alpha[i][j];
            for (int j = 0; j <= i; j++)
            {
                double dScore = c.Alpha[i][j] * (dAlpha[j] - dot) * scale;
                for (int e = 0; e < _d; e++)
                {
                    dQ[i][e] += dScore * c.K[j][e];
                    dK[j][e] += dScore * c.Q[i][e];
                }
            }
        }

        // ----- Q/K/V projections back to X
        for (int i = 0; i < L; i++)
        {
            LinearBackward(c.X[i], _wq, dQ[i], dX[i]);
            LinearBackward(c.X[i], _wk, dK[i], dX[i]);
            LinearBackward(c.X[i], _wv, dV[i], dX[i]);
        }

        // ----- embeddings + positions
        for (int i = 0; i < L; i++)
            for (int e = 0; e < _d; e++)
            {
                _emb.Grad[c.Tokens[i] * _d + e] += dX[i][e];
                _pos.Grad[i * _d + e] += dX[i][e];
            }

        return loss * invN;
    }

    /// <summary>Backward of y = g ⊙ û + b where û = (u-μ)/σ. Accumulates g/b grads, writes du.</summary>
    private static void LayerNormBackward(double[] dy, double[] uHat, double sigma, Tensor g, Tensor b, double[] du)
    {
        int d = dy.Length;
        double meanDuHat = 0, meanDuHatUhat = 0;
        var duHat = new double[d];
        for (int e = 0; e < d; e++)
        {
            g.Grad[e] += dy[e] * uHat[e];
            b.Grad[e] += dy[e];
            duHat[e] = dy[e] * g[0, e];
            meanDuHat += duHat[e];
            meanDuHatUhat += duHat[e] * uHat[e];
        }
        meanDuHat /= d;
        meanDuHatUhat /= d;
        for (int e = 0; e < d; e++)
            du[e] = (duHat[e] - meanDuHat - uHat[e] * meanDuHatUhat) / sigma;
    }

    /// <summary>Backward of y = x·w. Accumulates w.Grad and dx given dy.</summary>
    private static void LinearBackward(double[] x, Tensor w, double[] dy, double[] dx)
    {
        for (int r = 0; r < w.Rows; r++)
        {
            double s = 0;
            for (int col = 0; col < w.Cols; col++)
            {
                w.Grad[r * w.Cols + col] += x[r] * dy[col];
                s += dy[col] * w[r, col];
            }
            dx[r] += s;
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test --filter "FullyQualifiedName~MicroTransformerTests"`
Expected: 4 passed. If the gradient check fails, the failure message names the tensor shape and index — compare the corresponding backward block against this plan line by line before changing any math.

- [ ] **Step 5: Commit**

```
git add src/Services/MicroTransformer.cs tests/MicroTransformerTests.cs
git -c user.email="8763989+marvingarcia25@users.noreply.github.com" commit -m "feat: hand-derived backward pass, verified by finite-difference gradient check"  (append the Co-Authored-By line)
```

---

### Task 4: Adam optimizer + training loop

**Files:**
- Modify: `src/Services/MicroTransformer.cs` (add `AdamStep`, `Train`, `Shuffle`)
- Modify: `tests/MicroTransformerTests.cs` (add one test)

- [ ] **Step 1: Write the failing test**

Add to `tests/MicroTransformerTests.cs`:

```csharp
    [Fact]
    public void Training_reduces_loss()
    {
        var vocab = new LmVocabulary(new[] { "good day today", "bad day today" });
        var seqs = new List<int[]>
        {
            new[] { vocab.PosId }.Concat(vocab.Encode("good day today")).Append(vocab.EndId).ToArray(),
            new[] { vocab.NegId }.Concat(vocab.Encode("bad day today")).Append(vocab.EndId).ToArray(),
        };
        var model = new MicroTransformer(vocab.Size, dModel: 8, hidden: 16, seed: 1);
        double initial = seqs.Average(s => model.ComputeLoss(s[..^1], s[1..]));
        double final = model.Train(seqs, epochs: 200, learningRate: 0.01, seed: 1);
        Assert.True(final < initial * 0.5, $"initial={initial:F3}, final={final:F3}");
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test --filter "FullyQualifiedName~Training_reduces_loss"`
Expected: build error — `Train` does not exist.

- [ ] **Step 3: Write the optimizer and training loop**

Add to `MicroTransformer`:

```csharp
    public void AdamStep(double learningRate)
    {
        _adamStep++;
        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        double c1 = 1 - Math.Pow(beta1, _adamStep);
        double c2 = 1 - Math.Pow(beta2, _adamStep);
        foreach (var p in _params)
        {
            for (int i = 0; i < p.Data.Length; i++)
            {
                double g = p.Grad[i];
                p.M[i] = beta1 * p.M[i] + (1 - beta1) * g;
                p.V[i] = beta2 * p.V[i] + (1 - beta2) * g * g;
                p.Data[i] -= learningRate * (p.M[i] / c1) / (Math.Sqrt(p.V[i] / c2) + eps);
            }
        }
    }

    /// <summary>
    /// Trains on full sequences, split into chunks of ContextWindow+1 tokens
    /// (each chunk yields next-token targets at every position). One Adam step
    /// per chunk. Returns the mean loss of the final epoch.
    /// </summary>
    public double Train(IReadOnlyList<int[]> sequences, int epochs, double learningRate, int seed)
    {
        var rng = new Random(seed);
        var chunks = new List<int[]>();
        foreach (var seq in sequences)
        {
            for (int start = 0; start + 1 < seq.Length; start += ContextWindow)
            {
                int len = Math.Min(ContextWindow + 1, seq.Length - start);
                if (len < 2) break;
                var chunk = new int[len];
                Array.Copy(seq, start, chunk, 0, len);
                chunks.Add(chunk);
            }
        }

        double lastEpochLoss = 0;
        for (int ep = 0; ep < epochs; ep++)
        {
            Shuffle(chunks, rng);
            double sum = 0;
            foreach (var chunk in chunks)
            {
                int len = chunk.Length - 1;
                var tokens = new int[len];
                var targets = new int[len];
                for (int i = 0; i < len; i++)
                {
                    tokens[i] = chunk[i];
                    targets[i] = chunk[i + 1];
                }
                ZeroGrads();
                sum += ForwardBackward(tokens, targets);
                AdamStep(learningRate);
            }
            lastEpochLoss = sum / chunks.Count;
        }
        return lastEpochLoss;
    }

    private static void Shuffle(List<int[]> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test`
Expected: all tests pass (9 classifier + 5 vocab + 5 transformer).

- [ ] **Step 5: Commit**

```
git add src/Services/MicroTransformer.cs tests/MicroTransformerTests.cs
git -c user.email="8763989+marvingarcia25@users.noreply.github.com" commit -m "feat: Adam optimizer and chunked training loop"  (append the Co-Authored-By line)
```

---

### Task 5: LanguageModelService (generation + suggestions)

**Files:**
- Create: `src/Services/LanguageModelService.cs`
- Test: `tests/LanguageModelServiceTests.cs`

- [ ] **Step 1: Write the failing tests**

Create `tests/LanguageModelServiceTests.cs`:

```csharp
using Day17ReflectionClassifier.Data;
using Day17ReflectionClassifier.Services;
using Xunit;

namespace Day17ReflectionClassifier.Tests;

public class LanguageModelServiceTests
{
    // Training is the slow part — share one trained service across tests.
    private static readonly Lazy<LanguageModelService> Shared =
        new(() => new LanguageModelService(TrainingData.All, epochs: 40));

    [Fact]
    public void Generation_emits_known_words_and_terminates()
    {
        var result = Shared.Value.Generate("positive", temperature: 1.0, maxWords: 20);
        Assert.True(result.Tokens.Count <= 20);
        foreach (var t in result.Tokens)
        {
            Assert.False(t.Word.StartsWith('<'), $"special token leaked: {t.Word}");
            Assert.InRange(t.Prob, 0.0, 1.0);
        }
        Assert.Equal(string.Join(" ", result.Tokens.Select(t => t.Word)), result.Text);
    }

    [Fact]
    public void Pos_and_neg_prompts_yield_different_next_word_distributions()
    {
        var lm = Shared.Value;
        double[] p = Softmax(lm.Model.LastLogits(new[] { lm.Vocab.PosId }));
        double[] n = Softmax(lm.Model.LastLogits(new[] { lm.Vocab.NegId }));
        double tv = 0;
        for (int v = 0; v < p.Length; v++) tv += Math.Abs(p[v] - n[v]);
        tv /= 2;
        Assert.True(tv > 0.05, $"total variation distance = {tv:F4}");
    }

    [Fact]
    public void Suggest_returns_top_words_without_specials()
    {
        var suggestions = Shared.Value.Suggest("i feel");
        Assert.Equal(3, suggestions.Count);
        Assert.All(suggestions, s => Assert.False(s.Word.StartsWith('<')));
        Assert.True(suggestions[0].Prob >= suggestions[1].Prob);
        Assert.True(suggestions[1].Prob >= suggestions[2].Prob);
    }

    [Fact]
    public void Suggest_handles_empty_and_unknown_text()
    {
        Assert.Equal(3, Shared.Value.Suggest("").Count);
        Assert.Equal(3, Shared.Value.Suggest("xylophone qwertyuiop").Count);
    }

    [Fact]
    public void Generation_is_reproducible_for_same_seed()
    {
        var a = new LanguageModelService(TrainingData.All.Take(20).ToList(), epochs: 5, seed: 99);
        var b = new LanguageModelService(TrainingData.All.Take(20).ToList(), epochs: 5, seed: 99);
        Assert.Equal(a.Generate("positive", 1.0, 10).Text, b.Generate("positive", 1.0, 10).Text);
    }

    private static double[] Softmax(double[] z)
    {
        double max = z.Max();
        var p = z.Select(v => Math.Exp(v - max)).ToArray();
        double sum = p.Sum();
        return p.Select(v => v / sum).ToArray();
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `dotnet test --filter "FullyQualifiedName~LanguageModelServiceTests"`
Expected: build error — `LanguageModelService` does not exist.

- [ ] **Step 3: Write the implementation**

Create `src/Services/LanguageModelService.cs`:

```csharp
namespace Day17ReflectionClassifier.Services;

public sealed record GeneratedToken(string Word, double Prob);
public sealed record GenerationResult(string Text, IReadOnlyList<GeneratedToken> Tokens);

/// <summary>
/// Trains the micro-GPT on the labelled reflections at construction time and
/// exposes generation + next-word suggestions. Immutable after training;
/// safe for concurrent reads (the sampling RNG is locked).
/// </summary>
public sealed class LanguageModelService
{
    public const int MaxWordsCap = 30;

    private readonly Random _rng;
    private readonly object _rngLock = new();

    public LanguageModelService(
        IReadOnlyList<(string Text, string Label)> examples,
        int dModel = 24, int hidden = 64, int epochs = 50, double learningRate = 0.003, int seed = 17)
    {
        Vocab = new LmVocabulary(examples.Select(e => e.Text));
        Model = new MicroTransformer(Vocab.Size, dModel, hidden, seed);

        var sequences = new List<int[]>();
        foreach (var (text, label) in examples)
        {
            var words = Vocab.Encode(text);
            var seq = new int[words.Length + 2];
            seq[0] = label == "pos" ? Vocab.PosId : Vocab.NegId;
            Array.Copy(words, 0, seq, 1, words.Length);
            seq[^1] = Vocab.EndId;
            sequences.Add(seq);
        }
        FinalLoss = Model.Train(sequences, epochs, learningRate, seed);
        _rng = new Random(seed);
    }

    public LmVocabulary Vocab { get; }
    public MicroTransformer Model { get; }
    public double FinalLoss { get; }

    public GenerationResult Generate(string sentiment, double temperature, int maxWords)
    {
        temperature = Math.Clamp(temperature, 0.2, 2.0);
        maxWords = Math.Clamp(maxWords, 1, MaxWordsCap);

        var context = new List<int> { sentiment == "negative" ? Vocab.NegId : Vocab.PosId };
        var tokens = new List<GeneratedToken>();
        for (int n = 0; n < maxWords; n++)
        {
            var probs = NextProbs(context.ToArray(), temperature, allowEnd: true);
            int id = SampleFrom(probs);
            if (id == Vocab.EndId) break;
            tokens.Add(new GeneratedToken(Vocab.Decode(id), probs[id]));
            context.Add(id);
        }
        return new GenerationResult(string.Join(" ", tokens.Select(t => t.Word)), tokens);
    }

    public IReadOnlyList<GeneratedToken> Suggest(string text, int count = 3)
    {
        var encoded = Vocab.Encode(text ?? "");
        int[] context = encoded.Length == 0 ? new[] { Vocab.PosId } : encoded;
        var probs = NextProbs(context, temperature: 1.0, allowEnd: false);
        return probs
            .Select((p, id) => new GeneratedToken(Vocab.Decode(id), p))
            .Where(t => !t.Word.StartsWith('<'))
            .OrderByDescending(t => t.Prob)
            .Take(count)
            .ToList();
    }

    private double[] NextProbs(int[] context, double temperature, bool allowEnd)
    {
        var logits = Model.LastLogits(context);
        logits[Vocab.PosId] = double.NegativeInfinity;
        logits[Vocab.NegId] = double.NegativeInfinity;
        logits[Vocab.UnkId] = double.NegativeInfinity;
        if (!allowEnd) logits[Vocab.EndId] = double.NegativeInfinity;

        double max = logits.Max();
        var probs = new double[logits.Length];
        double sum = 0;
        for (int v = 0; v < probs.Length; v++)
        {
            probs[v] = double.IsNegativeInfinity(logits[v]) ? 0 : Math.Exp((logits[v] - max) / temperature);
            sum += probs[v];
        }
        for (int v = 0; v < probs.Length; v++) probs[v] /= sum;
        return probs;
    }

    private int SampleFrom(double[] probs)
    {
        double r;
        lock (_rngLock) r = _rng.NextDouble();
        double cum = 0;
        for (int v = 0; v < probs.Length; v++)
        {
            cum += probs[v];
            if (r < cum) return v;
        }
        return probs.Length - 1;
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `dotnet test --filter "FullyQualifiedName~LanguageModelServiceTests"`
Expected: 5 passed (allow ~10–30 s — the shared service trains once). Then run the full suite: `dotnet test` — everything passes.

- [ ] **Step 5: Commit**

```
git add src/Services/LanguageModelService.cs tests/LanguageModelServiceTests.cs
git -c user.email="8763989+marvingarcia25@users.noreply.github.com" commit -m "feat: language model service — sentiment-prompted generation and suggestions"  (append the Co-Authored-By line)
```

---

### Task 6: API endpoints

**Files:**
- Modify: `src/Program.cs`
- Modify: `tests/day17_ReflectionClassifier.Tests.csproj` (add one package)
- Test: `tests/LmApiTests.cs`

- [ ] **Step 1: Add the test package**

In `tests/day17_ReflectionClassifier.Tests.csproj`, add to the existing `<ItemGroup>` with package references:

```xml
    <PackageReference Include="Microsoft.AspNetCore.Mvc.Testing" Version="8.0.6" />
```

- [ ] **Step 2: Write the failing tests**

Create `tests/LmApiTests.cs`:

```csharp
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Xunit;

namespace Day17ReflectionClassifier.Tests;

public class LmApiTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly WebApplicationFactory<Program> _factory;

    public LmApiTests(WebApplicationFactory<Program> factory) => _factory = factory;

    [Fact]
    public async Task Generate_returns_text_and_tokens()
    {
        using var client = _factory.CreateClient();
        var res = await client.PostAsJsonAsync("/generate", new { sentiment = "positive" });
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var body = await res.Content.ReadFromJsonAsync<JsonElement>();
        Assert.True(body.TryGetProperty("text", out _));
        Assert.Equal(JsonValueKind.Array, body.GetProperty("tokens").ValueKind);
    }

    [Fact]
    public async Task Generate_rejects_invalid_sentiment()
    {
        using var client = _factory.CreateClient();
        var res = await client.PostAsJsonAsync("/generate", new { sentiment = "confused" });
        Assert.Equal(HttpStatusCode.BadRequest, res.StatusCode);
    }

    [Fact]
    public async Task Suggest_returns_three_suggestions()
    {
        using var client = _factory.CreateClient();
        var res = await client.PostAsJsonAsync("/suggest", new { text = "i feel" });
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var body = await res.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal(3, body.GetProperty("suggestions").GetArrayLength());
    }

    [Fact]
    public async Task Suggest_accepts_empty_text()
    {
        using var client = _factory.CreateClient();
        var res = await client.PostAsJsonAsync("/suggest", new { text = "" });
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
    }
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `dotnet test --filter "FullyQualifiedName~LmApiTests"`
Expected: 4 failures — 404 (endpoints don't exist yet).

- [ ] **Step 4: Wire up the endpoints**

In `src/Program.cs`, after the existing `builder.Services.AddSingleton(new NaiveBayesClassifier(TrainingData.All));` line, add:

```csharp
// Train the micro-GPT once at startup, same pattern as the classifier.
var lmTimer = System.Diagnostics.Stopwatch.StartNew();
var languageModel = new LanguageModelService(TrainingData.All);
lmTimer.Stop();
Console.WriteLine(
    $"micro-GPT trained: vocab={languageModel.Vocab.Size}, final loss={languageModel.FinalLoss:F3}, {lmTimer.ElapsedMilliseconds} ms");
builder.Services.AddSingleton(languageModel);
```

After the existing `app.MapPost("/classify", …)` block, add:

```csharp
app.MapPost("/generate", (GenerateRequest req, LanguageModelService lm) =>
{
    if (req.Sentiment != "positive" && req.Sentiment != "negative")
        return Results.BadRequest(new { error = "Field 'sentiment' must be 'positive' or 'negative'." });

    var g = lm.Generate(req.Sentiment, req.Temperature ?? 1.0, req.MaxWords ?? 20);
    return Results.Ok(new
    {
        text = g.Text,
        tokens = g.Tokens.Select(t => new { word = t.Word, prob = Math.Round(t.Prob, 4) })
    });
});

app.MapPost("/suggest", (SuggestRequest req, LanguageModelService lm) =>
    Results.Ok(new
    {
        suggestions = lm.Suggest(req.Text ?? "")
            .Select(t => new { word = t.Word, prob = Math.Round(t.Prob, 4) })
    }));
```

At the bottom of the file, next to the existing `ClassifyRequest` record, add:

```csharp
public sealed record GenerateRequest(string? Sentiment, double? Temperature, int? MaxWords);
public sealed record SuggestRequest(string? Text);
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `dotnet test`
Expected: all tests pass (the API fixture trains the model once at startup; allow extra seconds).

- [ ] **Step 6: Commit**

```
git add src/Program.cs tests/LmApiTests.cs tests/day17_ReflectionClassifier.Tests.csproj
git -c user.email="8763989+marvingarcia25@users.noreply.github.com" commit -m "feat: /generate and /suggest endpoints"  (append the Co-Authored-By line)
```

---

### Task 7: UI — generate card + autocomplete chips

**Files:**
- Modify: `src/wwwroot/index.html`

No automated test for this task; Task 9 verifies it in a browser. Keep the existing page's styling system (CSS custom properties, Space Mono labels, `.card` / `.chip` / `.run` classes).

- [ ] **Step 1: Add CSS**

In the `<style>` block of `src/wwwroot/index.html`, after the `.note` rules, add:

```css
  .genbar{display:flex; gap:10px; align-items:stretch}
  .toggle{display:flex; border:1px solid var(--line); border-radius:10px; overflow:hidden}
  .toggle .opt{border:none; background:transparent; color:var(--muted); font-family:"Space Mono",monospace; font-size:13px; padding:0 16px; cursor:pointer; transition:.15s}
  .toggle .opt.active.pos{background:var(--pos-dim); color:var(--pos)}
  .toggle .opt.active.neg{background:var(--neg-dim); color:var(--neg)}
  .run.gen{margin-top:0; flex:1}
  .gen-out{margin-top:16px; min-height:28px; font-size:17px; line-height:1.9}
  .gentok{border-radius:5px; padding:2px 5px; margin-right:2px}
  .suggest-label{font-family:"Space Mono",monospace; font-size:11px; color:var(--muted); margin-top:10px}
```

- [ ] **Step 2: Add markup**

Inside the existing classifier card, directly after the `<textarea id="entry" …></textarea>` line, add the autocomplete strip:

```html
    <div class="suggest-label">micro-GPT suggests:</div>
    <div class="chips" id="suggest"></div>
```

After the entire classifier `<div class="card">…</div>` block (after its closing `</div>`), add a second card:

```html
  <div class="card">
    <label class="field-label">Generate a reflection — micro-GPT</label>
    <div class="genbar">
      <div class="toggle" id="toggle">
        <button type="button" class="opt pos active" data-s="positive">positive</button>
        <button type="button" class="opt neg" data-s="negative">negative</button>
      </div>
      <button class="run gen" id="gen">Generate</button>
    </div>
    <div class="gen-out" id="genout" aria-live="polite"></div>
    <div class="chips"><button class="chip" id="classifyGen" hidden>classify this ↑</button></div>
    <div class="note" id="gennote"></div>
  </div>
```

- [ ] **Step 3: Add JS**

At the end of the existing `<script>` block (after the `$("entry").addEventListener…` line), add:

```javascript
// ---- micro-GPT: generation ----
let sentiment = "positive";
document.querySelectorAll("#toggle .opt").forEach(btn=>{
  btn.onclick = ()=>{
    sentiment = btn.dataset.s;
    document.querySelectorAll("#toggle .opt").forEach(o=>o.classList.toggle("active", o===btn));
  };
});

async function generate(){
  const btn = $("gen");
  btn.disabled = true; btn.textContent = "Sampling…";
  $("gennote").textContent = "";
  try{
    const res = await fetch("/generate", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ sentiment })
    });
    if(!res.ok) throw new Error("Request failed (" + res.status + ")");
    const data = await res.json();
    const out = $("genout"); out.innerHTML = "";
    const color = sentiment === "positive" ? "79,209,165" : "255,107,129";
    (data.tokens || []).forEach((t,i)=>{
      const span = document.createElement("span");
      span.className = "gentok";
      span.textContent = t.word;
      span.title = "sampled at p=" + t.prob;
      span.style.background = `rgba(${color},${(0.06 + 0.30*t.prob).toFixed(3)})`;
      span.style.opacity = "0";
      out.appendChild(span);
      setTimeout(()=>{ span.style.transition = "opacity .25s"; span.style.opacity = "1"; }, i*90);
    });
    $("classifyGen").hidden = !data.text;
    $("classifyGen").dataset.text = data.text || "";
    if(!data.text) $("gennote").textContent = "The model went straight to <end> — try again.";
  }catch(err){
    $("gennote").textContent = err.message; $("gennote").classList.add("error");
  }finally{
    btn.disabled = false; btn.textContent = "Generate";
  }
}
$("gen").onclick = generate;
$("classifyGen").onclick = ()=>{ $("entry").value = $("classifyGen").dataset.text; run(); };

// ---- micro-GPT: autocomplete ----
let suggestTimer = null;
async function refreshSuggestions(){
  try{
    const res = await fetch("/suggest", {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ text: $("entry").value })
    });
    if(!res.ok) return;
    const data = await res.json();
    const box = $("suggest"); box.innerHTML = "";
    (data.suggestions || []).forEach(s=>{
      const b = document.createElement("button");
      b.className = "chip"; b.type = "button";
      b.textContent = s.word + " · " + (s.prob*100).toFixed(0) + "%";
      b.onclick = ()=>{
        const v = $("entry").value;
        $("entry").value = v + (v && !v.endsWith(" ") ? " " : "") + s.word;
        $("entry").focus();
        refreshSuggestions();
      };
      box.appendChild(b);
    });
  }catch{ /* suggestions are decorative — stay quiet on failure */ }
}
$("entry").addEventListener("input", ()=>{
  clearTimeout(suggestTimer);
  suggestTimer = setTimeout(refreshSuggestions, 250);
});
refreshSuggestions();
```

- [ ] **Step 4: Update the "How this works" panel**

In the `<details>` element, after the existing second `<p>…</p>`, add:

```html
    <p><strong>The generator is a micro-GPT.</strong> A single-head transformer — token and positional embeddings, causal self-attention, a feed-forward layer, ~25k parameters — written from scratch in C# and trained at startup on the same 150 reflections. Generation starts from a <code>&lt;pos&gt;</code> or <code>&lt;neg&gt;</code> prompt token and samples word by word; each word's shading shows the probability it was sampled at. Same mechanism as an LLM, roughly ten million times smaller — which is why it rambles.</p>
```

- [ ] **Step 5: Verify the build still passes and commit**

Run: `dotnet build` — expected: success (static file change, but confirms nothing else broke).

```
git add src/wwwroot/index.html
git -c user.email="8763989+marvingarcia25@users.noreply.github.com" commit -m "feat: generation card and autocomplete chips in UI"  (append the Co-Authored-By line)
```

---

### Task 8: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Edit README.md**

1. Replace the intro paragraph line (starts with `` `day17_ReflectionClassifier` reads a short daily reflection ``) with:

```markdown
`day17_ReflectionClassifier` reads a short daily reflection ("how was your day?") and labels it **positive** or **negative** — then tells you which words drove the decision. The model is a multinomial Naive Bayes classifier written in plain C#. Alongside it lives a **micro-GPT**: a single-head transformer language model, also written from scratch, that *generates* reflections and autocompletes yours as you type. No ML.NET, no external model, no API. Both train in memory at startup from a labelled dataset baked into the build.
```

2. In the "What it does" list, add two bullets after the existing ones:

```markdown
- **Micro-GPT generator** — a decoder-only transformer (token + positional embeddings, one causal self-attention head, FFN, ~25k parameters) with hand-written backprop, verified against finite differences. `POST /generate` writes a new reflection; steering it positive or negative works by *prompting* with a `<pos>`/`<neg>` token.
- **Autocomplete** — `POST /suggest` returns the model's top next words while you type.
```

3. After the "Is this an LLM?"-relevant context — i.e., directly after the "Why this one" section — add a new section:

```markdown
## Is it an LLM?

The classifier is not — it's classical ML, a bag-of-words counter. The generator now genuinely is a (tiny) language model: the same decoder-only transformer mechanism as GPT — embeddings, causal self-attention, next-token prediction — at roughly one ten-millionth the size, trained on 150 sentences instead of the internet. That's why it rambles. The point is the mechanism, not the eloquence.
```

4. In the "Layout" block, update the `src/` listing to:

```
src/
  Program.cs                       minimal API (POST /classify, /generate, /suggest) + static UI
  wwwroot/index.html               the UI — classify, generate, autocomplete
  Services/Tokenizer.cs            text -> tokens, with negation (classifier)
  Services/NaiveBayesClassifier.cs the classifier, from scratch
  Services/LmVocabulary.cs         word-level vocab for the language model
  Services/Tensor.cs               matrix parameter + gradient + Adam state
  Services/MicroTransformer.cs     single-head transformer, hand-written backprop
  Services/LanguageModelService.cs training at startup, sampling, suggestions
  Data/TrainingData.cs             150 labelled reflections
tests/
  *.cs                             xUnit suites (classifier, vocab, model, service, API)
```

5. In "What I'd reach for next", remove the line about bigrams ("Add bigrams so phrases like "not bad" are learned directly.") and add:

```markdown
- More training data — the transformer is mechanism-complete but data-starved.
- Byte-pair encoding and multi-head attention, the next two rungs toward a real GPT.
```

6. Update the "Stack" line to:

```markdown
C# / .NET 8, ASP.NET Core minimal API, xUnit. No machine-learning libraries — that's the point, twice over now.
```

- [ ] **Step 2: Commit**

```
git add README.md
git -c user.email="8763989+marvingarcia25@users.noreply.github.com" commit -m "docs: README — micro-GPT generator, honest is-it-an-LLM section"  (append the Co-Authored-By line)
```

---

### Task 9: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `dotnet test`
Expected: all tests pass, zero failures.

- [ ] **Step 2: Startup time check**

Run: `dotnet run --project src --urls http://localhost:5000` (background) and read the console line `micro-GPT trained: vocab=…, final loss=…, … ms`.
Expected: training under ~10,000 ms and final loss well below 6.0 (ln(vocab) ≈ 6.07 is the random-guess baseline). If training exceeds 10 s, reduce `dModel` to 16 and `hidden` to 32 in the `LanguageModelService` constructor defaults and re-run tests.

- [ ] **Step 3: Smoke-test the endpoints**

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:5000/generate -ContentType "application/json" -Body '{"sentiment":"positive"}'
Invoke-RestMethod -Method Post -Uri http://localhost:5000/generate -ContentType "application/json" -Body '{"sentiment":"negative"}'
Invoke-RestMethod -Method Post -Uri http://localhost:5000/suggest  -ContentType "application/json" -Body '{"text":"i feel"}'
```

Expected: `/generate` returns `text` + `tokens` made of real words; positive and negative outputs read differently; `/suggest` returns 3 plausible next words (e.g. after "i feel" something like "proud", "strong", "tired").

- [ ] **Step 4: UI check**

Open http://localhost:5000 — the Generate card renders, the toggle switches color, Generate produces shaded words, "classify this ↑" feeds the result into the classifier, and suggestion chips appear under the textarea while typing.

- [ ] **Step 5: Stop the server**

Stop the background `dotnet run`. Do not push or deploy — that happens after final review.
