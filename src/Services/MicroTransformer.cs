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

    public double[] NextProbabilities(int[] tokens, double temperature = 1.0)
    {
        temperature = Math.Clamp(temperature, 0.2, 2.0);
        var logits = LastLogits(tokens);
        var probs = new double[logits.Length];
        double max = logits.Max();
        double sum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = Math.Exp((logits[i] - max) / temperature);
            sum += probs[i];
        }
        for (int i = 0; i < probs.Length; i++) probs[i] /= sum;
        return probs;
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

    public double Train(IReadOnlyList<int[]> sequences, int epochs, double learningRate, int seed)
    {
        var rng = new Random(seed);
        var examples = new List<(int[] Context, int Target)>();
        foreach (var seq in sequences)
        {
            for (int i = 1; i < seq.Length; i++)
            {
                int start = Math.Max(0, i - ContextWindow);
                examples.Add((seq[start..i], seq[i]));
            }
        }

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            foreach (var ex in examples.OrderBy(_ => rng.Next()))
                TrainExample(ex.Context, ex.Target, learningRate);
        }

        return examples.Count == 0 ? 0 : examples.Average(ex => -Math.Log(NextProbabilities(ex.Context)[ex.Target] + 1e-12));
    }

    private void TrainExample(int[] context, int target, double learningRate)
    {
        var c = Forward(context);
        var probs = c.Probs[c.L - 1];
        var grad = new double[_vocab];
        Array.Copy(probs, grad, _vocab);
        grad[target] -= 1.0;

        var x2 = c.X2[c.L - 1];
        for (int e = 0; e < _d; e++)
            for (int v = 0; v < _vocab; v++)
                _wout[e, v] -= learningRate * x2[e] * grad[v];
        for (int v = 0; v < _vocab; v++)
            _bout[0, v] -= learningRate * grad[v];

        var gx2 = new double[_d];
        for (int e = 0; e < _d; e++)
        {
            double s = 0;
            for (int v = 0; v < _vocab; v++) s += _wout[e, v] * grad[v];
            gx2[e] = s;
        }

        int last = c.L - 1;
        for (int e = 0; e < _d; e++)
        {
            _lg2[0, e] -= learningRate * gx2[e] * c.U2Hat[last][e];
            _lb2[0, e] -= learningRate * gx2[e];
        }

        for (int k = 0; k < _h; k++)
        {
            double gact = 0;
            for (int e = 0; e < _d; e++)
            {
                _w2[k, e] -= learningRate * c.Act[last][k] * gx2[e];
                gact += _w2[k, e] * gx2[e];
            }
            double gz = gact * (1 - c.Act[last][k] * c.Act[last][k]);
            for (int e = 0; e < _d; e++)
                _w1[e, k] -= learningRate * c.X1[last][e] * gz;
            _b1[0, k] -= learningRate * gz;
        }

        for (int e = 0; e < _d; e++)
        {
            _lg1[0, e] -= learningRate * gx2[e] * c.U1Hat[last][e] * 0.1;
            _lb1[0, e] -= learningRate * gx2[e] * 0.1;
            int token = c.Tokens[last];
            _emb[token, e] -= learningRate * gx2[e] * 0.05;
            _pos[last, e] -= learningRate * gx2[e] * 0.05;
        }
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
