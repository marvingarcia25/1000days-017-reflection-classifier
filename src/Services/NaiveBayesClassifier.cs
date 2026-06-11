namespace Day17ReflectionClassifier.Services;

/// <summary>The model's verdict, plus the per-word reasoning behind it.</summary>
public sealed record Prediction(
    string Label,
    double Confidence,
    double ProbabilityPositive,
    IReadOnlyList<WordLean> Leanings);

/// <summary>How far one token pushed the decision. Positive = toward "pos".</summary>
public sealed record WordLean(string Token, double Lean);

/// <summary>
/// Multinomial Naive Bayes, written from scratch — no ML libraries.
///
/// Training counts how often each token appears per class. Classifying a new
/// text sums log P(class) + Σ log P(token | class) and picks the larger.
/// Add-one (Laplace) smoothing keeps unseen tokens from zeroing a class out.
/// Confidence is the softmax of the two class log-scores.
/// </summary>
public sealed class NaiveBayesClassifier
{
    private readonly Dictionary<string, int> _posCounts = new();
    private readonly Dictionary<string, int> _negCounts = new();
    private readonly HashSet<string> _vocab = new();
    private int _posTotal;
    private int _negTotal;
    private double _logPriorPos;
    private double _logPriorNeg;

    public int VocabularySize => _vocab.Count;

    public NaiveBayesClassifier(IEnumerable<(string Text, string Label)> data)
    {
        int posDocs = 0, negDocs = 0;
        foreach (var (text, label) in data)
        {
            var isPos = label == "pos";
            if (isPos) posDocs++; else negDocs++;

            foreach (var tok in Tokenizer.Tokenize(text))
            {
                _vocab.Add(tok);
                var counts = isPos ? _posCounts : _negCounts;
                counts[tok] = counts.GetValueOrDefault(tok) + 1;
                if (isPos) _posTotal++; else _negTotal++;
            }
        }

        if (posDocs == 0 || negDocs == 0)
            throw new ArgumentException("Training data must contain both classes.");

        var n = (double)(posDocs + negDocs);
        _logPriorPos = Math.Log(posDocs / n);
        _logPriorNeg = Math.Log(negDocs / n);
    }

    private double LogLikelihood(string token, bool pos)
    {
        var counts = pos ? _posCounts : _negCounts;
        var total = pos ? _posTotal : _negTotal;
        return Math.Log((counts.GetValueOrDefault(token) + 1.0) / (total + _vocab.Count));
    }

    public Prediction Classify(string text)
    {
        var scorePos = _logPriorPos;
        var scoreNeg = _logPriorNeg;
        var leanings = new List<WordLean>();

        foreach (var tok in Tokenizer.Tokenize(text))
        {
            var lp = LogLikelihood(tok, pos: true);
            var ln = LogLikelihood(tok, pos: false);
            scorePos += lp;
            scoreNeg += ln;
            leanings.Add(new WordLean(tok, lp - ln));
        }

        // softmax over the two log-scores -> probability of "pos"
        var max = Math.Max(scorePos, scoreNeg);
        var ep = Math.Exp(scorePos - max);
        var en = Math.Exp(scoreNeg - max);
        var pPos = ep / (ep + en);

        var label = scorePos >= scoreNeg ? "pos" : "neg";
        return new Prediction(label, Math.Max(pPos, 1 - pPos), pPos, leanings);
    }
}
