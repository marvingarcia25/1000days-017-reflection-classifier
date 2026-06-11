using Day17ReflectionClassifier.Data;

namespace Day17ReflectionClassifier.Services;

public sealed record GeneratedToken(string Word, double Prob);
public sealed record GenerationResult(string Text, IReadOnlyList<GeneratedToken> Tokens);

public sealed class LanguageModelService
{
    private const int MaxWordsCap = 30;
    private readonly Random _rng;
    private readonly object _rngLock = new();

    public LanguageModelService(
        IReadOnlyList<(string Text, string Label)>? data = null,
        int dModel = 16,
        int hidden = 32,
        int epochs = 3,
        double learningRate = 0.006,
        int seed = 17)
    {
        data ??= TrainingData.All;
        Vocab = new LmVocabulary(data.Select(d => d.Text));
        Model = new MicroTransformer(Vocab.Size, dModel, hidden, seed);

        var sequences = new List<int[]>();
        foreach (var (text, label) in data)
        {
            var words = Vocab.Encode(text);
            var seq = new int[words.Length + 2];
            seq[0] = label == "neg" ? Vocab.NegId : Vocab.PosId;
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
        for (int i = 0; i < maxWords; i++)
        {
            var probs = NextProbs(context.ToArray(), temperature, allowEnd: true);
            var id = SampleFrom(probs);
            if (id == Vocab.EndId) break;
            tokens.Add(new GeneratedToken(Vocab.Decode(id), probs[id]));
            context.Add(id);
        }

        return new GenerationResult(string.Join(" ", tokens.Select(t => t.Word)), tokens);
    }

    public IReadOnlyList<GeneratedToken> Suggest(string text, int count = 3)
    {
        var encoded = Vocab.Encode(text ?? "");
        var context = encoded.Length == 0 ? new[] { Vocab.PosId } : encoded;
        var probs = NextProbs(context, 1.0, allowEnd: false);
        return probs
            .Select((p, id) => new GeneratedToken(Vocab.Decode(id), p))
            .Where(t => !t.Word.StartsWith('<'))
            .OrderByDescending(t => t.Prob)
            .Take(count)
            .ToList();
    }

    private double[] NextProbs(int[] context, double temperature, bool allowEnd)
    {
        var probs = Model.NextProbabilities(context, temperature);
        probs[Vocab.PosId] = 0;
        probs[Vocab.NegId] = 0;
        probs[Vocab.UnkId] = 0;
        if (!allowEnd) probs[Vocab.EndId] = 0;

        for (int i = 0; i < probs.Length; i++)
            probs[i] = Math.Pow(Math.Max(probs[i], 0), 0.7);

        var sum = probs.Sum();
        if (sum <= 0)
            return Enumerable.Repeat(1.0 / probs.Length, probs.Length).ToArray();
        for (int i = 0; i < probs.Length; i++)
            probs[i] /= sum;
        return probs;
    }

    private int SampleFrom(double[] probs)
    {
        double r;
        lock (_rngLock) r = _rng.NextDouble();
        double cum = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cum += probs[i];
            if (r < cum) return i;
        }
        return probs.Length - 1;
    }
}
