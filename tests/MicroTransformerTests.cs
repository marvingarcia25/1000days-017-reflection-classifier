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
