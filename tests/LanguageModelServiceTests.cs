using Day17ReflectionClassifier.Services;
using Xunit;

namespace Day17ReflectionClassifier.Tests;

public class LanguageModelServiceTests
{
    private static readonly (string Text, string Label)[] Tiny =
    {
        ("happy calm day", "pos"),
        ("proud bright morning", "pos"),
        ("sad tired day", "neg"),
        ("drained dark morning", "neg"),
    };

    [Fact]
    public void Generate_returns_known_words_within_limit()
    {
        var lm = new LanguageModelService(Tiny, dModel: 8, hidden: 12, epochs: 2, seed: 4);
        var result = lm.Generate("positive", 1.0, 5);
        Assert.True(result.Tokens.Count <= 5);
        Assert.All(result.Tokens, t => Assert.False(t.Word.StartsWith('<')));
    }

    [Fact]
    public void Suggest_returns_requested_count()
    {
        var lm = new LanguageModelService(Tiny, dModel: 8, hidden: 12, epochs: 2, seed: 4);
        var suggestions = lm.Suggest("happy", 3);
        Assert.Equal(3, suggestions.Count);
        Assert.All(suggestions, t => Assert.InRange(t.Prob, 0, 1));
    }
}
