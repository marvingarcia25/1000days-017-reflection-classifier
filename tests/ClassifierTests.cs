using Day17ReflectionClassifier.Data;
using Day17ReflectionClassifier.Services;
using Xunit;

namespace Day17ReflectionClassifier.Tests;

public class TokenizerTests
{
    [Fact]
    public void Lowercases_And_Strips_Punctuation_And_Digits()
    {
        var tokens = Tokenizer.Tokenize("Today: I ran 5km!!  Great.");
        Assert.Equal(new[] { "today", "i", "ran", "km", "great" }, tokens);
    }

    [Fact]
    public void Negation_Tags_The_Following_Word()
    {
        var tokens = Tokenizer.Tokenize("i am not happy today");
        Assert.Contains("not_happy", tokens);
        Assert.DoesNotContain("happy", tokens);
    }

    [Fact]
    public void Negation_Scope_Is_A_Single_Word()
    {
        var tokens = Tokenizer.Tokenize("not good day");
        Assert.Contains("not_good", tokens);
        Assert.Contains("day", tokens); // "day" is outside the negation scope
    }
}

public class ClassifierTests
{
    private static NaiveBayesClassifier Model() => new(TrainingData.All);

    [Fact]
    public void Clearly_Positive_Text_Is_Positive()
    {
        var p = Model().Classify("i feel proud grateful and motivated today");
        Assert.Equal("pos", p.Label);
        Assert.True(p.Confidence > 0.7);
    }

    [Fact]
    public void Clearly_Negative_Text_Is_Negative()
    {
        var p = Model().Classify("i feel exhausted hopeless and defeated today");
        Assert.Equal("neg", p.Label);
        Assert.True(p.Confidence > 0.7);
    }

    [Fact]
    public void Negation_Flips_Sentiment()
    {
        var p = Model().Classify("i am not happy with my progress");
        Assert.Equal("neg", p.Label);
    }

    [Fact]
    public void Unseen_Words_Do_Not_Throw_And_Stay_Probabilistic()
    {
        var p = Model().Classify("xylophone qwerty zzz");
        Assert.InRange(p.ProbabilityPositive, 0.0, 1.0);
    }

    [Fact]
    public void Confidence_Is_A_Valid_Probability()
    {
        var p = Model().Classify("a calm productive rewarding morning");
        Assert.InRange(p.Confidence, 0.5, 1.0);
    }

    // Trains on 4/5 of the data and evaluates on the held-out 5th (every index % 5 == 0).
    // This split is identical to the verified Python reference, which scores 28/30.
    [Fact]
    public void Holdout_Accuracy_Meets_Threshold()
    {
        var all = TrainingData.All;
        var train = all.Where((_, i) => i % 5 != 0).ToList();
        var test = all.Where((_, i) => i % 5 == 0).ToList();

        var model = new NaiveBayesClassifier(train);
        var correct = test.Count(item => model.Classify(item.Text).Label == item.Label);
        var accuracy = (double)correct / test.Count;

        Assert.True(accuracy >= 0.85, $"Holdout accuracy {accuracy:P1} fell below 85%.");
    }
}
