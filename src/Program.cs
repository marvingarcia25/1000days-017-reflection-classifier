using Day17ReflectionClassifier.Data;
using Day17ReflectionClassifier.Services;

var builder = WebApplication.CreateBuilder(args);

// Train once at startup; the model is immutable and safe to share.
builder.Services.AddSingleton(new NaiveBayesClassifier(TrainingData.All));
var lmTimer = System.Diagnostics.Stopwatch.StartNew();
var languageModel = new LanguageModelService(TrainingData.All);
lmTimer.Stop();
Console.WriteLine($"micro-GPT trained: vocab={languageModel.Vocab.Size}, final loss={languageModel.FinalLoss:F3}, {lmTimer.ElapsedMilliseconds} ms");
builder.Services.AddSingleton(languageModel);

var app = builder.Build();

// Serve the UI from wwwroot (index.html at "/").
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPost("/classify", (ClassifyRequest req, NaiveBayesClassifier model) =>
{
    if (string.IsNullOrWhiteSpace(req.Text))
        return Results.BadRequest(new { error = "Send a non-empty 'text' field." });

    var p = model.Classify(req.Text);
    return Results.Ok(new
    {
        label = p.Label == "pos" ? "positive" : "negative",
        confidence = Math.Round(p.Confidence, 4),
        // every word's lean (positive value => pushes toward "positive"), strongest first
        leanings = p.Leanings
            .OrderByDescending(l => Math.Abs(l.Lean))
            .Select(l => new { word = l.Token, lean = Math.Round(l.Lean, 3) })
    });
});

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

app.Run();

public sealed record ClassifyRequest(string Text);
public sealed record GenerateRequest(string? Sentiment, double? Temperature, int? MaxWords);
public sealed record SuggestRequest(string? Text);

// Exposed so the test project can drive the same entry point if needed.
public partial class Program { }
