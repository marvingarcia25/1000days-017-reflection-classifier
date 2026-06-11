using Day17ReflectionClassifier.Data;
using Day17ReflectionClassifier.Services;

var builder = WebApplication.CreateBuilder(args);

// Train once at startup; the model is immutable and safe to share.
builder.Services.AddSingleton(new NaiveBayesClassifier(TrainingData.All));

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

app.Run();

public sealed record ClassifyRequest(string Text);

// Exposed so the test project can drive the same entry point if needed.
public partial class Program { }
