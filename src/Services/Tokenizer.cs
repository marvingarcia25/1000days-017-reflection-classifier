namespace Day17ReflectionClassifier.Services;

/// <summary>
/// Turns raw text into tokens. Two deliberate choices:
///   1. Keep only letters; lowercase everything. Punctuation and digits drop out.
///   2. Handle negation with a one-word scope: a word immediately after a
///      negator ("not", "never", ...) is prefixed with "not_", so "not happy"
///      becomes ["not", "not_happy"] and never reads as positive.
/// </summary>
public static class Tokenizer
{
    private static readonly HashSet<string> Negators = new()
    {
        "not", "no", "never", "cant", "cannot", "dont", "didnt", "wont", "isnt", "wasnt"
    };

    public static List<string> Tokenize(string text)
    {
        var sb = new System.Text.StringBuilder(text.Length);
        foreach (var c in text.ToLowerInvariant())
            sb.Append(c is >= 'a' and <= 'z' ? c : ' ');

        var words = sb.ToString().Split(' ', StringSplitOptions.RemoveEmptyEntries);

        var tokens = new List<string>(words.Length);
        var negate = false;
        foreach (var w in words)
        {
            if (Negators.Contains(w))
            {
                negate = true;
                tokens.Add(w);
                continue;
            }
            tokens.Add(negate ? "not_" + w : w);
            negate = false; // negation scope is a single following word
        }
        return tokens;
    }
}
