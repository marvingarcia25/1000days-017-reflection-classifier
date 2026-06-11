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
