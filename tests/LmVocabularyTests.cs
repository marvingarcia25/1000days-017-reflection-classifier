using Day17ReflectionClassifier.Services;
using Xunit;

namespace Day17ReflectionClassifier.Tests;

public class LmVocabularyTests
{
    [Fact]
    public void Specials_come_first_and_have_stable_ids()
    {
        var v = new LmVocabulary(new[] { "a happy day" });
        Assert.Equal(0, v.PosId);
        Assert.Equal(1, v.NegId);
        Assert.Equal(2, v.EndId);
        Assert.Equal(3, v.UnkId);
        Assert.Equal("<pos>", v.Decode(v.PosId));
    }

    [Fact]
    public void Encode_decode_round_trips_known_words()
    {
        var v = new LmVocabulary(new[] { "I felt Great today!" });
        var ids = v.Encode("great TODAY");
        Assert.Equal(2, ids.Length);
        Assert.Equal("great", v.Decode(ids[0]));
        Assert.Equal("today", v.Decode(ids[1]));
    }

    [Fact]
    public void Unknown_words_map_to_unk()
    {
        var v = new LmVocabulary(new[] { "a happy day" });
        var ids = v.Encode("zebra happy");
        Assert.Equal(v.UnkId, ids[0]);
        Assert.NotEqual(v.UnkId, ids[1]);
    }

    [Fact]
    public void Punctuation_is_stripped_and_apostrophes_kept()
    {
        var v = new LmVocabulary(new[] { "i didn't stop, really." });
        var ids = v.Encode("didn't really");
        Assert.Equal("didn't", v.Decode(ids[0]));
        Assert.Equal("really", v.Decode(ids[1]));
    }

    [Fact]
    public void Size_counts_specials_plus_unique_words()
    {
        var v = new LmVocabulary(new[] { "good day", "good night" });
        Assert.Equal(4 + 3, v.Size); // specials + good, day, night
    }
}
