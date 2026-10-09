using Microsoft.Extensions.AI;
using Qdrant.Client;
using Qdrant.Client.Grpc;

public sealed class Library(
    QdrantClient qdrant,
    IEmbeddingGenerator<string, Embedding<float>> embeddings)
{
    public const string Collection = "library";

    public async Task IndexAsync(string directory)
    {
        if (await qdrant.CollectionExistsAsync(Collection))
        {
            await qdrant.DeleteCollectionAsync(Collection);
        }

        var chunks = Directory.GetFiles(directory, "*.md")
            .SelectMany(file => Chunk(file))
            .ToList();

        var vectors = await embeddings.GenerateAsync(chunks.Select(c => c.Text));

        await qdrant.CreateCollectionAsync(Collection, new VectorParams
        {
            Size = (ulong)vectors[0].Vector.Length,
            Distance = Distance.Cosine,
        });

        var points = chunks.Select((chunk, i) => new PointStruct
        {
            Id = (ulong)i,
            Vectors = vectors[i].Vector.ToArray(),
            Payload =
            {
                ["source"] = chunk.Source,
                ["text"] = chunk.Text,
            },
        }).ToList();

        await qdrant.UpsertAsync(Collection, points);
    }

    public async Task<IReadOnlyList<(string Source, string Text, float Score)>> SearchAsync(
        string query, int limit = 3)
    {
        var vector = await embeddings.GenerateVectorAsync(query);
        var hits = await qdrant.QueryAsync(Collection, vector.ToArray(), limit: (ulong)limit);

        return [.. hits.Select(h => (
            h.Payload["source"].StringValue,
            h.Payload["text"].StringValue,
            h.Score))];
    }

    // One chunk per "## " section, prefixed with the tome title so the
    // chunk still makes sense when it is read alone.
    private static IEnumerable<(string Source, string Text)> Chunk(string file)
    {
        var source = Path.GetFileNameWithoutExtension(file);
        var text = File.ReadAllText(file);
        var title = text.Split('\n')[0].TrimStart('#', ' ').Trim();

        foreach (var section in text.Split("\n## ", StringSplitOptions.RemoveEmptyEntries).Skip(1))
        {
            yield return (source, $"{title}\n## {section.Trim()}");
        }
    }
}
