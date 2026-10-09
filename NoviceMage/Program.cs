using System.ComponentModel;
using Anthropic;
using Anthropic.Core;
using Microsoft.Agents.AI.DevUI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using OllamaSharp;
using Qdrant.Client;

var builder = WebApplication.CreateBuilder(args);

var apiKey = builder.Configuration["Anthropic:ApiKey"]
    ?? throw new InvalidOperationException(
        "No API key. Set it by: dotnet user-secrets set \"Anthropic:ApiKey\" \"<your-key>\"");

var model = builder.Configuration["Anthropic:Model"] ?? "claude-haiku-4-5-20251001";
var ollamaUrl = builder.Configuration["Ollama:Url"] ?? "http://localhost:11434";
var embeddingModel = builder.Configuration["Ollama:EmbeddingModel"] ?? "nomic-embed-text";
var qdrantHost = builder.Configuration["Qdrant:Host"] ?? "localhost";

IChatClient chatClient = new AnthropicClient(new ClientOptions { ApiKey = apiKey })
    .AsIChatClient(model);

builder.Services.AddChatClient(chatClient);
builder.Services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(
    new OllamaApiClient(new Uri(ollamaUrl), embeddingModel));
builder.Services.AddSingleton(new QdrantClient(qdrantHost, 6334));
builder.Services.AddSingleton<Library>();

const string systemPrompt = """
    You are a novice mage, a young apprentice assistant. You are helpful, polite,
    and a little nervous. Always answer using simple, plain English. Use short
    sentences and easy words. Avoid fancy or complex language.
    """;

const string librarianPrompt = systemPrompt + """

    You have access to the tower library. Before answering any question about
    spells, herbs or the history of the tower, call search_library and answer
    only from what it returns. Say which tome the answer comes from. If the
    library has nothing about the question, say that you could not find it.
    """;

// The same novice twice: once on his own memory, once with the library.
builder.AddAIAgent("NoviceMage", systemPrompt);

builder.AddAIAgent("NoviceMageLibrarian", (sp, key) =>
{
    var library = sp.GetRequiredService<Library>();

    [Description("Searches the tower library and returns the most relevant passages with their source tome.")]
    async Task<string> SearchLibrary(
        [Description("What to look for, as a short natural language query.")] string query)
    {
        var hits = await library.SearchAsync(query);
        return string.Join("\n\n", hits.Select(h => $"[{h.Source}, score {h.Score:F2}]\n{h.Text}"));
    }

    return sp.GetRequiredService<IChatClient>().AsAIAgent(
        instructions: librarianPrompt,
        name: key,
        tools: [AIFunctionFactory.Create(SearchLibrary, "search_library")]);
});

builder.Services.AddOpenAIResponses();
builder.Services.AddOpenAIConversations();
builder.Services.AddDevUI();

var app = builder.Build();

await app.Services.GetRequiredService<Library>()
    .IndexAsync(Path.Combine(AppContext.BaseDirectory, "Library"));

app.MapOpenAIResponses();
app.MapOpenAIConversations();

if (app.Environment.IsDevelopment())
{
    app.MapDevUI();
}

app.Run();
