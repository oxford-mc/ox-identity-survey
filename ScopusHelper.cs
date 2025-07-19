using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using Porter2Stemmer;

public class ScopusHelper
{
    private const string ApiKey = "81374fafb11e543fc9622b36194d3947";
    private const string InstToken = "fac75069f3e0572c8e1f0de37e75b694";
    private const string BaseUrl = "https://api.elsevier.com/content/search/scopus";

    private readonly HttpClient _httpClient;
    private readonly EnglishPorter2Stemmer _stemmer = new();
    private readonly ArticleManager _articleManager;

    public ScopusHelper(ArticleManager articleManager)
    {
        _articleManager = articleManager;

        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("X-ELS-APIKey", ApiKey);
        _httpClient.DefaultRequestHeaders.Add("X-ELS-InstToken", InstToken);
        _httpClient.DefaultRequestHeaders.Add("Accept", "application/json");
        _httpClient.DefaultRequestHeaders.Add("X-ELS-ResourceVersion", "XOCS"); // Required for cursor support
    }

    public async Task<Dictionary<int, List<string>>> FetchAbstractsByYear(string query, int startYear, int endYear)
    {
        var abstractsByYear = new Dictionary<int, List<string>>();

        for (int year = endYear; year >= startYear; year--)
        {
            Console.WriteLine($"Fetching abstracts for {year}...");

            var yearAbstracts = new List<string>();
            string encodedQuery = Uri.EscapeDataString($"TITLE-ABS-KEY({query}) AND PUBYEAR IS {year}");
            string cursor = "*";
            int count = 25;

            bool hasMore = true;
            while (hasMore)
            {
                string url = $"{BaseUrl}?query={encodedQuery}&cursor={Uri.EscapeDataString(cursor)}&count={count}";
                try
                {
                    var response = await _httpClient.GetStringAsync(url);
                    var json = JObject.Parse(response);
                    var results = json["search-results"];
                    var entries = results?["entry"];

                    if (entries == null || !entries.Any())
                        break;

                    foreach (var entry in entries)
                    {
                        string scopusId = entry["dc:identifier"]?.ToString()?.Replace("SCOPUS_ID:", "");
                        if (string.IsNullOrEmpty(scopusId)) continue;

                        if (!_articleManager.Exists(scopusId))
                        {
                            JObject detail = await GetAbstractDetailAsJson(scopusId);
                            if (detail != null)
                                await _articleManager.AddAsync(detail, scopusId);
                        }

                        if (_articleManager.TryGet(scopusId, out JObject cached))
                        {
                            var abstractText = cached["abstracts-retrieval-response"]?["coredata"]?["dc:description"]?.ToString();
                            if (!string.IsNullOrWhiteSpace(abstractText))
                                yearAbstracts.Add(abstractText);
                        }
                    }

                    // Get next cursor value
                    string nextCursor = results["cursor"]?["@next"]?.ToString();
                    if (!string.IsNullOrWhiteSpace(nextCursor))
                    {
                        cursor = nextCursor;
                    }
                    else
                    {
                        hasMore = false;
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error fetching with cursor={cursor}: {ex.Message}");
                    break;
                }

                _articleManager.Save();
            }

            abstractsByYear[year] = yearAbstracts;
            Console.WriteLine($"Total abstracts for {year}: {yearAbstracts.Count}");
        }

        return abstractsByYear;
    }

    public async Task<JObject> GetAbstractDetailAsJson(string scopusId)
    {
        string detailUrl = $"https://api.elsevier.com/content/abstract/scopus_id/{scopusId}";
        try
        {
            var detailResponse = await _httpClient.GetStringAsync(detailUrl);
            await Task.Delay(100); // avoid API rate limits
            return JObject.Parse(detailResponse);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to fetch detail for {scopusId}: {ex.Message}");
            return null;
        }
    }

    public Dictionary<string, int> ExtractTopTerms(List<string> abstracts, int topN = 30)
    {
        var stopWords = new HashSet<string>(System.IO.File.ReadAllLines("stopwords.txt"));
        var termFrequency = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var text in abstracts)
        {
            var words = Regex.Matches(text.ToLower(), @"\b[a-z]{4,}\b")
                .Select(m => m.Value)
                .Where(w => !stopWords.Contains(w));

            foreach (var word in words)
            {
                var stemmed = _stemmer.Stem(word).Value;
                if (!termFrequency.ContainsKey(stemmed))
                    termFrequency[stemmed] = 0;
                termFrequency[stemmed]++;
            }
        }

        return termFrequency
            .OrderByDescending(kvp => kvp.Value)
            .Take(topN)
            .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
    }

    public Dictionary<string, Dictionary<int, int>> LongitudinalTermAnalysis(Dictionary<int, List<string>> abstractsByYear, List<string> topTerms)
    {
        var longitudinal = new Dictionary<string, Dictionary<int, int>>();

        foreach (var term in topTerms)
            longitudinal[term] = new Dictionary<int, int>();

        foreach (var (year, texts) in abstractsByYear)
        {
            var termCounts = new Dictionary<string, int>();

            foreach (var text in texts)
            {
                var words = Regex.Matches(text.ToLower(), @"\b[a-z]{4,}\b")
                    .Select(m => _stemmer.Stem(m.Value).Value);

                foreach (var word in words)
                {
                    if (topTerms.Contains(word))
                    {
                        if (!termCounts.ContainsKey(word)) termCounts[word] = 0;
                        termCounts[word]++;
                    }
                }
            }

            foreach (var term in topTerms)
                longitudinal[term][year] = termCounts.TryGetValue(term, out int count) ? count : 0;
        }

        return longitudinal;
    }
}
