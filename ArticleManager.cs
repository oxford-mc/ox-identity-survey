using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

public class ArticleManager
{
    private readonly string _cacheFilePath;
    private readonly Dictionary<string, JObject> _cache;

    public ArticleManager(string cacheFilePath)
    {
        _cacheFilePath = cacheFilePath;
        _cache = new Dictionary<string, JObject>();
        this.Load();
    }

    public bool Exists(string scopusId)
    {
        return _cache.ContainsKey(scopusId);
    }

    public Task AddAsync(JObject detail, string scopusId)
    {
        if (_cache.ContainsKey(scopusId))
        {
            Console.WriteLine($"✅ Cached: {scopusId}");
            return Task.CompletedTask;
        }

        if (detail != null)
        {
            _cache[scopusId] = detail;
            Console.WriteLine($"➕ Added: {scopusId}");
        }
        else
        {
            Console.WriteLine($"⚠️ No data for: {scopusId}");
        }

        return Task.CompletedTask;
    }

    public void Save()
    {
        var json = JsonConvert.SerializeObject(_cache, Formatting.Indented);
        File.WriteAllText(_cacheFilePath, json);
        Console.WriteLine("💾 Cache saved.");
    }

    public void Load()
    {
        if (!File.Exists(_cacheFilePath))
        {
            Console.WriteLine("📂 No existing cache file.");
            return;
        }

        var json = File.ReadAllText(_cacheFilePath);
        var data = JsonConvert.DeserializeObject<Dictionary<string, JObject>>(json);
        foreach (var kvp in data)
        {
            _cache[kvp.Key] = kvp.Value;
        }

        Console.WriteLine($"📥 Loaded {data.Count} cached items.");
    }

    public void Empty()
    {
        _cache.Clear();
        Console.WriteLine("🧹 Cache cleared.");
    }

    public Dictionary<string, JObject> GetAll() => _cache;

    public bool TryGet(string id, out JObject value) => _cache.TryGetValue(id, out value);

    // ───── STATS METHODS ─────

    public Dictionary<int, int> GetArticleCountByYear()
    {
        var result = new Dictionary<int, int>();

        foreach (var article in _cache.Values)
        {
            var dateStr = article["abstracts-retrieval-response"]?["coredata"]?["prism:coverDate"]?.ToString();
            if (DateTime.TryParse(dateStr, out var date))
            {
                int year = date.Year;
                result.TryAdd(year, 0);
                result[year]++;
            }
        }

        Console.WriteLine("\n--- Article Count by Year ---");
        foreach (var kvp in result.OrderBy(k => k.Key))
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        return result;
    }

    public Dictionary<string, string> GetKeywordTrendsAsDelimitedStrings()
    {
        var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var years = new SortedSet<int>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
            var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];
            var keywords = ExtractKeywordsFromArticle(article);

            var dateStr = core?["prism:coverDate"]?.ToString();
            if (!DateTime.TryParse(dateStr, out var date))
                continue;

            int year = date.Year;
            years.Add(year);

            foreach (var keyword in keywords.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (!keywordYearCount.ContainsKey(keyword))
                    keywordYearCount[keyword] = new Dictionary<int, int>();

                if (!keywordYearCount[keyword].ContainsKey(year))
                    keywordYearCount[keyword][year] = 0;

                keywordYearCount[keyword][year]++;
                keywordTotals.TryAdd(keyword, 0);
                keywordTotals[keyword]++;
            }
        }

        var topKeywords = keywordTotals
            .OrderByDescending(kvp => kvp.Value)
            .Take(20)
            .Select(kvp => kvp.Key)
            .ToList();

        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (var keyword in topKeywords)
        {
            var counts = years.Select(y => keywordYearCount[keyword].TryGetValue(y, out int c) ? c : 0);
            result[keyword] = string.Join(",", counts);
        }

        Console.WriteLine("\n--- Keyword Trends (CSV-style lines) ---");
        foreach (var kvp in result)
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        return result;
    }

    public Dictionary<string, string> GetKeywordTrendRatiosByYear()
    {
        var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var yearArticleCount = new Dictionary<int, int>();
        var years = new SortedSet<int>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
            var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];
            var keywords = ExtractKeywordsFromArticle(article);

            var dateStr = core?["prism:coverDate"]?.ToString();
            if (!DateTime.TryParse(dateStr, out var date))
                continue;

            int year = date.Year;
            years.Add(year);
            yearArticleCount.TryAdd(year, 0);
            yearArticleCount[year]++;

            foreach (var keyword in keywords.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (!keywordYearCount.ContainsKey(keyword))
                    keywordYearCount[keyword] = new Dictionary<int, int>();

                if (!keywordYearCount[keyword].ContainsKey(year))
                    keywordYearCount[keyword][year] = 0;

                keywordYearCount[keyword][year]++;
                keywordTotals.TryAdd(keyword, 0);
                keywordTotals[keyword]++;
            }
        }

        var topKeywords = keywordTotals
            .OrderByDescending(kvp => kvp.Value)
            .Take(20)
            .Select(kvp => kvp.Key)
            .ToList();

        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (var keyword in topKeywords)
        {
            var ratios = years.Select(y =>
            {
                int count = keywordYearCount[keyword].TryGetValue(y, out int c) ? c : 0;
                int total = yearArticleCount.TryGetValue(y, out int t) ? t : 1;
                return ((double)count / total).ToString("0.000");
            });

            result[keyword] = string.Join(",", ratios);
        }

        Console.WriteLine("\n--- Keyword Ratio Trends (CSV-style lines) ---");
        foreach (var kvp in result)
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        return result;
    }

    public Dictionary<string, string> GetSubjectTrendRatiosByYear()
    {
        var subjectYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var subjectTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var yearArticleCount = new Dictionary<int, int>();
        var years = new SortedSet<int>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
            var dateStr = core?["prism:coverDate"]?.ToString();
            if (!DateTime.TryParse(dateStr, out var date))
                continue;

            int year = date.Year;
            years.Add(year);
            yearArticleCount.TryAdd(year, 0);
            yearArticleCount[year]++;

            var subjectToken = article["abstracts-retrieval-response"]?["subject-areas"]?["subject-area"];
            var subjects = new List<string>();

            if (subjectToken is JArray subjectArray)
            {
                subjects.AddRange(subjectArray
                    .Select(s => s["$"]?.ToString()?.Trim())
                    .Where(s => !string.IsNullOrEmpty(s)));
            }
            else if (subjectToken is JObject single)
            {
                var s = single["$"]?.ToString()?.Trim();
                if (!string.IsNullOrEmpty(s))
                    subjects.Add(s);
            }

            foreach (var subject in subjects.Distinct())
            {
                if (!subjectYearCount.ContainsKey(subject))
                    subjectYearCount[subject] = new Dictionary<int, int>();

                if (!subjectYearCount[subject].ContainsKey(year))
                    subjectYearCount[subject][year] = 0;

                subjectYearCount[subject][year]++;
                subjectTotals.TryAdd(subject, 0);
                subjectTotals[subject]++;
            }
        }

        var topSubjects = subjectTotals
            .OrderByDescending(kvp => kvp.Value)
            .Take(20)
            .Select(kvp => kvp.Key)
            .ToList();

        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (var subject in topSubjects)
        {
            var ratios = years.Select(y =>
            {
                int count = subjectYearCount[subject].TryGetValue(y, out int c) ? c : 0;
                int total = yearArticleCount.TryGetValue(y, out int t) ? t : 1;
                return ((double)count / total).ToString("0.000");
            });

            result[subject] = string.Join(",", ratios);
        }

        Console.WriteLine("\n--- Subject Ratio Trends (CSV-style lines) ---");
        foreach (var kvp in result)
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        return result;
    }

    private List<string> ExtractKeywordsFromArticle(JObject article)
    {
        var keywords = new List<string>();

        var response = article["abstracts-retrieval-response"];
        if (response == null || response.Type != JTokenType.Object)
            return keywords;

        // 1. idxterms.mainterm
        try
        {
            var idxTerms = response["idxterms"];
            if (idxTerms != null && idxTerms.Type == JTokenType.Object)
            {
                var mainterms = idxTerms["mainterm"];
                if (mainterms != null)
                {
                    if (mainterms.Type == JTokenType.Array)
                    {
                        foreach (var t in mainterms)
                        {
                            var term = t["$"]?.ToString()?.Trim();
                            if (!string.IsNullOrEmpty(term))
                                keywords.Add(term.ToLowerInvariant());
                        }
                    }
                    else if (mainterms.Type == JTokenType.Object)
                    {
                        var term = mainterms["$"]?.ToString()?.Trim();
                        if (!string.IsNullOrEmpty(term))
                            keywords.Add(term.ToLowerInvariant());
                    }
                }
            }
        }
        catch { /* skip idxterms parsing on error */ }

        // 2. authkeywords (fallback)
        if (keywords.Count == 0)
        {
            try
            {
                var authKeywords = response["authkeywords"];
                if (authKeywords is JArray kwArray)
                {
                    keywords.AddRange(kwArray
                        .Select(k => k["$"]?.ToString()?.Trim()?.ToLowerInvariant())
                        .Where(k => !string.IsNullOrEmpty(k)));
                }
            }
            catch { /* skip fallback if malformed */ }
        }

        // 3. title fallback (last resort)
        if (keywords.Count == 0)
        {
            try
            {
                var title = response["coredata"]?["dc:title"]?.ToString();
                if (!string.IsNullOrEmpty(title))
                {
                    keywords.AddRange(
                        title.Split(new[] { ' ', ',', '.', ':', '-', '–' }, StringSplitOptions.RemoveEmptyEntries)
                             .Where(w => w.Length > 4)
                             .Select(w => w.ToLowerInvariant()));
                }
            }
            catch { /* ignore malformed title */ }
        }

        return keywords.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }


    public Dictionary<string, string> GetSubjectTrendsAsDelimitedStrings()
    {
        var subjectYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var subjectTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var years = new SortedSet<int>();

        foreach (var article in _cache.Values)
        {
            var subjectToken = article["abstracts-retrieval-response"]?["subject-areas"]?["subject-area"];
            var core = article["abstracts-retrieval-response"]?["coredata"];
            var dateStr = core?["prism:coverDate"]?.ToString();
            if (!DateTime.TryParse(dateStr, out var date))
                continue;

            int year = date.Year;
            years.Add(year);

            var subjects = new List<string>();
            if (subjectToken is JArray subjectArray)
            {
                subjects.AddRange(subjectArray
                    .Select(s => s["$"]?.ToString()?.Trim())
                    .Where(s => !string.IsNullOrEmpty(s)));
            }
            else if (subjectToken is JObject single)
            {
                var s = single["$"]?.ToString()?.Trim();
                if (!string.IsNullOrEmpty(s))
                    subjects.Add(s);
            }

            foreach (var subject in subjects.Distinct())
            {
                if (!subjectYearCount.ContainsKey(subject))
                    subjectYearCount[subject] = new Dictionary<int, int>();

                if (!subjectYearCount[subject].ContainsKey(year))
                    subjectYearCount[subject][year] = 0;

                subjectYearCount[subject][year]++;
                subjectTotals.TryAdd(subject, 0);
                subjectTotals[subject]++;
            }
        }

        var topSubjects = subjectTotals
            .OrderByDescending(kvp => kvp.Value)
            .Take(20)
            .Select(kvp => kvp.Key)
            .ToList();

        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        foreach (var subject in topSubjects)
        {
            var counts = years.Select(y => subjectYearCount[subject].TryGetValue(y, out int c) ? c : 0);
            result[subject] = string.Join(",", counts);
        }

        Console.WriteLine("\n--- Subject Trends (CSV-style lines) ---");
        foreach (var kvp in result)
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        return result;
    }


    public Dictionary<string, int> GetArticleCountBySubjectArea()
    {
        var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _cache.Values)
        {
            var subjects = article["abstracts-retrieval-response"]?["subject-areas"]?["subject-area"];
            if (subjects == null) continue;

            if (subjects is JArray subjectArray)
            {
                foreach (var s in subjectArray)
                {
                    string subject = s["$"]?.ToString()?.Trim();
                    if (!string.IsNullOrEmpty(subject))
                    {
                        result.TryAdd(subject, 0);
                        result[subject]++;
                    }
                }
            }
            else if (subjects is JObject singleSubject)
            {
                string subject = singleSubject["$"]?.ToString()?.Trim().Replace(" (all)","");
                if (!string.IsNullOrEmpty(subject))
                {
                    result.TryAdd(subject, 0);
                    result[subject]++;
                }
            }
        }

        Console.WriteLine("\n--- Article Count by Subject Area ---");
        foreach (var kvp in result.OrderByDescending(k => k.Value).Take(20))
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        return result;
    }
    public Dictionary<string, int> GetArticleCountByAuthor()
    {
        var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _cache.Values)
        {
            var authorsRoot = article["abstracts-retrieval-response"]?["authors"];

            if (authorsRoot == null || authorsRoot.Type != JTokenType.Object)
                continue;

            var authorToken = authorsRoot["author"];
            if (authorToken == null)
                continue;

            // If it's an array of authors
            if (authorToken.Type == JTokenType.Array)
            {
                foreach (var author in authorToken)
                {
                    string name = author["ce:indexed-name"]?.ToString()?.Trim();
                    if (!string.IsNullOrEmpty(name))
                    {
                        result.TryAdd(name, 0);
                        result[name]++;
                    }
                }
            }
            // If it's a single author object
            else if (authorToken.Type == JTokenType.Object)
            {
                string name = authorToken["ce:indexed-name"]?.ToString()?.Trim();
                if (!string.IsNullOrEmpty(name))
                {
                    result.TryAdd(name, 0);
                    result[name]++;
                }
            }
        }

        Console.WriteLine("\n--- Article Count by Author ---");
        foreach (var kvp in result.OrderByDescending(k => k.Value).Take(20))
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        return result;
    }

    public Dictionary<string, int> GetAuthorCitations()
    {
        var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
            if (!int.TryParse(core?["citedby-count"]?.ToString(), out int citations))
                citations = 0;

            var authorsRoot = article["abstracts-retrieval-response"]?["authors"];
            if (authorsRoot == null || authorsRoot.Type != JTokenType.Object)
                continue;

            var authorToken = authorsRoot["author"];
            if (authorToken == null)
                continue;

            if (authorToken.Type == JTokenType.Array)
            {
                foreach (var author in authorToken)
                {
                    string name = author["ce:indexed-name"]?.ToString()?.Trim();
                    if (!string.IsNullOrEmpty(name))
                    {
                        result.TryAdd(name, 0);
                        result[name] += citations;
                    }
                }
            }
            else if (authorToken.Type == JTokenType.Object)
            {
                string name = authorToken["ce:indexed-name"]?.ToString()?.Trim();
                if (!string.IsNullOrEmpty(name))
                {
                    result.TryAdd(name, 0);
                    result[name] += citations;
                }
            }
        }

        Console.WriteLine("\n--- Total Citations by Author ---");
        foreach (var kvp in result.OrderByDescending(k => k.Value).Take(20))
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        return result;
    }


    public Dictionary<string, Dictionary<int, int>> GetKeywordTrendsByYear()
    {
        var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var years = new SortedSet<int>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
            var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];
            //var keywords = new List<string>();

            var keywords = ExtractKeywordsFromArticle(article);

            var dateStr = core?["prism:coverDate"]?.ToString();
            if (!DateTime.TryParse(dateStr, out var date))
                continue;

            int year = date.Year;
            years.Add(year);

            foreach (var keyword in keywords.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (!keywordYearCount.ContainsKey(keyword))
                    keywordYearCount[keyword] = new Dictionary<int, int>();

                if (!keywordYearCount[keyword].ContainsKey(year))
                    keywordYearCount[keyword][year] = 0;

                keywordYearCount[keyword][year]++;
                keywordTotals.TryAdd(keyword, 0);
                keywordTotals[keyword]++;
            }
        }

        var topKeywords = keywordTotals
            .OrderByDescending(kvp => kvp.Value)
            .Take(20)
            .Select(kvp => kvp.Key)
            .ToList();

        var result = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        foreach (var keyword in topKeywords)
        {
            result[keyword] = new Dictionary<int, int>();
            foreach (var year in years)
            {
                int count = keywordYearCount[keyword].TryGetValue(year, out var c) ? c : 0;
                result[keyword][year] = count;
            }
        }

        // Print to console
        Console.WriteLine("\n--- Keyword Trends by Year ---");

        // Header
        var header = "Keyword".PadRight(20) + string.Join(" ", years.Select(y => y.ToString().PadLeft(6)));
        Console.WriteLine(header);
        Console.WriteLine(new string('-', header.Length));

        foreach (var kvp in result)
        {
            string line = kvp.Key.PadRight(20);
            foreach (var year in years)
            {
                int count = kvp.Value.TryGetValue(year, out var c) ? c : 0;
                line += count.ToString().PadLeft(6);
            }
            Console.WriteLine(line);
        }

        return result;
    }



    public Dictionary<string, int> GetArticleCountByKeyword()
    {
        var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
            var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];

            //var keywords = new List<string>();
            var keywords = ExtractKeywordsFromArticle(article);

            foreach (var keyword in keywords.Distinct())
            {
                result.TryAdd(keyword, 0);
                result[keyword]++;
            }
        }

        Console.WriteLine("\n--- Top Keywords by Count ---");
        foreach (var kvp in result.OrderByDescending(k => k.Value).Take(50))
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        return result;
    }

    public Dictionary<(int year, string keyword), int> GetArticleCountByYearAndKeyword()
    {
        var result = new Dictionary<(int, string), int>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
            var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];

            var dateStr = core?["prism:coverDate"]?.ToString();
            if (!DateTime.TryParse(dateStr, out var date))
                continue;

            int year = date.Year;
            //var keywords = new List<string>();
            var keywords = ExtractKeywordsFromArticle(article);

            foreach (var keyword in keywords.Distinct())
            {
                var key = (year, keyword);
                result.TryAdd(key, 0);
                result[key]++;
            }
        }

        Console.WriteLine("\n--- Keyword Count by Year (top 5 per year) ---");
        foreach (var group in result.GroupBy(kvp => kvp.Key.Item1).OrderBy(g => g.Key))
        {
            Console.WriteLine($"\n{group.Key}:");
            foreach (var kvp in group.OrderByDescending(kvp => kvp.Value).Take(5))
                Console.WriteLine($"  {kvp.Key.Item2}: {kvp.Value}");
        }

        return result;
    }

    public Dictionary<int, double> GetAverageCitationByYear()
    {
        var totals = new Dictionary<int, (int citationSum, int count)>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
            var dateStr = core?["prism:coverDate"]?.ToString();

            if (!DateTime.TryParse(dateStr, out var date))
                continue;

            int year = date.Year;

            if (int.TryParse(core?["citedby-count"]?.ToString(), out var citations))
            {
                if (!totals.ContainsKey(year))
                    totals[year] = (0, 0);

                totals[year] = (
                    totals[year].citationSum + citations,
                    totals[year].count + 1
                );
            }
        }

        var result = totals.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value.count > 0 ? (double)kvp.Value.citationSum / kvp.Value.count : 0.0
        );

        Console.WriteLine("\n--- Average Citations by Year ---");
        foreach (var kvp in result.OrderBy(k => k.Key))
            Console.WriteLine($"{kvp.Key}: {kvp.Value:F2}");

        return result;
    }
}
