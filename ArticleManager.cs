using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Scopus_Analysis.Helper;
using Scopus_Analysis.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Word2Vec;

public class AbstractData
{
    public string Text { get; set; }
}

public class ClusteringPrediction
{
    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId { get; set; }
}

public class TransformedAbstract
{
    public float[] Features { get; set; }
}

public class WordVector
{
    public string Word { get; set; }

    [VectorType] // This tells ML.NET this property is a vector
    public float[] Vector { get; set; }
}

public class WordClusterResult
{
    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId { get; set; }
}


public class ArticleManager
{
    public readonly string _cacheFilePath;
    public readonly Dictionary<string, JObject> _cache;
    public readonly List<Article> _articles;
    public readonly string _outputDirectory = @"C:\Development\Oxford\ox-identity-survey\data";

    public ArticleManager(string cacheFilePath)
    {
        _cacheFilePath = cacheFilePath;
        _cache = new Dictionary<string, JObject>();
        _articles = new List<Article>();
        Directory.CreateDirectory(_outputDirectory);
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

    #region Utility Functions
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

        foreach (var ob in _cache.Values)
        {
            _articles.Add(new Article(ob));
        }

        Console.WriteLine($"📥 Loaded {data.Count} cached items.");
    }

    #endregion

    public bool TryGet(string id, out JObject value) => _cache.TryGetValue(id, out value);

    public Dictionary<int, int> GetArticleCountByYear()
    {
        var result = new Dictionary<int, int>();

        foreach (var ob in _cache.Values)
        {
            var article = new Article(ob);
            result.TryAdd(article.Published.Year, 0);
            result[article.Published.Year]++;
        }

        Console.WriteLine("\n--- Article Count by Year ---");
        foreach (var kvp in result.OrderBy(k => k.Key))
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "article_count_by_year.csv");
        var lines = new List<string> { "Year,Count" };
        lines.AddRange(result.OrderBy(k => k.Key).Select(kvp => $"{kvp.Key},{kvp.Value}"));
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }

    public Dictionary<string, string> GetKeywordTrendsAsDelimitedStrings()
    {
        var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var years = new SortedSet<int>();


        foreach (var article in _articles) 
        { 
            var keywords = article.Keywords;
            var year = article.Published.Year;
            years.Add(article.Published.Year);

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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "keyword_trends_by_year.csv");
        using (var writer = new StreamWriter(csvPath))
        {
            // Header
            writer.Write("Keyword");
            foreach (var year in years)
                writer.Write($",{year}");
            writer.WriteLine();

            // Rows
            foreach (var kvp in result)
            {
                writer.WriteLine($"{kvp.Key},{kvp.Value}");
            }
        }

        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }

    public Dictionary<string, string> GetKeywordTrendRatiosByYear()
    {
        var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var yearArticleCount = new Dictionary<int, int>();
        var years = new SortedSet<int>();

        foreach (var article in _articles)
        {
            var keywords = article.Keywords;

            int year = article.Published.Year;
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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "keyword_ratios_by_year.csv");
        using (var writer = new StreamWriter(csvPath))
        {
            // Header
            writer.Write("Keyword");
            foreach (var year in years)
                writer.Write($",{year}");
            writer.WriteLine();

            foreach (var kvp in result)
            {
                writer.WriteLine($"{kvp.Key},{kvp.Value}");
            }
        }

        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }

    public Dictionary<string, string> GetSubjectTrendRatiosByYear()
    {
        var subjectYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var subjectTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var yearArticleCount = new Dictionary<int, int>();
        var years = new SortedSet<int>();

        foreach (var article in _articles)
        {
            var year = article.Published.Year;
            years.Add(year);
            yearArticleCount.TryAdd(year, 0);
            yearArticleCount[year]++;

            var subjects = article.Subjects;

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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "subject_ratios_by_year.csv");
        using (var writer = new StreamWriter(csvPath))
        {
            // Write header
            writer.Write("Subject");
            foreach (var year in years)
                writer.Write($",{year}");
            writer.WriteLine();

            foreach (var kvp in result)
            {
                writer.WriteLine($"{kvp.Key},{kvp.Value}");
            }
        }

        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }


    public Dictionary<string, string> GetSubjectTrendsAsDelimitedStrings()
    {
        var subjectYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var subjectTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var years = new SortedSet<int>();

        foreach (var article in _articles)
        {
            int year = article.Published.Year;
            years.Add(year);

            var subjects = article.Subjects;

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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "subject_trends_by_year.csv");
        using (var writer = new StreamWriter(csvPath))
        {
            // Header
            writer.Write("Subject");
            foreach (var year in years)
                writer.Write($",{year}");
            writer.WriteLine();

            foreach (var kvp in result)
            {
                writer.WriteLine($"{kvp.Key},{kvp.Value}");
            }
        }

        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }

    public void Increment(Dictionary<string, int> dict, string key)
    {
        if (!string.IsNullOrEmpty(key))
        {
            dict.TryAdd(key, 0);
            dict[key]++;
        }
    }

    public void Increment(Dictionary<string, int> dict, string key, int count)
    {
        if (!string.IsNullOrEmpty(key))
        {
            dict.TryAdd(key, 0);
            dict[key]+= count;
        }
    }

    public void IncrementForAll(Dictionary<string, int> dict, List<string> strings)
    {
        if (strings == null) return;

        foreach (var str in strings)
        {
            if (!string.IsNullOrEmpty(str))
            {
                dict.TryAdd(str, 0);
                dict[str]++;
            }
        }
    }

    public void WriteToCSV(Dictionary<string,int> dict, string keyName, string valueName, string fileName)
    {
        var csvPath = Path.Combine(_outputDirectory, fileName);
        var lines = new List<string> { keyName + "," + valueName };
        lines.AddRange(dict.OrderByDescending(kvp => kvp.Value).Select(kvp => $"{kvp.Key},{kvp.Value}"));
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);
    }

    public void WriteArticles(List<Article> topArticles, List<string> lines, string[] vars, string key)
    {
        foreach (var article in topArticles)
        {
            Console.WriteLine($" [{article.Citations}][{article.Published.Year}]{article.Title}");

            var articleType = article.GetType();

            foreach (var name in vars)
            {
                var property = articleType.GetProperty(name);
                if (property == null) continue;

                var value = property.GetValue(article);

                switch (value)
                {
                    case IEnumerable<string> enumerable:
                        Console.WriteLine($"  {name}: {string.Join("; ", enumerable)}");
                        break;

                    case string s:
                        string output = s.Length > 200 ? s.Substring(0, 200) + "..." : s;
                        Console.WriteLine($"  {name}: {output}");
                        break;
                }
            }

            lines.Add($"\"{key}\",[{article.Published.Year}],\"{article.Title.Replace("\"", "'")}\",[{article.Citations}],\"{string.Join(";", article.Subjects)}\",\"{string.Join(";", article.Keywords)}\",\"{string.Join(";", article.Authors)}\",\"{article.Abstract.Replace("\"", "'")}\"");
        }
    }

    public Dictionary<string, int> GetArticleCountBySubjectArea()
    {
        var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _articles)
        {
            IncrementForAll(result, article.Subjects);
        }

        WriteToConsole(result, "\n--- Article Count by Subject Area ---");
        WriteToCSV(result, "Subject", "Count", "subject_area_count.csv");

        return result;
    }

    private static void WriteToConsole(Dictionary<string, int> result, string name, int count = 100)
    {
        Console.WriteLine(name);
        foreach (var kvp in result.OrderByDescending(k => k.Value).Take(count))
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");
    }

    public void ExportTopArticlesPerSubject(int topN = 10)
    {
        var subjectArticles = new Dictionary<string, List<(string Title, List<string> Authors, List<string> Keywords, string Abstract, int Citations)>>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _cache.Values)
        {
            var response = article["abstracts-retrieval-response"];
            if (response == null) continue;

            // Title
            var title = response["coredata"]?["dc:title"]?.ToString()?.Trim();
            if (string.IsNullOrWhiteSpace(title)) continue;

            // Abstract
            var abstractText = response["coredata"]?["dc:description"]?.ToString()?.Trim() ?? "";

            // Citations
            int citations = 0;
            int.TryParse(response["coredata"]?["citedby-count"]?.ToString(), out citations);

            // Authors
            var authors = new List<string>();
            var authorToken = response["authors"]?["author"];
            if (authorToken is JArray aArray)
            {
                foreach (var a in aArray.OfType<JObject>())
                {
                    var name = a["ce:indexed-name"]?.ToString()?.Trim();
                    if (!string.IsNullOrWhiteSpace(name)) authors.Add(name);
                }
            }
            else if (authorToken is JObject singleAuthor)
            {
                var name = singleAuthor["ce:indexed-name"]?.ToString()?.Trim();
                if (!string.IsNullOrWhiteSpace(name)) authors.Add(name);
            }

            // Keywords
            var keywords = new List<string>();
            var keywordWrapper = response["authkeywords"];
            if (keywordWrapper is JObject kwObj && kwObj.TryGetValue("author-keyword", out var keywordToken))
            {
                if (keywordToken is JArray kArray)
                {
                    keywords = kArray
                        .OfType<JObject>()
                        .Select(k => k["$"]?.ToString())
                        .Where(k => !string.IsNullOrWhiteSpace(k))
                        .ToList();
                }
                else if (keywordToken is JObject kObj)
                {
                    var kw = kObj["$"]?.ToString();
                    if (!string.IsNullOrWhiteSpace(kw))
                        keywords.Add(kw);
                }
            }

            // Subject Areas
            var subjectToken = response["subject-areas"]?["subject-area"];
            var subjects = new List<string>();
            if (subjectToken is JArray sArray)
            {
                subjects = sArray
                    .OfType<JObject>()
                    .Select(s => s["$"]?.ToString()?.Trim())
                    .Where(s => !string.IsNullOrWhiteSpace(s))
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();
            }
            else if (subjectToken is JObject sObj)
            {
                var subject = sObj["$"]?.ToString()?.Trim();
                if (!string.IsNullOrWhiteSpace(subject)) subjects.Add(subject);
            }

            // Associate with each subject
            foreach (var subject in subjects)
            {
                if (!subjectArticles.ContainsKey(subject))
                    subjectArticles[subject] = new List<(string, List<string>, List<string>, string, int)>();

                subjectArticles[subject].Add((title, authors, keywords, abstractText, citations));
            }
        }

        // Sort and export
        var outputPath = Path.Combine(_outputDirectory, "top_articles_by_subject.csv");
        var lines = new List<string> { "Subject,Title,Citations,Authors,Keywords,Abstract" };

        foreach (var kvp in subjectArticles.OrderBy(k => k.Key))
        {
            var topArticles = kvp.Value
                .OrderByDescending(a => a.Citations)
                .Take(topN);

            foreach (var (Title, Authors, Keywords, Abstract, Citations) in topArticles)
            {
                string authorsStr = string.Join("; ", Authors);
                string keywordsStr = string.Join("; ", Keywords);
                string cleanedAbstract = Abstract.Replace("\n", " ").Replace("\r", " ").Replace(",", " "); // Basic CSV-safe formatting

                lines.Add($"\"{kvp.Key}\",\"{Title}\",{Citations},\"{authorsStr}\",\"{keywordsStr}\",\"{cleanedAbstract}\"");
            }
        }

        foreach (var kvp in subjectArticles.OrderBy(k => k.Key))
        {
            Console.WriteLine($"\n--- Subject: {kvp.Key} ---");
            var topArticles = kvp.Value.OrderByDescending(a => a.Citations).Take(topN);

            foreach (var (Title, Authors, Keywords, Abstract, Citations) in topArticles)
            {
                Console.WriteLine($"[{Citations}] {Title}");
                Console.WriteLine($"  Authors: {string.Join("; ", Authors)}");
                Console.WriteLine($"  Keywords: {string.Join("; ", Keywords)}");
                Console.WriteLine($"  Abstract: {Abstract.Substring(0, Math.Min(200, Abstract.Length))}...");
                Console.WriteLine();
            }
        }

        File.WriteAllLines(outputPath, lines);
        Console.WriteLine("Top articles by subject exported to: " + outputPath);
    }


    public List<string> GetClusteredWords()
    {
        //word = word.ToLower();
        var keywordFreq = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var ob in _cache.Values)
        {
            var article = new Article(ob);
            var keywords = article.Keywords;// ExtractKeywordsFromArticle(article);
            foreach (var kw in keywords)
            {
                var word2 = kw.ToLowerInvariant();
                keywordFreq.TryAdd(word2, 0);
                keywordFreq[word2]++;
            }
        }

        var allVectors = NLPHelper.LoadWord2VecTxt(_outputDirectory, 100);

        // Step 3: Take top N keywords that exist in the embedding model
        var topKeywords = keywordFreq
            .Where(kvp => allVectors.ContainsKey(kvp.Key))
            .OrderByDescending(kvp => kvp.Value)
            .Take(150)
            .Select(kvp => kvp.Key)
            .ToList();

        return topKeywords;//.Contains(word);
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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "author_article_count.csv");
        var lines = new List<string> { "Author,Count" };
        lines.AddRange(result.OrderByDescending(kvp => kvp.Value).Select(kvp => $"{kvp.Key},{kvp.Value}"));
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }

    public void GetTopArticlesPerSubject()
    {
        var subjectToArticles = new Dictionary<string, List<Article>>();

        var clusterWords = GetClusteredWords();

        foreach (var article in _articles)
        {

            int year = article.Published.Year;

            string title = article.Title;
            string abstractText = article.Abstract;
            int citations = article.Citations;

            // Subjects
            var subjects = article.Subjects;

            if (subjects.Count == 0) continue;

            // Keywords
            var keywords = article.Keywords;

            // Authors
            var authors = article.Authors;

            foreach (var keyword in keywords.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (!subjectToArticles.ContainsKey(keyword) && clusterWords.Contains(keyword))
                    subjectToArticles[keyword] = new List<Article>();

                if (subjectToArticles.Keys.Contains(keyword))
                {
                    subjectToArticles[keyword].Add(article);
                }
            }
        }

        // Write to console and CSV
        var lines = new List<string> { "Subject,Title,Citations,Subjects,Keywords,Authors,Abstract" };
        Console.WriteLine("\n--- Top 10 Articles per Subject ---");

        foreach (var kvp in subjectToArticles.OrderBy(k => k.Key))
        {
            var topArticles = kvp.Value
                .OrderByDescending(a => a.Citations)
                .Where(a => a.Citations > 5 * (2025-a.Published.Year))
                .Take(10)
                .ToList();

            if (topArticles.Count == 0) continue;

            Console.WriteLine($"\nSubject: {kvp.Key}");
            WriteArticles(topArticles, lines, ["Keywords", "Authors", "Subjects", "Abstract"], kvp.Key);

        }

        var csvPath = Path.Combine(_outputDirectory, "top_articles_by_subject.csv");
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);
    }

    public Dictionary<string, int> GetAuthorCitations()
    {
        var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _articles)
        { 
            Increment(result, article.PrimaryAuthor, article.Citations);
        }

        WriteToConsole(result, "\n--- Total Citations by Author ---", 20);
        WriteToCSV(result, "Author", "Citations", "author_citations.csv");

        return result;
    }

    public Dictionary<string, Dictionary<int, int>> GetKeywordTrendsByYear()
    {
        var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var years = new SortedSet<int>();

        foreach (var ob in _cache.Values)
        {
            var article = new Article(ob);
            var keywords = article.Keywords;


            int year = article.Published.Year;
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

        // Write to CSV
        var csvPath = Path.Combine(_outputDirectory, "keyword_trends_by_year_table.csv");
        using (var writer = new StreamWriter(csvPath))
        {
            writer.Write("Keyword");
            foreach (var year in years)
                writer.Write($",{year}");
            writer.WriteLine();

            foreach (var kvp in result)
            {
                writer.Write($"{kvp.Key}");
                foreach (var year in years)
                {
                    int count = kvp.Value.TryGetValue(year, out var c) ? c : 0;
                    writer.Write($",{count}");
                }
                writer.WriteLine();
            }
        }

        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }

    public Dictionary<string, int> GetArticleCountByKeyword()
    {
        var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var ob in _cache.Values)
        {
            var article = new Article(ob);
            var keywords = article.Keywords;

            foreach (var keyword in keywords.Distinct())
            {
                result.TryAdd(keyword, 0);
                result[keyword]++;
            }
        }

        Console.WriteLine("\n--- Top Keywords by Count ---");
        foreach (var kvp in result.OrderByDescending(k => k.Value).Take(50))
            Console.WriteLine($"{kvp.Key}: {kvp.Value}");

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "keyword_count.csv");
        var lines = new List<string> { "Keyword,Count" };
        lines.AddRange(result.OrderByDescending(kvp => kvp.Value).Select(kvp => $"{kvp.Key},{kvp.Value}"));
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }

    public Dictionary<(int year, string keyword), int> GetArticleCountByYearAndKeyword()
    {
        var result = new Dictionary<(int, string), int>();

        foreach (var ob in _cache.Values)
        {
            var article = new Article(ob);

            int year = article.Published.Year;
            var keywords = article.Keywords; 

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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "article_count_by_year_and_keyword.csv");
        var csvLines = new List<string> { "Year,Keyword,Count" };
        csvLines.AddRange(result.Select(kvp => $"{kvp.Key.Item1},{kvp.Key.Item2},{kvp.Value}"));
        File.WriteAllLines(csvPath, csvLines);
        Console.WriteLine("CSV saved to: " + csvPath);

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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "average_citations_by_year.csv");
        var lines = new List<string> { "Year,AvgCitations" };
        lines.AddRange(result.OrderBy(kvp => kvp.Key).Select(kvp => $"{kvp.Key},{kvp.Value:F2}"));
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }

    public void ClusterCustomWord2VecModel(int dimensions = 100, int numClusters = 8, int topN = 100)
    {
        var context = new MLContext();

        //var filePath = Path.Combine(_outputDirectory, "corpus.txt");
        //var modelPath = Path.Combine(_outputDirectory, "model.bin");

        NLPHelper.BuildCorpusModel(_outputDirectory);

        // Step 1: Load Word2Vec .bin vectors
        var allVectors = NLPHelper.LoadWord2VecTxt(_outputDirectory, dimensions);

        // Step 2: Extract and rank keywords from your articles
        var keywordFreq = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var ob in _cache.Values)
        {
            var article = new Article(ob);
            var keywords = article.Keywords; 
            foreach (var kw in keywords)
            {
                var word = kw.ToLowerInvariant();
                keywordFreq.TryAdd(word, 0);
                keywordFreq[word]++;
            }
        }

        // Step 3: Take top N keywords that exist in the embedding model
        var topKeywords = keywordFreq
            .Where(kvp => allVectors.ContainsKey(kvp.Key))
            .OrderByDescending(kvp => kvp.Value)
            .Take(topN)
            .Select(kvp => kvp.Key)
            .ToList();

        if (topKeywords.Count < numClusters)
        {
            Console.WriteLine("Not enough valid keywords found in the embedding model.");
            return;
        }

        // Step 4: Build WordVector list
        var wordVectors = topKeywords.Select(w => new WordVector
        {
            Word = w,
            Vector = allVectors[w]
        }).ToList();

        // Step 5: Create schema and IDataView
        var schema = SchemaDefinition.Create(typeof(WordVector));
        schema[nameof(WordVector.Vector)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, dimensions);
        var dataView = context.Data.LoadFromEnumerable(wordVectors, schema);

        // Step 6: KMeans clustering
        var options = new Microsoft.ML.Trainers.KMeansTrainer.Options
        {
            NumberOfClusters = numClusters,
            FeatureColumnName = nameof(WordVector.Vector)
        };

        var model = context.Clustering.Trainers.KMeans(options).Fit(dataView);
        var predictions = model.Transform(dataView);
        var clusterResults = context.Data.CreateEnumerable<WordClusterResult>(predictions, reuseRowObject: false).ToList();

        // Step 7: Group words by cluster
        var clusters = new Dictionary<uint, List<string>>();
        for (int i = 0; i < wordVectors.Count; i++)
        {
            var clusterId = clusterResults[i].PredictedClusterId;
            if (!clusters.ContainsKey(clusterId))
                clusters[clusterId] = new List<string>();
            clusters[clusterId].Add(wordVectors[i].Word);
        }

        // Step 8: Output to console and CSV
        Console.WriteLine("\n--- Clusters from Custom Word2Vec Model ---");

        var csvPath = Path.Combine(_outputDirectory, "custom_word2vec_clusters.csv");
        using (var writer = new StreamWriter(csvPath))
        {
            writer.WriteLine("Cluster,Words");

            foreach (var kvp in clusters.OrderBy(c => c.Key))
            {
                var line = string.Join(", ", kvp.Value.OrderBy(w => w));
                Console.WriteLine($"Cluster {kvp.Key}: {line}");
                writer.WriteLine($"{kvp.Key},\"{line}\"");
            }
        }

        Console.WriteLine("CSV saved to: " + csvPath);

        Console.WriteLine("\n--- Top Articles per Cluster (5 recent + 5 historical, Title Contains 'Digital Identity') ---");

        int cutoffYear = DateTime.Now.Year - 5;

        // Step 1: Build clusterWords lookup
        var clusterWordSets = clusters.ToDictionary(
            c => c.Key,
            c => new HashSet<string>(c.Value, StringComparer.OrdinalIgnoreCase)
        );

        // Step 2: Assign each article to its best-matching cluster (based on keyword overlap)
        var articleAssignments = new List<(uint ClusterId, string Title, string Author, int Year, int Citations)>();

        foreach (var ob in _cache.Values)
        {
            var article = new Article(ob);

            var title = article.Title;
            var keywords = article.Keywords;
            if (keywords.Count == 0) continue;

            var titleLower = title.ToLowerInvariant();
            var keywordSet = new HashSet<string>(keywords, StringComparer.OrdinalIgnoreCase);

            bool titleMatches =
                titleLower.Contains("digital identity") ||
                (keywordSet.Contains("digital") && keywordSet.Contains("identity")) ||
                (titleLower.Contains("identity") && keywordSet.Contains("digital"));

            if (!titleMatches)
                continue;

            if (keywords.Count == 0) continue;

            //string firstAuthor = "Unknown";
            //var authorsRoot = article["abstracts-retrieval-response"]?["authors"];
            //if (authorsRoot is JObject authorsObj)
            //{
            //    var authorToken = authorsObj["author"];
            //    if (authorToken is JArray authorArray && authorArray.Count > 0)
            //        firstAuthor = authorArray[0]?["ce:indexed-name"]?.ToString() ?? "Unknown";
            //    else if (authorToken is JObject singleAuthor)
            //        firstAuthor = singleAuthor["ce:indexed-name"]?.ToString() ?? "Unknown";
            //}

            //int citationCount = int.TryParse(core?["citedby-count"]?.ToString(), out var c) ? c : 0;

            var bestCluster = clusterWordSets
            .Select(kvp =>
            {
                int matchCount = keywords.Count(kw =>
                    kvp.Value.Any(cw =>
                    kw.Contains(cw, StringComparison.OrdinalIgnoreCase) ||
                    cw.Contains(kw, StringComparison.OrdinalIgnoreCase)
                ));

                return new
                {
                    ClusterId = kvp.Key,
                    MatchCount = matchCount
                };
            })
            .OrderByDescending(x => x.MatchCount)
            .FirstOrDefault();

            if (bestCluster != null && bestCluster.MatchCount > 0)
            {
                articleAssignments.Add((
                    ClusterId: bestCluster.ClusterId,
                    Title: title,
                    Author: article.PrimaryAuthor,
                    Year: article.Published.Year,
                    Citations: article.Citations
                ));
            }
        }

        // Step 3: Output top 5 recent + 5 earlier articles per cluster
        foreach (var clusterId in clusterWordSets.Keys.OrderBy(k => k))
        {
            var assignedArticles = articleAssignments
                .Where(a => a.ClusterId == clusterId)
                .OrderByDescending(a => a.Citations)
                .ToList();

            var recent = assignedArticles
                .Where(a => a.Year >= cutoffYear)
                .Take(5)
                .ToList();

            var historical = assignedArticles
                .Where(a => a.Year < cutoffYear)
                .Take(5)
                .ToList();

            Console.WriteLine($"\nCluster {clusterId}:");

            foreach (var article in recent)
                Console.WriteLine($"• [Recent {article.Year}] {article.Title} — {article.Author} ({article.Citations} citations)");

            foreach (var article in historical)
                Console.WriteLine($"• [Historical {article.Year}] {article.Title} — {article.Author} ({article.Citations} citations)");
        }

        Console.WriteLine("\n--- Top 30 Cited Articles with Unmatched Keywords ---");

        // Flatten all cluster words into one set
        var allClusterWords = new HashSet<string>(
            clusterWordSets.SelectMany(c => c.Value),
            StringComparer.OrdinalIgnoreCase
        );

        var unmatchedArticles = _cache.Values
            .Select(ob =>
            {
                var article = new Article(ob);
                var title = article.Title;
                var citationCount = article.Citations;

                string firstAuthor = article.PrimaryAuthor;

                var keywords = article.Keywords;
                var hasMatch = keywords.Any(kw =>
                    allClusterWords.Any(cw =>
                        kw.Contains(cw, StringComparison.OrdinalIgnoreCase) ||
                        cw.Contains(kw, StringComparison.OrdinalIgnoreCase)
                    ));

                return new
                {
                    Title = title,
                    Author = firstAuthor,
                    Citations = citationCount,
                    HasMatch = hasMatch,
                    Keywords = string.Join(", ", keywords.OrderBy(k => k))
                };
            })
            .Where(x => !x.HasMatch)
            .OrderByDescending(x => x.Citations)
            .Take(30)
            .ToList();

        // Output results
        foreach (var article in unmatchedArticles)
        {
            Console.WriteLine($"• {article.Title} — {article.Author} ({article.Citations} citations)");
            Console.WriteLine($"  Keywords: {article.Keywords}\n");
        }

        Console.WriteLine("\n--- Top 5 Cited Articles per Subject Area (with >5 citations, prioritizing primary subject) ---");

        var subjectToArticles = new Dictionary<string, List<(string Title, string Author, int Year, int Citations, string Keywords, string Subjects)>>(StringComparer.OrdinalIgnoreCase);

        foreach (var ob in _cache.Values)
        {
            var article = new Article(ob);
            var title = article.Title;
            var citationCount = article.Citations;
            if (citationCount <= 5) continue; // Skip low-citation articles

            var year = article.Published.Year;
            var keywords = article.Keywords;
            string keywordStr = string.Join(", ", keywords.OrderBy(k => k));

            string firstAuthor = article.PrimaryAuthor;


            

            string subjectListStr = string.Join(", ", article.Subjects);

            if (!string.IsNullOrWhiteSpace(article.PrimarySubject))
            {
                if (!subjectToArticles.ContainsKey(article.PrimarySubject))
                    subjectToArticles[article.PrimarySubject] = new List<(string, string, int, int, string, string)>();

                //subjectToArticles[primarySubject].Add((title, firstAuthor, year, citationCount, keywordStr));
                subjectToArticles[article.PrimarySubject].Add(
                        (title, firstAuthor, year, citationCount, keywordStr, subjectListStr)
                );
            }
        }

        // Output top 5 articles per subject
        foreach (var kvp in subjectToArticles.OrderBy(k => k.Key))
        {
            var subject = kvp.Key;
            var articles = kvp.Value
                .OrderByDescending(a => a.Citations)
                .Where(a => a.Subjects.ToLower().Contains("identity") || a.Keywords.ToLower().Contains("identity") || a.Title.ToLower().Contains("identity"))
                .Take(5)
                .ToList();

            if (articles.Count == 0) continue;

            Console.WriteLine($"\nSubject: {subject}");
            foreach (var article in articles)
            {
                Console.WriteLine($"• [{article.Year}] {article.Title} — {article.Author} ({article.Citations} citations)");
                Console.WriteLine($"  Keywords: {article.Keywords}");
                Console.WriteLine($"  Subjects: {article.Subjects}");
            }
        }

    }

    



}
