using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Word2Vec;

public static class Normaliser
{
    public static string Normalise(string word)
    {
        // Remove any non-letter characters (numbers, punctuation, whitespace, etc.) from the start of the word..
        var startAt = -1;
        for (var i = 0; i < word.Length; i++)
        {
            if (char.IsLetter(word[i]))
            {
                startAt = i;
                break;
            }
        }
        if (startAt == -1)
        {
            return "";
        }
        else if (startAt > 0)
        {
            word = word[startAt..];
        }

        // .. and from the end
        var lengthToKeep = -1;
        for (var i = word.Length - 1; i >= 0; i--)
        {
            if (char.IsLetter(word[i]))
            {
                lengthToKeep = i + 1;
                break;
            }
        }
        if (lengthToKeep < word.Length)
        {
            word = word[..lengthToKeep];
        }

        // Make the casing consistent
        return word.ToLower();
    }
}

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
    private readonly string _cacheFilePath;
    private readonly Dictionary<string, JObject> _cache;
    private readonly string _outputDirectory = @"C:\Development\Oxford\ox-identity-survey\data";

    public ArticleManager(string cacheFilePath)
    {
        _cacheFilePath = cacheFilePath;
        _cache = new Dictionary<string, JObject>();
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




    public void GetAbstractClusters(int numClusters = 5)
    {
        var context = new MLContext();

        // Extract abstract text from cache
        var abstracts = _cache.Values
            .Select(j => j["abstracts-retrieval-response"]?["coredata"]?["dc:description"]?.ToString())
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();

        if (!abstracts.Any())
        {
            Console.WriteLine("No abstracts available for clustering.");
            return;
        }

        var data = context.Data.LoadFromEnumerable(abstracts.Select(a => new AbstractData { Text = a }));

        // TF-IDF featurization
        var pipeline = context.Transforms.Text.FeaturizeText(
            outputColumnName: "Features", inputColumnName: nameof(AbstractData.Text));

        var transformedData = pipeline.Fit(data).Transform(data);

        // KMeans clustering
        var options = new Microsoft.ML.Trainers.KMeansTrainer.Options
        {
            NumberOfClusters = numClusters,
            FeatureColumnName = "Features"
        };

        var trainer = context.Clustering.Trainers.KMeans(options);
        var model = trainer.Fit(transformedData);
        var predictions = model.Transform(transformedData);

        var predictionResults = context.Data.CreateEnumerable<ClusteringPrediction>(predictions, reuseRowObject: false).ToList();

        // Output cluster results
        Console.WriteLine("\n--- Abstract Clusters ---");
        var csvPath = Path.Combine(_outputDirectory, "abstract_clusters.csv");
        using (var writer = new StreamWriter(csvPath))
        {
            writer.WriteLine("Cluster,Abstract");

            for (int i = 0; i < abstracts.Count; i++)
            {
                var clusterId = predictionResults[i].PredictedClusterId;
                var text = abstracts[i].Replace("\n", " ").Replace("\r", " ").Replace("\"", "'").Trim();
                var preview = text.Length > 300 ? text.Substring(0, 300) + "..." : text;

                Console.WriteLine($"Cluster {clusterId}: {preview}");
                writer.WriteLine($"{clusterId},\"{preview}\"");
            }
        }

        Console.WriteLine("CSV saved to: " + csvPath);
    }

    public void GetKeywordClusters(int numClusters = 5, int topN = 100)
    {
        var context = new MLContext();

        // Step 1: Aggregate all keywords
        var keywordCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _cache.Values)
        {
            var keywords = ExtractKeywordsFromArticle(article);
            foreach (var keyword in keywords)
            {
                var k = keyword.ToLowerInvariant().Trim();
                keywordCounts.TryAdd(k, 0);
                keywordCounts[k]++;
            }
        }

        if (keywordCounts.Count == 0)
        {
            Console.WriteLine("No keywords available.");
            return;
        }

        // Step 2: Take top N keywords
        var topKeywords = keywordCounts
            .OrderByDescending(kvp => kvp.Value)
            .Take(topN)
            .Select(kvp => kvp.Key)
            .ToList();

        if (topKeywords.Count < numClusters)
        {
            Console.WriteLine("Not enough keywords to form requested number of clusters.");
            return;
        }

        // Step 3: Create dummy documents from keywords
        var keywordSentences = topKeywords.Select(k => new AbstractData { Text = k }).ToList();
        var data = context.Data.LoadFromEnumerable(keywordSentences);

        // Step 4: TF-IDF vectorization
        var pipeline = context.Transforms.Text.FeaturizeText(
            outputColumnName: "Features", inputColumnName: nameof(AbstractData.Text));

        var transformedData = pipeline.Fit(data).Transform(data);

        // Step 5: KMeans clustering
        var trainer = context.Clustering.Trainers.KMeans(new Microsoft.ML.Trainers.KMeansTrainer.Options
        {
            NumberOfClusters = numClusters,
            FeatureColumnName = "Features"
        });

        var model = trainer.Fit(transformedData);
        var predictions = model.Transform(transformedData);

        var predictionResults = context.Data.CreateEnumerable<ClusteringPrediction>(predictions, reuseRowObject: false).ToList();

        // Step 6: Group keywords by cluster
        var keywordClusters = new Dictionary<uint, List<string>>();
        for (int i = 0; i < topKeywords.Count; i++)
        {
            var clusterId = predictionResults[i].PredictedClusterId;
            if (!keywordClusters.ContainsKey(clusterId))
                keywordClusters[clusterId] = new List<string>();

            keywordClusters[clusterId].Add(topKeywords[i]);
        }

        // Step 7: Output to console and CSV
        Console.WriteLine($"\n--- Keyword Clusters (Top {topN}) ---");
        var csvPath = Path.Combine(_outputDirectory, "keyword_clusters.csv");
        using (var writer = new StreamWriter(csvPath))
        {
            writer.WriteLine("Cluster,Keywords");
            foreach (var cluster in keywordClusters.OrderBy(k => k.Key))
            {
                var keywordsLine = string.Join(", ", cluster.Value);
                Console.WriteLine($"Cluster {cluster.Key}: {keywordsLine}");
                writer.WriteLine($"{cluster.Key},\"{keywordsLine}\"");
            }
        }

        Console.WriteLine("CSV saved to: " + csvPath);
    }

    public void EvaluateKeywordClusterCoherence(int minClusters = 2, int maxClusters = 10, int topN = 100)
    {
        var context = new MLContext();

        // Step 1: Gather and count keywords
        var keywordCounts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var article in _cache.Values)
        {
            var keywords = ExtractKeywordsFromArticle(article);
            foreach (var keyword in keywords.Select(k => k.ToLowerInvariant().Trim()))
            {
                keywordCounts.TryAdd(keyword, 0);
                keywordCounts[keyword]++;
            }
        }

        var topKeywords = keywordCounts
            .OrderByDescending(kvp => kvp.Value)
            .Take(topN)
            .Select(kvp => kvp.Key)
            .ToList();

        if (topKeywords.Count < maxClusters)
        {
            Console.WriteLine("Not enough keywords to evaluate clustering.");
            return;
        }

        var keywordData = topKeywords.Select(k => new AbstractData { Text = k }).ToList();
        var data = context.Data.LoadFromEnumerable(keywordData);

        var pipeline = context.Transforms.Text.FeaturizeText("Features", nameof(AbstractData.Text));
        var transformedData = pipeline.Fit(data).Transform(data);

        var featureVectors = context.Data
            .CreateEnumerable<TransformedAbstract>(transformedData, reuseRowObject: false)
            .Select(x => x.Features)
            .ToList();

        double bestCoherence = double.MinValue;
        int bestK = -1;

        for (int k = minClusters; k <= maxClusters; k++)
        {
            var kmeans = context.Clustering.Trainers.KMeans(new Microsoft.ML.Trainers.KMeansTrainer.Options
            {
                NumberOfClusters = k,
                FeatureColumnName = "Features"
            });

            var model = kmeans.Fit(transformedData);
            var predictions = model.Transform(transformedData);
            var clusterResults = context.Data.CreateEnumerable<ClusteringPrediction>(predictions, reuseRowObject: false).ToList();

            var clusterGroups = new Dictionary<uint, List<float[]>>();
            for (int i = 0; i < clusterResults.Count; i++)
            {
                uint clusterId = clusterResults[i].PredictedClusterId;
                if (!clusterGroups.ContainsKey(clusterId))
                    clusterGroups[clusterId] = new List<float[]>();
                clusterGroups[clusterId].Add(featureVectors[i]);
            }

            // Calculate average pairwise cosine similarity per cluster
            double totalCoherence = 0;
            foreach (var vectors in clusterGroups.Values)
            {
                if (vectors.Count < 2) continue;

                double simSum = 0;
                int pairCount = 0;
                for (int i = 0; i < vectors.Count; i++)
                {
                    for (int j = i + 1; j < vectors.Count; j++)
                    {
                        simSum += CosineSimilarity(vectors[i], vectors[j]);
                        pairCount++;
                    }
                }

                if (pairCount > 0)
                    totalCoherence += simSum / pairCount;
            }

            double avgCoherence = totalCoherence / clusterGroups.Count;
            Console.WriteLine($"k = {k} → Avg Coherence = {avgCoherence:F4}");

            if (avgCoherence > bestCoherence)
            {
                bestCoherence = avgCoherence;
                bestK = k;
            }
        }

        Console.WriteLine($"\n Best k = {bestK} with coherence score = {bestCoherence:F4}");
    }

    //public void EvaluateWordClusterCoherence(int minK = 2, int maxK = 10, int topNWords = 100)
    //{






    //    var context = new MLContext();

    //    // Build word × abstract binary presence vectors
    //    var abstracts = _cache.Values
    //        .Select(j => j["abstracts-retrieval-response"]?["coredata"]?["dc:description"]?.ToString())
    //        .Where(s => !string.IsNullOrWhiteSpace(s))
    //        .ToList();

    //    int vectorSize = abstracts.Count;

    //    var wordMap = new Dictionary<string, HashSet<int>>(StringComparer.OrdinalIgnoreCase);
    //    for (int i = 0; i < abstracts.Count; i++)
    //    {
    //        var tokens = abstracts[i].ToLower()
    //            .Split(new[] { ' ', '.', ',', ':', ';', '\"', '\'', '(', ')', '\n', '\r', '\t', '-', '_', '?' }, StringSplitOptions.RemoveEmptyEntries)
    //            .Where(w => w.Length > 3)
    //            .Distinct();

    //        foreach (var word in tokens)
    //        {
    //            if (!wordMap.ContainsKey(word))
    //                wordMap[word] = new HashSet<int>();
    //            wordMap[word].Add(i);
    //        }
    //    }

    //    var topWords = wordMap
    //        .OrderByDescending(kvp => kvp.Value.Count)
    //        .Take(topNWords)
    //        .ToList();

    //    var wordVectors = topWords.Select(kvp =>
    //        new WordVector
    //        {
    //            Word = kvp.Key,
    //            Vector = abstracts.Select((_, i) => kvp.Value.Contains(i) ? 1f : 0f).ToArray()
    //        }).ToList();

    //    var dataView = context.Data.LoadFromEnumerable(wordVectors);

    //    // Manually specify schema
    //    var schema = SchemaDefinition.Create(typeof(WordVector));
    //    schema[nameof(WordVector.Vector)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, vectorSize);

    //    var typedDataView = context.Data.LoadFromEnumerable(wordVectors, schema);

    //    //var dataView = context.Data.LoadFromEnumerable(wordVectors);
    //    var bestScore = double.MinValue;
    //    var bestK = -1;

    //    for (int k = minK; k <= maxK; k++)
    //    {
    //        var options = new Microsoft.ML.Trainers.KMeansTrainer.Options
    //        {
    //            NumberOfClusters = k,
    //            FeatureColumnName = nameof(WordVector.Vector)
    //        };

    //        var model = context.Clustering.Trainers.KMeans(options).Fit(typedDataView);
    //        var predictions = model.Transform(typedDataView);
    //        var clusterResults = context.Data.CreateEnumerable<WordClusterResult>(predictions, reuseRowObject: false).ToList();

    //        var clusters = new Dictionary<uint, List<float[]>>();
    //        for (int i = 0; i < wordVectors.Count; i++)
    //        {
    //            uint cid = clusterResults[i].PredictedClusterId;
    //            if (!clusters.ContainsKey(cid))
    //                clusters[cid] = new List<float[]>();
    //            clusters[cid].Add(wordVectors[i].Vector);
    //        }

    //        double totalSim = 0;
    //        int validClusters = 0;

    //        foreach (var cluster in clusters.Values)
    //        {
    //            if (cluster.Count < 2) continue;

    //            double simSum = 0;
    //            int pairCount = 0;

    //            for (int i = 0; i < cluster.Count; i++)
    //            {
    //                for (int j = i + 1; j < cluster.Count; j++)
    //                {
    //                    simSum += CosineSimilarity(cluster[i], cluster[j]);
    //                    pairCount++;
    //                }
    //            }

    //            if (pairCount > 0)
    //            {
    //                totalSim += simSum / pairCount;
    //                validClusters++;
    //            }
    //        }

    //        double avgCoherence = validClusters > 0 ? totalSim / validClusters : 0;
    //        Console.WriteLine($"k = {k}: average cluster coherence = {avgCoherence:F4}");

    //        if (avgCoherence > bestScore)
    //        {
    //            bestScore = avgCoherence;
    //            bestK = k;
    //        }
    //    }

    //    Console.WriteLine($"\n Best number of clusters: {bestK} with coherence {bestScore:F4}");

    //    // Step: Final clustering with bestK and output clusters
    //    if (bestK > 0)
    //    {
    //        Console.WriteLine($"\n--- Final Clustering for k = {bestK} ---");

    //        var finalOptions = new Microsoft.ML.Trainers.KMeansTrainer.Options
    //        {
    //            NumberOfClusters = bestK,
    //            FeatureColumnName = nameof(WordVector.Vector)
    //        };

    //        var finalModel = context.Clustering.Trainers.KMeans(finalOptions).Fit(typedDataView);
    //        var finalPredictions = finalModel.Transform(typedDataView);
    //        var finalClusterResults = context.Data.CreateEnumerable<WordClusterResult>(finalPredictions, reuseRowObject: false).ToList();

    //        var finalClusters = new Dictionary<uint, List<string>>();
    //        for (int i = 0; i < wordVectors.Count; i++)
    //        {
    //            uint clusterId = finalClusterResults[i].PredictedClusterId;
    //            if (!finalClusters.ContainsKey(clusterId))
    //                finalClusters[clusterId] = new List<string>();

    //            finalClusters[clusterId].Add(wordVectors[i].Word);
    //        }

    //        var clusterCsvPath = Path.Combine(_outputDirectory, "best_word_clusters.csv");
    //        using (var writer = new StreamWriter(clusterCsvPath))
    //        {
    //            writer.WriteLine("Cluster,Words");

    //            foreach (var cluster in finalClusters.OrderBy(c => c.Key))
    //            {
    //                var words = string.Join(", ", cluster.Value.OrderBy(w => w));
    //                Console.WriteLine($"Cluster {cluster.Key}: {words}");
    //                writer.WriteLine($"{cluster.Key},\"{words}\"");
    //            }
    //        }

    //        Console.WriteLine("CSV saved to: " + clusterCsvPath);
    //    }
    //}

    public void EvaluateWordClusterCoherence(int minK = 2, int maxK = 10, int topNWords = 100)
    {
        var context = new MLContext();

        // Step 1: Extract all abstracts
        var abstracts = _cache.Values
            .Select(j => j["abstracts-retrieval-response"]?["coredata"]?["dc:description"]?.ToString())
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .ToList();

        if (abstracts.Count == 0)
        {
            Console.WriteLine("No abstracts available.");
            return;
        }

        int vectorSize = abstracts.Count;

        // Step 2: Build keyword → abstract map using your keyword extractor
        var keywordMap = new Dictionary<string, HashSet<int>>(StringComparer.OrdinalIgnoreCase);
        var keywordCount = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < abstracts.Count; i++)
        {
            var article = _cache.Values.ElementAt(i);
            var keywords = ExtractKeywordsFromArticle(article);

            foreach (var keyword in keywords.Distinct(StringComparer.OrdinalIgnoreCase))
            {
                if (!keywordMap.ContainsKey(keyword))
                    keywordMap[keyword] = new HashSet<int>();

                keywordMap[keyword].Add(i);
                keywordCount.TryAdd(keyword, 0);
                keywordCount[keyword]++;
            }
        }

        // Step 3: Select top keywords
        var topKeywords = keywordCount
            .OrderByDescending(kvp => kvp.Value)
            .Take(topNWords)
            .Select(kvp => kvp.Key)
            .ToList();

        // Step 4: Build TF-IDF word vectors
        int totalDocs = abstracts.Count;
        var wordVectors = topKeywords.Select(keyword =>
        {
            float[] vector = new float[totalDocs];
            var docsWithKeyword = keywordMap[keyword];
            int docFreq = docsWithKeyword.Count;
            float idf = (float)Math.Log((double)(1 + totalDocs) / (1 + docFreq));

            foreach (int docIndex in docsWithKeyword)
            {
                // Term frequency: number of times the keyword appears in the abstract
                int tf = abstracts[docIndex]
                    .Split(new[] { ' ', '.', ',', ':', ';', '\"', '\'', '(', ')', '\n', '\r', '\t', '-', '_', '?' },
                           StringSplitOptions.RemoveEmptyEntries)
                    .Count(w => w.Equals(keyword, StringComparison.OrdinalIgnoreCase));

                vector[docIndex] = tf * idf;
            }

            return new WordVector
            {
                Word = keyword,
                Vector = vector
            };
        }).ToList();

        // Step 5: Prepare ML.NET typed data view
        var schema = SchemaDefinition.Create(typeof(WordVector));
        schema[nameof(WordVector.Vector)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, vectorSize);
        var dataView = context.Data.LoadFromEnumerable(wordVectors, schema);

        double bestScore = double.MinValue;
        int bestK = -1;

        for (int k = minK; k <= maxK; k++)
        {
            var options = new Microsoft.ML.Trainers.KMeansTrainer.Options
            {
                NumberOfClusters = k,
                FeatureColumnName = nameof(WordVector.Vector)
            };

            var model = context.Clustering.Trainers.KMeans(options).Fit(dataView);
            var predictions = model.Transform(dataView);
            var clusterResults = context.Data.CreateEnumerable<WordClusterResult>(predictions, reuseRowObject: false).ToList();

            var clusters = new Dictionary<uint, List<float[]>>();
            for (int i = 0; i < wordVectors.Count; i++)
            {
                uint clusterId = clusterResults[i].PredictedClusterId;
                if (!clusters.ContainsKey(clusterId))
                    clusters[clusterId] = new List<float[]>();
                clusters[clusterId].Add(wordVectors[i].Vector);
            }

            double totalSim = 0;
            int validClusters = 0;

            foreach (var cluster in clusters.Values)
            {
                if (cluster.Count < 2) continue;

                double simSum = 0;
                int pairCount = 0;

                for (int i = 0; i < cluster.Count; i++)
                {
                    for (int j = i + 1; j < cluster.Count; j++)
                    {
                        simSum += CosineSimilarity(cluster[i], cluster[j]);
                        pairCount++;
                    }
                }

                if (pairCount > 0)
                {
                    totalSim += simSum / pairCount;
                    validClusters++;
                }
            }

            double avgCoherence = validClusters > 0 ? totalSim / validClusters : 0;
            Console.WriteLine($"k = {k}: average cluster coherence = {avgCoherence:F4}");

            if (avgCoherence > bestScore)
            {
                bestScore = avgCoherence;
                bestK = k;
            }
        }

        Console.WriteLine($"\n✅ Best number of clusters: {bestK} with coherence {bestScore:F4}");

        // Step 6: Final clustering with bestK
        if (bestK > 0)
        {
            var finalOptions = new Microsoft.ML.Trainers.KMeansTrainer.Options
            {
                NumberOfClusters = bestK,
                FeatureColumnName = nameof(WordVector.Vector)
            };

            var finalModel = context.Clustering.Trainers.KMeans(finalOptions).Fit(dataView);
            var finalPredictions = finalModel.Transform(dataView);
            var finalResults = context.Data.CreateEnumerable<WordClusterResult>(finalPredictions, reuseRowObject: false).ToList();

            var finalClusters = new Dictionary<uint, List<string>>();
            for (int i = 0; i < wordVectors.Count; i++)
            {
                uint clusterId = finalResults[i].PredictedClusterId;
                if (!finalClusters.ContainsKey(clusterId))
                    finalClusters[clusterId] = new List<string>();
                finalClusters[clusterId].Add(wordVectors[i].Word);
            }

            var csvPath = Path.Combine(_outputDirectory, "best_word_clusters.csv");
            using (var writer = new StreamWriter(csvPath))
            {
                writer.WriteLine("Cluster,Words");
                foreach (var kvp in finalClusters.OrderBy(k => k.Key))
                {
                    var words = string.Join(", ", kvp.Value.OrderBy(w => w));
                    Console.WriteLine($"Cluster {kvp.Key}: {words}");
                    writer.WriteLine($"{kvp.Key},\"{words}\"");
                }
            }

            Console.WriteLine("CSV saved to: " + csvPath);
        }
    }


    //public void EvaluateWordClusterCoherenceo(int minK = 2, int maxK = 10, int topNWords = 100)
    //{
    //    var context = new MLContext();

    //    // Step 1: Extract abstracts
    //    var abstracts = _cache.Values
    //        .Select(j => j["abstracts-retrieval-response"]?["coredata"]?["dc:description"]?.ToString())
    //        .Where(s => !string.IsNullOrWhiteSpace(s))
    //        .ToList();

    //    if (abstracts.Count == 0)
    //    {
    //        Console.WriteLine("No abstracts available.");
    //        return;
    //    }

    //    int vectorSize = abstracts.Count;

    //    // Step 2: Extract keywords from abstracts using your method
    //    var keywordMap = new Dictionary<string, HashSet<int>>(StringComparer.OrdinalIgnoreCase);
    //    var keywordCount = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

    //    for (int i = 0; i < abstracts.Count; i++)
    //    {
    //        var article = _cache.Values.ElementAt(i);
    //        var keywords = ExtractKeywordsFromArticle(article);

    //        foreach (var keyword in keywords.Distinct(StringComparer.OrdinalIgnoreCase))
    //        {
    //            if (!keywordMap.ContainsKey(keyword))
    //                keywordMap[keyword] = new HashSet<int>();

    //            keywordMap[keyword].Add(i);

    //            keywordCount.TryAdd(keyword, 0);
    //            keywordCount[keyword]++;
    //        }
    //    }

    //    // Step 3: Select top N keywords
    //    var topKeywords = keywordCount
    //        .OrderByDescending(kvp => kvp.Value)
    //        .Take(topNWords)
    //        .Select(kvp => kvp.Key)
    //        .ToList();

    //    var wordVectors = topKeywords.Select(keyword =>
    //        new WordVector
    //        {
    //            Word = keyword,
    //            Vector = abstracts.Select((_, i) => keywordMap[keyword].Contains(i) ? 1f : 0f).ToArray()
    //        }).ToList();

    //    var schema = SchemaDefinition.Create(typeof(WordVector));
    //    schema[nameof(WordVector.Vector)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, vectorSize);
    //    var dataView = context.Data.LoadFromEnumerable(wordVectors, schema);

    //    double bestScore = double.MinValue;
    //    int bestK = -1;

    //    for (int k = minK; k <= maxK; k++)
    //    {
    //        var options = new Microsoft.ML.Trainers.KMeansTrainer.Options
    //        {
    //            NumberOfClusters = k,
    //            FeatureColumnName = nameof(WordVector.Vector)
    //        };

    //        var model = context.Clustering.Trainers.KMeans(options).Fit(dataView);
    //        var predictions = model.Transform(dataView);
    //        var clusterResults = context.Data.CreateEnumerable<WordClusterResult>(predictions, reuseRowObject: false).ToList();

    //        var clusters = new Dictionary<uint, List<float[]>>();
    //        for (int i = 0; i < wordVectors.Count; i++)
    //        {
    //            uint clusterId = clusterResults[i].PredictedClusterId;
    //            if (!clusters.ContainsKey(clusterId))
    //                clusters[clusterId] = new List<float[]>();
    //            clusters[clusterId].Add(wordVectors[i].Vector);
    //        }

    //        double totalSim = 0;
    //        int validClusters = 0;

    //        foreach (var cluster in clusters.Values)
    //        {
    //            if (cluster.Count < 2) continue;

    //            double simSum = 0;
    //            int pairCount = 0;

    //            for (int i = 0; i < cluster.Count; i++)
    //            {
    //                for (int j = i + 1; j < cluster.Count; j++)
    //                {
    //                    simSum += CosineSimilarity(cluster[i], cluster[j]);
    //                    pairCount++;
    //                }
    //            }

    //            if (pairCount > 0)
    //            {
    //                totalSim += simSum / pairCount;
    //                validClusters++;
    //            }
    //        }

    //        double avgCoherence = validClusters > 0 ? totalSim / validClusters : 0;
    //        Console.WriteLine($"k = {k}: average cluster coherence = {avgCoherence:F4}");

    //        if (avgCoherence > bestScore)
    //        {
    //            bestScore = avgCoherence;
    //            bestK = k;
    //        }
    //    }

    //    Console.WriteLine($"\n✅ Best number of clusters: {bestK} with coherence {bestScore:F4}");

    //    // Step 4: Final clustering using bestK
    //    if (bestK > 0)
    //    {
    //        var options = new Microsoft.ML.Trainers.KMeansTrainer.Options
    //        {
    //            NumberOfClusters = bestK,
    //            FeatureColumnName = nameof(WordVector.Vector)
    //        };

    //        var finalModel = context.Clustering.Trainers.KMeans(options).Fit(dataView);
    //        var finalPredictions = finalModel.Transform(dataView);
    //        var finalResults = context.Data.CreateEnumerable<WordClusterResult>(finalPredictions, reuseRowObject: false).ToList();

    //        var finalClusters = new Dictionary<uint, List<string>>();
    //        for (int i = 0; i < wordVectors.Count; i++)
    //        {
    //            uint clusterId = finalResults[i].PredictedClusterId;
    //            if (!finalClusters.ContainsKey(clusterId))
    //                finalClusters[clusterId] = new List<string>();
    //            finalClusters[clusterId].Add(wordVectors[i].Word);
    //        }

    //        // Output results to console and CSV
    //        var csvPath = Path.Combine(_outputDirectory, "best_word_clusters.csv");
    //        using (var writer = new StreamWriter(csvPath))
    //        {
    //            writer.WriteLine("Cluster,Words");
    //            foreach (var kvp in finalClusters.OrderBy(k => k.Key))
    //            {
    //                var words = string.Join(", ", kvp.Value.OrderBy(w => w));
    //                Console.WriteLine($"Cluster {kvp.Key}: {words}");
    //                writer.WriteLine($"{kvp.Key},\"{words}\"");
    //            }
    //        }

    //        Console.WriteLine("CSV saved to: " + csvPath);
    //    }
    //}


    public static double CosineSimilarity(float[] v1, float[] v2)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < v1.Length; i++)
        {
            dot += v1[i] * v2[i];
            normA += v1[i] * v1[i];
            normB += v2[i] * v2[i];
        }
        return normA == 0 || normB == 0 ? 0 : dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
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

    //public Dictionary<int, int> GetArticleCountByYear()
    //{
    //    var result = new Dictionary<int, int>();

    //    foreach (var article in _cache.Values)
    //    {
    //        var dateStr = article["abstracts-retrieval-response"]?["coredata"]?["prism:coverDate"]?.ToString();
    //        if (DateTime.TryParse(dateStr, out var date))
    //        {
    //            int year = date.Year;
    //            result.TryAdd(year, 0);
    //            result[year]++;
    //        }
    //    }

    //    Console.WriteLine("\n--- Article Count by Year ---");
    //    foreach (var kvp in result.OrderBy(k => k.Key))
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value}");

    //    return result;
    //}

    private void SaveToCsv(string filename, IEnumerable<string> lines)
    {
        var path = Path.Combine(_outputDirectory, filename);
        File.WriteAllLines(path, lines);
        Console.WriteLine("CSV saved to: " + path);
    }

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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "article_count_by_year.csv");
        var lines = new List<string> { "Year,Count" };
        lines.AddRange(result.OrderBy(k => k.Key).Select(kvp => $"{kvp.Key},{kvp.Value}"));
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }


    //public Dictionary<string, string> GetKeywordTrendsAsDelimitedStrings()
    //{
    //    var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
    //    var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
    //    var years = new SortedSet<int>();

    //    foreach (var article in _cache.Values)
    //    {
    //        var core = article["abstracts-retrieval-response"]?["coredata"];
    //        var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];
    //        var keywords = ExtractKeywordsFromArticle(article);

    //        var dateStr = core?["prism:coverDate"]?.ToString();
    //        if (!DateTime.TryParse(dateStr, out var date))
    //            continue;

    //        int year = date.Year;
    //        years.Add(year);

    //        foreach (var keyword in keywords.Distinct(StringComparer.OrdinalIgnoreCase))
    //        {
    //            if (!keywordYearCount.ContainsKey(keyword))
    //                keywordYearCount[keyword] = new Dictionary<int, int>();

    //            if (!keywordYearCount[keyword].ContainsKey(year))
    //                keywordYearCount[keyword][year] = 0;

    //            keywordYearCount[keyword][year]++;
    //            keywordTotals.TryAdd(keyword, 0);
    //            keywordTotals[keyword]++;
    //        }
    //    }

    //    var topKeywords = keywordTotals
    //        .OrderByDescending(kvp => kvp.Value)
    //        .Take(20)
    //        .Select(kvp => kvp.Key)
    //        .ToList();

    //    var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    //    foreach (var keyword in topKeywords)
    //    {
    //        var counts = years.Select(y => keywordYearCount[keyword].TryGetValue(y, out int c) ? c : 0);
    //        result[keyword] = string.Join(",", counts);
    //    }

    //    Console.WriteLine("\n--- Keyword Trends (CSV-style lines) ---");
    //    foreach (var kvp in result)
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value}");

    //    return result;
    //}

    public Dictionary<string, string> GetKeywordTrendsAsDelimitedStrings()
    {
        var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var years = new SortedSet<int>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
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



    //public Dictionary<string, string> GetKeywordTrendRatiosByYear()
    //{
    //    var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
    //    var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
    //    var yearArticleCount = new Dictionary<int, int>();
    //    var years = new SortedSet<int>();

    //    foreach (var article in _cache.Values)
    //    {
    //        var core = article["abstracts-retrieval-response"]?["coredata"];
    //        var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];
    //        var keywords = ExtractKeywordsFromArticle(article);

    //        var dateStr = core?["prism:coverDate"]?.ToString();
    //        if (!DateTime.TryParse(dateStr, out var date))
    //            continue;

    //        int year = date.Year;
    //        years.Add(year);
    //        yearArticleCount.TryAdd(year, 0);
    //        yearArticleCount[year]++;

    //        foreach (var keyword in keywords.Distinct(StringComparer.OrdinalIgnoreCase))
    //        {
    //            if (!keywordYearCount.ContainsKey(keyword))
    //                keywordYearCount[keyword] = new Dictionary<int, int>();

    //            if (!keywordYearCount[keyword].ContainsKey(year))
    //                keywordYearCount[keyword][year] = 0;

    //            keywordYearCount[keyword][year]++;
    //            keywordTotals.TryAdd(keyword, 0);
    //            keywordTotals[keyword]++;
    //        }
    //    }

    //    var topKeywords = keywordTotals
    //        .OrderByDescending(kvp => kvp.Value)
    //        .Take(20)
    //        .Select(kvp => kvp.Key)
    //        .ToList();

    //    var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    //    foreach (var keyword in topKeywords)
    //    {
    //        var ratios = years.Select(y =>
    //        {
    //            int count = keywordYearCount[keyword].TryGetValue(y, out int c) ? c : 0;
    //            int total = yearArticleCount.TryGetValue(y, out int t) ? t : 1;
    //            return ((double)count / total).ToString("0.000");
    //        });

    //        result[keyword] = string.Join(",", ratios);
    //    }

    //    Console.WriteLine("\n--- Keyword Ratio Trends (CSV-style lines) ---");
    //    foreach (var kvp in result)
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value}");

    //    return result;
    //}

    public Dictionary<string, string> GetKeywordTrendRatiosByYear()
    {
        var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var yearArticleCount = new Dictionary<int, int>();
        var years = new SortedSet<int>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
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



    //public Dictionary<string, string> GetSubjectTrendRatiosByYear()
    //{
    //    var subjectYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
    //    var subjectTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
    //    var yearArticleCount = new Dictionary<int, int>();
    //    var years = new SortedSet<int>();

    //    foreach (var article in _cache.Values)
    //    {
    //        var core = article["abstracts-retrieval-response"]?["coredata"];
    //        var dateStr = core?["prism:coverDate"]?.ToString();
    //        if (!DateTime.TryParse(dateStr, out var date))
    //            continue;

    //        int year = date.Year;
    //        years.Add(year);
    //        yearArticleCount.TryAdd(year, 0);
    //        yearArticleCount[year]++;

    //        var subjectToken = article["abstracts-retrieval-response"]?["subject-areas"]?["subject-area"];
    //        var subjects = new List<string>();

    //        if (subjectToken is JArray subjectArray)
    //        {
    //            subjects.AddRange(subjectArray
    //                .Select(s => s["$"]?.ToString()?.Trim())
    //                .Where(s => !string.IsNullOrEmpty(s)));
    //        }
    //        else if (subjectToken is JObject single)
    //        {
    //            var s = single["$"]?.ToString()?.Trim();
    //            if (!string.IsNullOrEmpty(s))
    //                subjects.Add(s);
    //        }

    //        foreach (var subject in subjects.Distinct())
    //        {
    //            if (!subjectYearCount.ContainsKey(subject))
    //                subjectYearCount[subject] = new Dictionary<int, int>();

    //            if (!subjectYearCount[subject].ContainsKey(year))
    //                subjectYearCount[subject][year] = 0;

    //            subjectYearCount[subject][year]++;
    //            subjectTotals.TryAdd(subject, 0);
    //            subjectTotals[subject]++;
    //        }
    //    }

    //    var topSubjects = subjectTotals
    //        .OrderByDescending(kvp => kvp.Value)
    //        .Take(20)
    //        .Select(kvp => kvp.Key)
    //        .ToList();

    //    var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    //    foreach (var subject in topSubjects)
    //    {
    //        var ratios = years.Select(y =>
    //        {
    //            int count = subjectYearCount[subject].TryGetValue(y, out int c) ? c : 0;
    //            int total = yearArticleCount.TryGetValue(y, out int t) ? t : 1;
    //            return ((double)count / total).ToString("0.000");
    //        });

    //        result[subject] = string.Join(",", ratios);
    //    }

    //    Console.WriteLine("\n--- Subject Ratio Trends (CSV-style lines) ---");
    //    foreach (var kvp in result)
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value}");

    //    return result;
    //}

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
                            if (!string.IsNullOrEmpty(term) && !term.Contains(","))
                                keywords.Add(term.ToLowerInvariant());
                        }
                    }
                    else if (mainterms.Type == JTokenType.Object)
                    {
                        var term = mainterms["$"]?.ToString()?.Trim();
                        if (!string.IsNullOrEmpty(term) && !term.Contains(","))
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

        var stopWords = new HashSet<string>(
    System.IO.File.ReadAllLines(_outputDirectory + "//..//stopwords.txt")
        .Where(line => !string.IsNullOrWhiteSpace(line))
        .Select(word => word.Trim().ToLowerInvariant()),
        StringComparer.OrdinalIgnoreCase);

        return keywords
            .Where(k => !stopWords.Contains(k))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        //var stopWords = new HashSet<string>(System.IO.File.ReadAllLines("stopwords.txt"));
        //return keywords.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }


    //public Dictionary<string, string> GetSubjectTrendsAsDelimitedStrings()
    //{
    //    var subjectYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
    //    var subjectTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
    //    var years = new SortedSet<int>();

    //    foreach (var article in _cache.Values)
    //    {
    //        var subjectToken = article["abstracts-retrieval-response"]?["subject-areas"]?["subject-area"];
    //        var core = article["abstracts-retrieval-response"]?["coredata"];
    //        var dateStr = core?["prism:coverDate"]?.ToString();
    //        if (!DateTime.TryParse(dateStr, out var date))
    //            continue;

    //        int year = date.Year;
    //        years.Add(year);

    //        var subjects = new List<string>();
    //        if (subjectToken is JArray subjectArray)
    //        {
    //            subjects.AddRange(subjectArray
    //                .Select(s => s["$"]?.ToString()?.Trim())
    //                .Where(s => !string.IsNullOrEmpty(s)));
    //        }
    //        else if (subjectToken is JObject single)
    //        {
    //            var s = single["$"]?.ToString()?.Trim();
    //            if (!string.IsNullOrEmpty(s))
    //                subjects.Add(s);
    //        }

    //        foreach (var subject in subjects.Distinct())
    //        {
    //            if (!subjectYearCount.ContainsKey(subject))
    //                subjectYearCount[subject] = new Dictionary<int, int>();

    //            if (!subjectYearCount[subject].ContainsKey(year))
    //                subjectYearCount[subject][year] = 0;

    //            subjectYearCount[subject][year]++;
    //            subjectTotals.TryAdd(subject, 0);
    //            subjectTotals[subject]++;
    //        }
    //    }

    //    var topSubjects = subjectTotals
    //        .OrderByDescending(kvp => kvp.Value)
    //        .Take(20)
    //        .Select(kvp => kvp.Key)
    //        .ToList();

    //    var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    //    foreach (var subject in topSubjects)
    //    {
    //        var counts = years.Select(y => subjectYearCount[subject].TryGetValue(y, out int c) ? c : 0);
    //        result[subject] = string.Join(",", counts);
    //    }

    //    Console.WriteLine("\n--- Subject Trends (CSV-style lines) ---");
    //    foreach (var kvp in result)
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value}");

    //    return result;
    //}

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



    //public Dictionary<string, int> GetArticleCountBySubjectArea()
    //{
    //    var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

    //    foreach (var article in _cache.Values)
    //    {
    //        var subjects = article["abstracts-retrieval-response"]?["subject-areas"]?["subject-area"];
    //        if (subjects == null) continue;

    //        if (subjects is JArray subjectArray)
    //        {
    //            foreach (var s in subjectArray)
    //            {
    //                string subject = s["$"]?.ToString()?.Trim();
    //                if (!string.IsNullOrEmpty(subject))
    //                {
    //                    result.TryAdd(subject, 0);
    //                    result[subject]++;
    //                }
    //            }
    //        }
    //        else if (subjects is JObject singleSubject)
    //        {
    //            string subject = singleSubject["$"]?.ToString()?.Trim().Replace(" (all)","");
    //            if (!string.IsNullOrEmpty(subject))
    //            {
    //                result.TryAdd(subject, 0);
    //                result[subject]++;
    //            }
    //        }
    //    }

    //    Console.WriteLine("\n--- Article Count by Subject Area ---");
    //    foreach (var kvp in result.OrderByDescending(k => k.Value).Take(20))
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value}");

    //    return result;
    //}

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
                string subject = singleSubject["$"]?.ToString()?.Trim().Replace(" (all)", "");
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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "subject_area_count.csv");
        var lines = new List<string> { "Subject,Count" };
        lines.AddRange(result.OrderByDescending(kvp => kvp.Value).Select(kvp => $"{kvp.Key},{kvp.Value}"));
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }



    //public Dictionary<string, int> GetArticleCountByAuthor()
    //{
    //    var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

    //    foreach (var article in _cache.Values)
    //    {
    //        var authorsRoot = article["abstracts-retrieval-response"]?["authors"];

    //        if (authorsRoot == null || authorsRoot.Type != JTokenType.Object)
    //            continue;

    //        var authorToken = authorsRoot["author"];
    //        if (authorToken == null)
    //            continue;

    //        // If it's an array of authors
    //        if (authorToken.Type == JTokenType.Array)
    //        {
    //            foreach (var author in authorToken)
    //            {
    //                string name = author["ce:indexed-name"]?.ToString()?.Trim();
    //                if (!string.IsNullOrEmpty(name))
    //                {
    //                    result.TryAdd(name, 0);
    //                    result[name]++;
    //                }
    //            }
    //        }
    //        // If it's a single author object
    //        else if (authorToken.Type == JTokenType.Object)
    //        {
    //            string name = authorToken["ce:indexed-name"]?.ToString()?.Trim();
    //            if (!string.IsNullOrEmpty(name))
    //            {
    //                result.TryAdd(name, 0);
    //                result[name]++;
    //            }
    //        }
    //    }

    //    Console.WriteLine("\n--- Article Count by Author ---");
    //    foreach (var kvp in result.OrderByDescending(k => k.Value).Take(20))
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value}");

    //    return result;
    //}


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


    //public Dictionary<string, int> GetAuthorCitations()
    //{
    //    var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

    //    foreach (var article in _cache.Values)
    //    {
    //        var core = article["abstracts-retrieval-response"]?["coredata"];
    //        if (!int.TryParse(core?["citedby-count"]?.ToString(), out int citations))
    //            citations = 0;

    //        var authorsRoot = article["abstracts-retrieval-response"]?["authors"];
    //        if (authorsRoot == null || authorsRoot.Type != JTokenType.Object)
    //            continue;

    //        var authorToken = authorsRoot["author"];
    //        if (authorToken == null)
    //            continue;

    //        if (authorToken.Type == JTokenType.Array)
    //        {
    //            foreach (var author in authorToken)
    //            {
    //                string name = author["ce:indexed-name"]?.ToString()?.Trim();
    //                if (!string.IsNullOrEmpty(name))
    //                {
    //                    result.TryAdd(name, 0);
    //                    result[name] += citations;
    //                }
    //            }
    //        }
    //        else if (authorToken.Type == JTokenType.Object)
    //        {
    //            string name = authorToken["ce:indexed-name"]?.ToString()?.Trim();
    //            if (!string.IsNullOrEmpty(name))
    //            {
    //                result.TryAdd(name, 0);
    //                result[name] += citations;
    //            }
    //        }
    //    }

    //    Console.WriteLine("\n--- Total Citations by Author ---");
    //    foreach (var kvp in result.OrderByDescending(k => k.Value).Take(20))
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value}");

    //    return result;
    //}


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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "author_citations.csv");
        var lines = new List<string> { "Author,Citations" };
        lines.AddRange(result.OrderByDescending(kvp => kvp.Value).Select(kvp => $"{kvp.Key},{kvp.Value}"));
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }

    //
    //
    //


    //public Dictionary<string, Dictionary<int, int>> GetKeywordTrendsByYear()
    //{
    //    var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
    //    var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
    //    var years = new SortedSet<int>();

    //    foreach (var article in _cache.Values)
    //    {
    //        var core = article["abstracts-retrieval-response"]?["coredata"];
    //        var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];
    //        //var keywords = new List<string>();

    //        var keywords = ExtractKeywordsFromArticle(article);

    //        var dateStr = core?["prism:coverDate"]?.ToString();
    //        if (!DateTime.TryParse(dateStr, out var date))
    //            continue;

    //        int year = date.Year;
    //        years.Add(year);

    //        foreach (var keyword in keywords.Distinct(StringComparer.OrdinalIgnoreCase))
    //        {
    //            if (!keywordYearCount.ContainsKey(keyword))
    //                keywordYearCount[keyword] = new Dictionary<int, int>();

    //            if (!keywordYearCount[keyword].ContainsKey(year))
    //                keywordYearCount[keyword][year] = 0;

    //            keywordYearCount[keyword][year]++;
    //            keywordTotals.TryAdd(keyword, 0);
    //            keywordTotals[keyword]++;
    //        }
    //    }

    //    var topKeywords = keywordTotals
    //        .OrderByDescending(kvp => kvp.Value)
    //        .Take(20)
    //        .Select(kvp => kvp.Key)
    //        .ToList();

    //    var result = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
    //    foreach (var keyword in topKeywords)
    //    {
    //        result[keyword] = new Dictionary<int, int>();
    //        foreach (var year in years)
    //        {
    //            int count = keywordYearCount[keyword].TryGetValue(year, out var c) ? c : 0;
    //            result[keyword][year] = count;
    //        }
    //    }

    //    // Print to console
    //    Console.WriteLine("\n--- Keyword Trends by Year ---");

    //    // Header
    //    var header = "Keyword".PadRight(20) + string.Join(" ", years.Select(y => y.ToString().PadLeft(6)));
    //    Console.WriteLine(header);
    //    Console.WriteLine(new string('-', header.Length));

    //    foreach (var kvp in result)
    //    {
    //        string line = kvp.Key.PadRight(20);
    //        foreach (var year in years)
    //        {
    //            int count = kvp.Value.TryGetValue(year, out var c) ? c : 0;
    //            line += count.ToString().PadLeft(6);
    //        }
    //        Console.WriteLine(line);
    //    }

    //    return result;
    //}


    public Dictionary<string, Dictionary<int, int>> GetKeywordTrendsByYear()
    {
        var keywordYearCount = new Dictionary<string, Dictionary<int, int>>(StringComparer.OrdinalIgnoreCase);
        var keywordTotals = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var years = new SortedSet<int>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
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







    //public Dictionary<string, int> GetArticleCountByKeyword()
    //{
    //    var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

    //    foreach (var article in _cache.Values)
    //    {
    //        var core = article["abstracts-retrieval-response"]?["coredata"];
    //        var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];

    //        //var keywords = new List<string>();
    //        var keywords = ExtractKeywordsFromArticle(article);

    //        foreach (var keyword in keywords.Distinct())
    //        {
    //            result.TryAdd(keyword, 0);
    //            result[keyword]++;
    //        }
    //    }

    //    Console.WriteLine("\n--- Top Keywords by Count ---");
    //    foreach (var kvp in result.OrderByDescending(k => k.Value).Take(50))
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value}");

    //    return result;
    //}

    public Dictionary<string, int> GetArticleCountByKeyword()
    {
        var result = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _cache.Values)
        {
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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "keyword_count.csv");
        var lines = new List<string> { "Keyword,Count" };
        lines.AddRange(result.OrderByDescending(kvp => kvp.Value).Select(kvp => $"{kvp.Key},{kvp.Value}"));
        File.WriteAllLines(csvPath, lines);
        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }


    //public Dictionary<(int year, string keyword), int> GetArticleCountByYearAndKeyword()
    //{
    //    var result = new Dictionary<(int, string), int>();

    //    foreach (var article in _cache.Values)
    //    {
    //        var core = article["abstracts-retrieval-response"]?["coredata"];
    //        var kwNode = article["abstracts-retrieval-response"]?["authkeywords"];

    //        var dateStr = core?["prism:coverDate"]?.ToString();
    //        if (!DateTime.TryParse(dateStr, out var date))
    //            continue;

    //        int year = date.Year;
    //        //var keywords = new List<string>();
    //        var keywords = ExtractKeywordsFromArticle(article);

    //        foreach (var keyword in keywords.Distinct())
    //        {
    //            var key = (year, keyword);
    //            result.TryAdd(key, 0);
    //            result[key]++;
    //        }
    //    }

    //    Console.WriteLine("\n--- Keyword Count by Year (top 5 per year) ---");
    //    foreach (var group in result.GroupBy(kvp => kvp.Key.Item1).OrderBy(g => g.Key))
    //    {
    //        Console.WriteLine($"\n{group.Key}:");
    //        foreach (var kvp in group.OrderByDescending(kvp => kvp.Value).Take(5))
    //            Console.WriteLine($"  {kvp.Key.Item2}: {kvp.Value}");
    //    }

    //    return result;
    //}


    public Dictionary<(int year, string keyword), int> GetArticleCountByYearAndKeyword()
    {
        var result = new Dictionary<(int, string), int>();

        foreach (var article in _cache.Values)
        {
            var core = article["abstracts-retrieval-response"]?["coredata"];
            var dateStr = core?["prism:coverDate"]?.ToString();
            if (!DateTime.TryParse(dateStr, out var date))
                continue;

            int year = date.Year;
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

        // Save to CSV
        var csvPath = Path.Combine(_outputDirectory, "article_count_by_year_and_keyword.csv");
        var csvLines = new List<string> { "Year,Keyword,Count" };
        csvLines.AddRange(result.Select(kvp => $"{kvp.Key.Item1},{kvp.Key.Item2},{kvp.Value}"));
        File.WriteAllLines(csvPath, csvLines);
        Console.WriteLine("CSV saved to: " + csvPath);

        return result;
    }






    //public Dictionary<int, double> GetAverageCitationByYear()
    //{
    //    var totals = new Dictionary<int, (int citationSum, int count)>();

    //    foreach (var article in _cache.Values)
    //    {
    //        var core = article["abstracts-retrieval-response"]?["coredata"];
    //        var dateStr = core?["prism:coverDate"]?.ToString();

    //        if (!DateTime.TryParse(dateStr, out var date))
    //            continue;

    //        int year = date.Year;

    //        if (int.TryParse(core?["citedby-count"]?.ToString(), out var citations))
    //        {
    //            if (!totals.ContainsKey(year))
    //                totals[year] = (0, 0);

    //            totals[year] = (
    //                totals[year].citationSum + citations,
    //                totals[year].count + 1
    //            );
    //        }
    //    }

    //    var result = totals.ToDictionary(
    //        kvp => kvp.Key,
    //        kvp => kvp.Value.count > 0 ? (double)kvp.Value.citationSum / kvp.Value.count : 0.0
    //    );

    //    Console.WriteLine("\n--- Average Citations by Year ---");
    //    foreach (var kvp in result.OrderBy(k => k.Key))
    //        Console.WriteLine($"{kvp.Key}: {kvp.Value:F2}");

    //    return result;
    //}

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

    public void BuildCorpusModel()
    {

        var path = Path.Combine(_outputDirectory, "corpus.txt");
        var modelFilePath = Path.Combine(_outputDirectory, "model.bin");
        //CreateDirectory(modelFilePath);

        string corpusTrainingFilePath = path;
        //string modelFilePath = corpusTrainingFilePath + ".model.bin";
        var shouldBuildModel = false;

        if (!File.Exists(modelFilePath))
        {
            if (!File.Exists(corpusTrainingFilePath))
            {
                Console.WriteLine("Neither training file nor model file exist, exiting..");
                return;
            }

            Console.WriteLine("Model file does not exist, will construct it from the corpus..");
            shouldBuildModel = true;
        }
        else
        {
            if (!File.Exists(corpusTrainingFilePath))
            {
                Console.WriteLine("The model file exists, but the training file does not, so there is no option to rebuild - the previously-built model will be used");
                shouldBuildModel = false;
            }
            else
            {
                Console.Write("Both the training file and model file are present - does the model need to rebuilt? {Y/N} ");
                shouldBuildModel = Console.ReadKey().Key == ConsoleKey.Y;
                Console.WriteLine();
            }
        }
        Console.WriteLine();

        if (shouldBuildModel)
        {
            // Train vector model and save to file.
            var word2vec = new Trainer();
            word2vec.Train(corpusTrainingFilePath, modelFilePath, Normaliser.Normalise);
        }
    }





    public void ClusterCustomWord2VecModel(int dimensions = 100, int numClusters = 8, int topN = 100)
    {
        var context = new MLContext();

        var filePath = Path.Combine(_outputDirectory, "corpus.txt");
        var modelPath = Path.Combine(_outputDirectory, "model.bin");

        BuildCorpusModel();

        // Step 1: Load Word2Vec .bin vectors
        var allVectors = LoadWord2VecTxt(dimensions);

        // Step 2: Extract and rank keywords from your articles
        var keywordFreq = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        foreach (var article in _cache.Values)
        {
            var keywords = ExtractKeywordsFromArticle(article);
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
    }



    public void ExportCorpus()
    {
        var filePath = Path.Combine(_outputDirectory, "corpus.txt");
        var lines = _cache.Values
            .Select(j => j["abstracts-retrieval-response"]?["coredata"]?["dc:description"]?.ToString())
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .Select(text => string.Join(" ", text
                .ToLowerInvariant()
                .Split(new[] { ' ', '.', ',', ':', ';', '\"', '\'', '(', ')', '\n', '\r', '\t', '-', '_', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(token => token.Length > 3)))
            .ToList();

        File.WriteAllLines(filePath, lines);
        Console.WriteLine($"Corpus written to {filePath}");
    }

    public Dictionary<string, float[]> LoadWord2VecTxt(int dimensions)
    {
        var modelPath = Path.Combine(_outputDirectory, "model.txt");
        var vectors = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);

        using var reader = new StreamReader(modelPath);
        string? header = reader.ReadLine(); // e.g. "10000 75"

        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) continue;

            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != dimensions + 1) continue;

            string word = parts[0];

            float[] vector = new float[dimensions];
            for (int i = 0; i < dimensions; i++)
            {
                if (!float.TryParse(parts[i + 1], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float val))
                {
                    val = 0f;
                }
                vector[i] = val;
            }

            // Optional: normalize the vector
            float norm = (float)Math.Sqrt(vector.Sum(v => v * v));
            if (norm > 0)
            {
                for (int i = 0; i < vector.Length; i++)
                    vector[i] /= norm;
            }

            vectors[word] = vector;
        }

        Console.WriteLine($"✅ Loaded {vectors.Count} word vectors from {modelPath}");
        return vectors;
    }

    public Dictionary<string, float[]> LoadWord2VecBin(int dimensions)
    {
        var result = new Dictionary<string, float[]>(StringComparer.OrdinalIgnoreCase);
        var modelPath = Path.Combine(_outputDirectory, "model.txt");

        using (var fs = new FileStream(modelPath, FileMode.Open, FileAccess.Read))
        using (var reader = new BinaryReader(fs))
        {
            // Step 1: Read header line byte-by-byte until newline
            var headerBytes = new List<byte>();
            byte b;
            while ((b = reader.ReadByte()) != '\n')
            {
                headerBytes.Add(b);
            }
            string headerLine = Encoding.UTF8.GetString(headerBytes.ToArray());
            var parts = headerLine.Split(' ');
            int vocabSize = int.Parse(parts[0]);
            int vectorSize = int.Parse(parts[1]);

            // Step 2: Read word vectors
            for (int i = 0; i < vocabSize; i++)
            {
                // Read word until space
                var wordBytes = new List<byte>();
                int count = 0;
                while ((b = reader.ReadByte()) != ' ')
                {
                    wordBytes.Add(b);
                    count++;
                }
                string word = Encoding.UTF8.GetString(wordBytes.ToArray());

                // Read vector
                float[] vector = new float[vectorSize];
                for (int j = 0; j < vectorSize; j++)
                {
                    vector[j] = reader.ReadSingle();
                }

                // Normalize
                float norm = (float)Math.Sqrt(vector.Sum(v => v * v));
                if (norm > 0)
                    for (int j = 0; j < vector.Length; j++)
                        vector[j] /= norm;

                result[word] = vector;
            }
        }

        Console.WriteLine($"✅ Loaded {result.Count} vectors from model.bin");
        return result;
    }


    //public Dictionary<string, float[]> LoadWord2VecBin(int dimensions)
    //{
    //    var result = new Dictionary<string, float[]>();
    //    var modelPath = Path.Combine(_outputDirectory, "model.bin");

    //    using (var reader = new BinaryReader(File.OpenRead(modelPath)))
    //    {
    //        string header = "";
    //        char c;
    //        while ((c = reader.ReadChar()) != '\n')
    //            header += c;

    //        var parts = header.Split(' ');
    //        int vocabSize = int.Parse(parts[0]);
    //        int vectorSize = int.Parse(parts[1]);

    //        for (int i = 0; i < vocabSize; i++)
    //        {
    //            var sb = new StringBuilder();
    //            char ch;
    //            while ((ch = reader.ReadChar()) != ' ')
    //                sb.Append(ch);
    //            string word = sb.ToString();

    //            float[] vector = new float[vectorSize];
    //            for (int j = 0; j < vectorSize; j++)
    //                vector[j] = reader.ReadSingle();

    //            // Normalize the vector
    //            float norm = (float)Math.Sqrt(vector.Sum(v => v * v));
    //            if (norm > 0)
    //                for (int j = 0; j < vector.Length; j++)
    //                    vector[j] /= norm;

    //            result[word] = vector;
    //        }
    //    }

    //    Console.WriteLine($"Loaded {result.Count} word vectors.");
    //    return result;
    //}


}
