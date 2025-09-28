using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Word2Vec;

namespace Scopus_Analysis.Helper
{
    public static class NLPHelper
    {
        public static void ExportCorpus(string filePath, Dictionary<string, JObject> _cache, string jsonPath)
        {
            var lines = _cache.Values
                .Select(j => j.SelectToken(jsonPath)?.ToString())
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .Select(text => string.Join(" ", text
                    .ToLowerInvariant()
                    .Split(new[] { ' ', '.', ',', ':', ';', '\"', '\'', '(', ')', '\n', '\r', '\t', '-', '_', '?' }, StringSplitOptions.RemoveEmptyEntries)
                    .Where(token => token.Length > 3)))
                .ToList();

            File.WriteAllLines(filePath, lines);
            Console.WriteLine($"Corpus written to {filePath}");
        }

        public static void BuildCorpusModel(string _outputDirectory)
        {

            var path = Path.Combine(_outputDirectory, "corpus.txt");
            var modelFilePath = Path.Combine(_outputDirectory, "model.bin");
            //CreateDirectory(modelFilePath);

            string corpusTrainingFilePath = path;
            //string modelFilePath = corpusTrainingFilePath + ".model.bin";
            var shouldBuildModel = true;

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


        public static Dictionary<string, float[]> LoadWord2VecTxt(string _outputDirectory, int dimensions)
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

        public static Dictionary<string, float[]> LoadWord2VecBin(string _outputDirectory, int dimensions)
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


    }
}
