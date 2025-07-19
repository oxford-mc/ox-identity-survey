namespace Word2Vec;

// Code initially courtesy of https://web.archive.org/web/20181015004944/http://www.robosoup.com/2016/02/word2vec-lightweight-port-c.html
class Trainer
{
    private static readonly int MinCount = 10;
    private const float sample = 1e-3f;
    private const float starting_alpha = 0.05f; // Starting learning rate.
    private const int dimensions = 75;          // Word vector dimensions.
    private const int exp_table_size = 1000;
    private const int iter = 5;                 // Training iterations.
    private const int max_exp = 6;
    private const int negative = 5;             // Number of negative examples.
    private const int window = 5;               // Window size.
    private readonly Dictionary<string, float[]> syn0 = [];
    private readonly Dictionary<string, float[]> syn1 = [];
    private readonly Dictionary<string, int> vocab = [];
    private readonly float[] expTable = new float[exp_table_size];
    private long train_words = 0;
    private readonly Random rnd = new();
    private string[] roulette;

    public void Train(string train_file, string model_file, Func<string, string> normalise)
    {
        BuildExpTable();
        LearnVocab(train_file, normalise);
        InitVectors();
        InitUnigramTable();
        TrainModel(train_file, normalise);
        WriteVectorsToFile(model_file);
        WriteVectorsToTxt(Path.ChangeExtension(model_file, ".txt"));
    }

    private void BuildExpTable()
    {
        for (int i = 0; i < exp_table_size; i++)
        {
            expTable[i] = (float)Math.Exp((i / (double)exp_table_size * 2 - 1) * max_exp);
            expTable[i] = expTable[i] / (expTable[i] + 1);
        }
    }

    private void InitVectors()
    {
        foreach (var key in vocab.Keys)
        {
            syn0.Add(key, new float[dimensions]);
            syn1.Add(key, new float[dimensions]);
            for (int i = 0; i < dimensions; i++)
                syn0[key][i] = (float)rnd.NextDouble() - 0.5f;
        }
    }

    private void WriteVectorsToFile(string output_file)
    {
        using BinaryWriter bw = new(File.Open(output_file, FileMode.Create));
        bw.Write(vocab.Count);
        bw.Write(dimensions);
        foreach (var vec in syn0)
        {
            bw.Write(vec.Key);
            for (int i = 0; i < dimensions; i++)
                bw.Write(vec.Value[i]);
        }
    }

    public void WriteVectorsToTxt(string outputFile)
    {
        using var writer = new StreamWriter(outputFile);
        writer.WriteLine($"{syn0.Count} {dimensions}");

        foreach (var kvp in syn0)
        {
            string word = kvp.Key;
            float[] vector = kvp.Value;
            string vectorLine = string.Join(" ", vector.Select(f => f.ToString("R", System.Globalization.CultureInfo.InvariantCulture)));
            writer.WriteLine($"{word} {vectorLine}");
        }

        Console.WriteLine($"✅ Word2Vec text model written to: {outputFile}");
    }


    private void LearnVocab(string train_file, Func<string, string> normalise)
    {
        using (StreamReader sr = new(train_file))
        {
            string? line;
            while ((line = sr.ReadLine()) != null)
            {
                foreach (var word in line.Split(' ').Select(normalise))
                {
                    if (word.Length == 0) continue;
                    train_words++;
                    if (train_words % 100000 == 0) Console.WriteLine("{0}k words read", train_words / 1000);
                    if (!vocab.TryGetValue(word, out int value)) vocab.Add(word, 1);
                    else vocab[word] = ++value;
                }
            }
        }
        Console.WriteLine();
        var tmp = (from w in vocab
                   where w.Value < MinCount
                   select w.Key).ToList();
        foreach (var key in tmp)
            vocab.Remove(key);
        Console.WriteLine("Vocab size: {0}", vocab.Count);
    }

    private void InitUnigramTable()
    {
        List<string> tmp = [];
        foreach (var word in vocab)
        {
            int count = (int)Math.Pow(word.Value, 0.75);
            for (int i = 0; i < count; i++) tmp.Add(word.Key);
        }
        roulette = [.. tmp];
    }

    private void TrainModel(string train_file, Func<string, string> normalise)
    {
        float alpha = starting_alpha;
        float[] neu1 = new float[dimensions];
        float[] neu1e = new float[dimensions];
        int last_word_count = 0;
        int sentence_position = 0;
        int word_count = 0;
        List<string> sentence = [];
        long word_count_actual = 0;
        DateTime start = DateTime.Now;
        for (int local_iter = 0; local_iter < iter; local_iter++)
        {
            using StreamReader sr = new(train_file);
            while (true)
            {
                if (word_count - last_word_count > 10000)
                {
                    word_count_actual += word_count - last_word_count;
                    last_word_count = word_count;
                    int seconds = (int)(DateTime.Now - start).TotalSeconds + 1;
                    float prog = (float)word_count_actual * 100 / (iter * train_words);
                    float rate = (float)word_count_actual / seconds / 1000;
                    Console.WriteLine("Progress: {0:0.00}% \tWords/sec: {1:0.00}k", prog, rate);
                    alpha = starting_alpha * (1 - word_count_actual / (float)(iter * train_words + 1));
                    if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001f;
                }
                if (sentence.Count == 0)
                {
                    if (sr.EndOfStream)
                    {
                        word_count_actual = train_words * (local_iter + 1);
                        word_count = 0;
                        last_word_count = 0;
                        sentence.Clear();
                        break;
                    }
                    sentence.Clear();
                    sentence_position = 0;
                    string? line = sr.ReadLine();
                    if (line != null)
                    {
                        foreach (var key in line.Split(' ').Select(normalise))
                        {
                            if (key.Length == 0) continue;
                            if (!vocab.ContainsKey(key)) continue;
                            word_count++;
                            if (sample > 0)
                            {
                                double ran = (Math.Sqrt(vocab[key] / (sample * train_words)) + 1) * (sample * train_words) / vocab[key];
                                if (ran < rnd.NextDouble()) continue;
                            }
                            sentence.Add(key);
                        }
                    }
                }
                if (sentence.Count == 0) continue;
                string word = sentence[sentence_position];
                for (int i = 0; i < dimensions; i++) neu1[i] = 0;
                for (int i = 0; i < dimensions; i++) neu1e[i] = 0;
                int cw = 0;
                for (int w = 0; w < window * 2 + 1; w++)
                {
                    if (w != window)
                    {
                        int p = sentence_position - window + w;
                        if (p < 0) continue;
                        if (p >= sentence.Count) continue;
                        string last_word = sentence[p];
                        float[] tmp0 = syn0[last_word];
                        for (int i = 0; i < dimensions; i++) neu1[i] += tmp0[i];
                        cw++;
                    }
                }
                if (cw > 0)
                {
                    for (int i = 0; i < dimensions; i++) neu1[i] /= cw;
                    for (int w = 0; w < negative + 1; w++)
                    {
                        string target;
                        int label;
                        if (w == 0)
                        {
                            target = word;
                            label = 1;
                        }
                        else
                        {
                            target = roulette[rnd.Next(roulette.Length)];
                            if (target == word) continue;
                            label = 0;
                        }
                        float a = 0;
                        float g = 0;
                        float[] tmp1 = syn1[target];
                        for (int i = 0; i < dimensions; i++) a += neu1[i] * tmp1[i];
                        if (a > max_exp) g = (label - 1) * alpha;
                        else if (a < -max_exp) g = (label - 0) * alpha;
                        else g = (label - expTable[(int)((a + max_exp) * (exp_table_size / max_exp / 2))]) * alpha;
                        for (int i = 0; i < dimensions; i++) neu1e[i] += g * tmp1[i];
                        for (int i = 0; i < dimensions; i++) tmp1[i] += g * neu1[i];
                    }
                    for (int w = 0; w < window * 2 + 1; w++)
                    {
                        if (w != window)
                        {
                            int p = sentence_position - window + w;
                            if (p < 0) continue;
                            if (p >= sentence.Count) continue;
                            string last_word = sentence[p];
                            float[] tmp0 = syn0[last_word];
                            for (int i = 0; i < dimensions; i++) tmp0[i] += neu1e[i];
                        }
                    }
                }
                sentence_position++;
                if (sentence_position >= sentence.Count)
                {
                    sentence.Clear();
                    continue;
                }
            }
        }
        Console.WriteLine();
    }
}