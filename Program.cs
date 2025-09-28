using Scopus_Analysis.Helper;

internal class Program
{
    private async static Task Main(string[] args)
    {
        var option = "";
        var keyword = "digital identity"; //default term used in this research
        var year = "2005-2024"; // default year range used in this research
        var articlePath = @"C:\Development\Oxford\ox-identity-survey\articles.json";
        var articleManager = new ArticleManager(articlePath);

        Console.WriteLine("Scopus Datamining Console");
        while (option != "Q")
        {
            Console.WriteLine("--------------------------");
            Console.WriteLine("");
            Console.WriteLine("Options:");
            Console.WriteLine("");
            Console.WriteLine("Keyword: " + keyword + ", Year Range " + year);
            Console.WriteLine("--------------------------");
            Console.WriteLine("");
            Console.WriteLine("Choose an option:");
            Console.WriteLine("1: Analyse Clustering");
            Console.WriteLine("2: Analyse Article Counts");
            Console.WriteLine("3: Analyse Citations");
            Console.WriteLine("4: Analyse Keywords");
            Console.WriteLine("5: Analyse Subjects");
            Console.WriteLine("C: Create Corpus");
            Console.WriteLine("K: Enter key word for data mining");
            Console.WriteLine("Y: Enter year range for data mining");
            Console.WriteLine("M: Mine articles from Scopus");
            Console.WriteLine("L: Load articles from Scopus");
            Console.WriteLine("Q: Quit Console");

            option = Console.ReadLine().ToUpper();

            switch (option)
            {
                case "C":
                    NLPHelper.ExportCorpus(Path.Combine(articleManager._outputDirectory, "corpus.txt"), articleManager._cache, "abstracts-retrieval-response.coredata.dc:description");
                    NLPHelper.BuildCorpusModel(articleManager._outputDirectory);
                    break;
                case "1":
                    articleManager.ClusterCustomWord2VecModel(100, 8, 150);
                    break;
                case "2":
                    articleManager.GetArticleCountByYear();
                    articleManager.GetArticleCountByKeyword();
                    articleManager.GetArticleCountBySubjectArea();
                    articleManager.GetArticleCountByAuthor();
                    articleManager.GetArticleCountByAuthorBySubject();
                    articleManager.GetArticleCountByYearAndKeyword();   
                    break;
                case "3":
                    articleManager.GetAuthorCitations();
                    articleManager.GetAverageCitationByYear();
                    break;
                case "4":
                    articleManager.GetKeywordTrendsAsDelimitedStrings();
                    articleManager.GetKeywordTrendRatiosByYear();      
                    break;
                case "5":
                    articleManager.GetSubjectTrendsAsDelimitedStrings();
                    articleManager.GetSubjectTrendRatiosByYear();                    
                    break;
                case "6":
                    articleManager.GetTopArticlesPerSubject();
                    articleManager.ExportTopArticlesPerSubject(10);
                    break;



                case "K":
                    Console.Write("Enter keyword: ");
                    keyword = Console.ReadLine();
                    Console.WriteLine($"You entered: {keyword}");
                    break;
                case "Y":
                    Console.Write("Enter year range (e.g., 2023-2025): ");
                    year = Console.ReadLine();
                    Console.WriteLine($"You entered: {year}");
                    break;
                case "M":
                    var scopus = new ScopusHelper(articleManager);
                    Console.WriteLine($"Mining articles for keyword '{keyword}' in year range '{year}'...");
                    var abstractsByYear = await scopus.FetchAbstractsByYear("\"digital identity\"", 2023, 2025);
                    articleManager.Save();
                    // Here you would call the method to start mining articles based on the keyword and year range
                    break;
                case "Q":
                    Console.WriteLine("Exiting...");
                    break;
            }
        }

        
        

        //NLPHelper.ExportCorpus(Path.Combine(articleManager._outputDirectory, "corpus.txt"), articleManager._cache, "abstracts-retrieval-response.coredata.dc:description");
        
        //articleManager.ClusterCustomWord2VecModel(100, 8, 150);
        //articleManager.GetArticleCountByYear();
        //articleManager.GetArticleCountByKeyword();

        ///* Subject Related */
        //articleManager.GetArticleCountBySubjectArea();
        //articleManager.GetTopArticlesPerSubject();
        ////articleManager.ExportTopArticlesPerSubject(10);

        
        //articleManager.GetArticleCountByAuthor();
        //articleManager.GetAuthorCitations();
        //articleManager.GetKeywordTrendsAsDelimitedStrings();
        //articleManager.GetKeywordTrendRatiosByYear();
        //articleManager.GetSubjectTrendsAsDelimitedStrings();
        //articleManager.GetSubjectTrendRatiosByYear();
        //articleManager.GetAverageCitationByYear();
        //articleManager.GetArticleCountByYearAndKeyword();

        //Console.ReadLine();


        //var allAbstracts = abstractsByYear.SelectMany(kvp => kvp.Value).ToList();
        //var topTerms = scopus.ExtractTopTerms(allAbstracts, 30).Keys.ToList();

        //var longitudinal = scopus.LongitudinalTermAnalysis(abstractsByYear, topTerms);

        //foreach (var term in longitudinal)
        //{
        //    Console.WriteLine($"\nTerm: {term.Key}");
        //    foreach (var year in term.Value.Keys.OrderBy(y => y))
        //    {
        //        Console.WriteLine($"{year}: {term.Value[year]}");
        //    }
        //}
    }
}