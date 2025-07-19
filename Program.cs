internal class Program
{
    private async static Task Main(string[] args)
    {
        Console.WriteLine("Hello, World!");

        var articleManager = new ArticleManager(@"C:\Development\Oxford\ox-identity-survey\articles.json");
        var scopus = new ScopusHelper(articleManager);
        articleManager.GetArticleCountByYear();
        articleManager.GetArticleCountByKeyword();
        articleManager.GetArticleCountBySubjectArea();
        articleManager.GetArticleCountByAuthor();
        articleManager.GetAuthorCitations();
        articleManager.GetKeywordTrendsAsDelimitedStrings();
        articleManager.GetKeywordTrendRatiosByYear();
        articleManager.GetSubjectTrendsAsDelimitedStrings();
        articleManager.GetSubjectTrendRatiosByYear();
        articleManager.GetAverageCitationByYear();
        articleManager.GetArticleCountByYearAndKeyword();


        Console.ReadLine();
        var abstractsByYear = await scopus.FetchAbstractsByYear("\"digital identity\"", 2005, 2024);
        articleManager.Save();

        var allAbstracts = abstractsByYear.SelectMany(kvp => kvp.Value).ToList();
        var topTerms = scopus.ExtractTopTerms(allAbstracts, 30).Keys.ToList();

        var longitudinal = scopus.LongitudinalTermAnalysis(abstractsByYear, topTerms);

        foreach (var term in longitudinal)
        {
            Console.WriteLine($"\nTerm: {term.Key}");
            foreach (var year in term.Value.Keys.OrderBy(y => y))
            {
                Console.WriteLine($"{year}: {term.Value[year]}");
            }
        }
    }
}