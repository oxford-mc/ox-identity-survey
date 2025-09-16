using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Scopus_Analysis.Model
{
    public class Article
    {
        private JObject _ob { get; set; }
        private string _directory { get; set; } = @"C:\Development\Oxford\ox-identity-survey\data";

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
        System.IO.File.ReadAllLines(_directory + "//..//stopwords.txt")
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

        public Article(JObject ob)
        {
            _ob = ob;
        }

        public DateTime Published { get
            {
                var dateStr = _ob["abstracts-retrieval-response"]?["coredata"]?["prism:coverDate"]?.ToString();
                if (DateTime.TryParse(dateStr, out var date))
                {
                    return date;
                }
                else
                {
                    return DateTime.MinValue;
                }
            }
        }

        public List<string> Authors
        {
            get
            {
                var authors = new List<string>();
                var authorsRoot = _ob["abstracts-retrieval-response"]?["authors"];
                if (authorsRoot != null && authorsRoot.Type == JTokenType.Object)
                {
                    var authorToken = authorsRoot["author"];
                    if (authorToken != null)
                    {
                        if (authorToken.Type == JTokenType.Array)
                        {
                            foreach (var author in authorToken)
                            {
                                string name = author["ce:indexed-name"]?.ToString()?.Trim();
                                if (!string.IsNullOrEmpty(name))
                                    authors.Add(name);
                            }
                        }
                        else if (authorToken.Type == JTokenType.Object)
                        {
                            string name = authorToken["ce:indexed-name"]?.ToString()?.Trim();
                            if (!string.IsNullOrEmpty(name))
                                authors.Add(name);
                        }
                    }
                }
                return authors;
            }
        }

        public string Abstract
        {
            get
            {
                var core = _ob["abstracts-retrieval-response"]?["coredata"];
                return core["dc:description"]?.ToString()?.Trim() ?? "";
            }
        }


        public int Citations
        {
            get
            {
                var core = _ob["abstracts-retrieval-response"]?["coredata"];
                if (!int.TryParse(core?["citedby-count"]?.ToString(), out int citations))
                    citations = 0;
                return citations;
            }
        }

        public string Title
        {
            get
            {
                var core = _ob["abstracts-retrieval-response"]?["coredata"];

                var title = core?["dc:title"]?.ToString() ?? "Untitled";
                return title;
            }
        }

        public string PrimarySubject
        {
            get
            {
                var subjectToken = _ob["abstracts-retrieval-response"]?["subject-areas"]?["subject-area"];

                string? primarySubject = null;


                List<string> subjectList = new();

                if (subjectToken is JArray array)
                {
                    subjectList = array
                        .Select(s => s["$"]?.ToString()?.Trim())
                        .Where(s => !string.IsNullOrWhiteSpace(s))
                        .Select(s => Regex.Replace(s, @"\s*\(.*?\)", "").Trim())  // 👈 Clean here
                        .Distinct(StringComparer.OrdinalIgnoreCase)
                        .ToList();

                    primarySubject = SelectBestSubject(subjectList, Keywords);
                }
                else if (subjectToken is JObject single)
                {
                    var s = single["$"]?.ToString()?.Trim();
                    if (!string.IsNullOrWhiteSpace(s))
                    {
                        s = Regex.Replace(s, @"\s*\(.*?\)", "").Trim();  // 👈 Clean single subject too
                        subjectList.Add(s);
                        primarySubject = s;
                    }
                }

                return primarySubject;
            }
        }

        public string PrimaryAuthor
        {
            get
            {
                var authorsRoot = _ob["abstracts-retrieval-response"]?["authors"];
                if (authorsRoot == null || authorsRoot.Type != JTokenType.Object)
                    return "";

                var authorToken = authorsRoot["author"];
                if (authorToken == null)
                    return "";

                if (authorToken.Type == JTokenType.Array)
                {
                    foreach (var author in authorToken)
                    {
                        string name = author["ce:indexed-name"]?.ToString()?.Trim();
                        if (!string.IsNullOrEmpty(name))
                        {
                            return name;
                        }
                    }
                }
                else if (authorToken.Type == JTokenType.Object)
                {
                    string name = authorToken["ce:indexed-name"]?.ToString()?.Trim();
                    if (!string.IsNullOrEmpty(name))
                    {
                        return name;
                    }
                }

                return "";
            }
        }

        public List<string> Keywords { get
            {
                return ExtractKeywordsFromArticle(_ob);
            } 
        }

        public List<string> Subjects { get
            {
                var subjectToken = _ob["abstracts-retrieval-response"]?["subject-areas"]?["subject-area"];

                List<string> subjectList = new();

                if (subjectToken is JArray array)
                {
                    subjectList = array
                        .Select(s => s["$"]?.ToString()?.Trim())
                        .Where(s => !string.IsNullOrWhiteSpace(s))
                        .Select(s => Regex.Replace(s, @"\s*\(.*?\)", "").Trim())  // 👈 Clean here
                        .Distinct(StringComparer.OrdinalIgnoreCase)
                        .ToList();
                }
                else if (subjectToken is JObject single)
                {
                    var s = single["$"]?.ToString()?.Trim();
                    if (!string.IsNullOrWhiteSpace(s))
                    {
                        s = Regex.Replace(s, @"\s*\(.*?\)", "").Trim();  // 👈 Clean single subject too
                        subjectList.Add(s);
                    }
                }

                return subjectList;
            } 
        }

        private string? SelectBestSubject(List<string> subjectList, List<string> keywords)
        {
            if (subjectList == null || subjectList.Count == 0)
                return null;

            var excludedSubjects = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "Multidisciplinary",
            "Agricultural and Biological Sciences (miscellaneous)",
            "General",
            "General Medicine",
            "Miscellaneous"
        };

            // ✅ Step 1: Manual keyword-to-subject mapping
            var subjectKeywordMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            { "aadhaar", "Political Science and International Relations" },
            { "access control", "Security and Privacy" },
            { "application", "Computer Science Applications" },
            { "authentication", "Security and Privacy" },
            { "authorization", "Security and Privacy" },
            { "biometric", "Security and Privacy" },
            { "biometrics", "Security and Privacy" },
            { "ble", "Computer Networks and Communications" },
            { "blockchain", "Computer Science Applications" },
            { "blockchain attacks", "Security and Privacy" },
            { "classroom", "Education" },
            { "citizen", "Public Administration" },
            { "compliance", "Law" },
            { "communication", "Computer Networks and Communications" },
            { "cultural identity", "Cultural Studies" },
            { "data governance", "Information Systems and Management" },
            { "data privacy", "Security and Privacy" },
            { "data protection", "Law" },
            { "data sharing", "Information Systems" },
            { "cyber security", "Security and Privacy" },
            { "cybersecurity", "Security and Privacy" },
            { "digital city", "Geography, Planning and Development" },
            { "digital economy", "Business and International Management" },
            { "digital footprint", "Social Media" },
            { "digital identity", "Information Systems" },
            { "digital self", "Social Media" },
            { "digital signature", "Security and Privacy" },
            { "digital transformation", "Business and International Management" },
            { "digital wallet", "Computer Science" },
            { "distributed ledger", "Computer Science Applications" },
            { "e-government", "Public Administration" },
            { "e-learning", "Education" },
            { "ehr", "Health Informatics" },
            { "electronic health", "Health Informatics" },
            { "electronic health record", "Health Informatics" },
            { "ethics", "Philosophy" },
            { "facebook", "Social Media" },
            { "federated identity", "Theoretical Computer Science" },
            { "ga4gh", "Biochemistry, Genetics and Molecular Biology" },
            { "health", "Health Informatics" },
            { "health record", "Health Informatics" },
            { "healthcare", "Health Informatics" },
            { "human rights", "Law" },
            { "identity federation", "Theoretical Computer Science" },
            { "identity management", "Information Systems" },
            { "identity theft", "Law" },
            { "identity verification", "Security and Privacy" },
            { "indigenous", "History" },
            { "instagram", "Social Media" },
            { "integration architecture", "Information Systems" },
            { "internet of things", "Computer Science Applications" },
            { "iot", "Computer Science Applications" },
            { "learning", "Education" },
            { "legal", "Law" },
            { "legal framework", "Law" },
            { "low energy", "Computer Networks and Communications" },
            { "low power", "Computer Networks and Communications" },
            { "metadata", "Information Systems" },
            { "metaverse", "Human-Computer Interaction" },
            { "migration", "Political Science and International Relations" },
            { "mobile communications", "Computer Networks and Communications" },
            { "museums", "Cultural Studies" },
            { "network", "Computer Networks and Communications" },
            { "networked publics", "Social Media" },
            { "oauth", "Information Systems" },
            { "online identity", "Social Media" },
            { "ontology", "Theoretical Computer Science" },
            { "pedagogic", "Education" },
            { "phishing", "Security and Privacy" },
            { "platform", "Social Media" },
            { "platform ecosystem", "Information Systems" },
            { "platform ecosystems", "Management of Technology and Innovation" },
            { "platform governance", "Management of Technology and Innovation" },
            { "policy", "Public Policy" },
            { "population management", "Geography, Planning and Development" },
            { "privacy", "Security and Privacy" },
            { "privacy preserving", "Security and Privacy" },
            { "proof system", "Theoretical Computer Science" },
            { "public key infrastructure", "Security and Privacy" },
            { "regulation", "Law" },
            { "representation", "Cultural Studies" },
            { "resource utilization", "Geography, Planning and Development" },
            { "self-sovereign", "Computer Science Applications" },
            { "self-sovereign identity", "Information Systems" },
            { "service provider", "Information Systems" },
            { "smart contract", "Software" },
            { "social media", "Social Media" },
            { "social platforms", "Social Media" },
            { "sovereign", "Political Science and International Relations" },
            { "students", "Education" },
            { "surveillance", "Sociology and Political Science" },
            { "sybil attack", "Computer Science" },
            { "teachers", "Education" },
            { "ticketing", "Information Systems" },
            { "trust", "Public Policy" },
            { "twitter", "Social Media" },
            { "ultra-low power", "Computer Networks and Communications" },
            { "urban", "Urban Studies" },
            { "user-generated content", "Social Media" },
            { "user identity", "Human-Computer Interaction" },
            { "user-centric", "Human-Computer Interaction" },
            { "usability", "Security and Privacy" },
            { "verifiable credential", "Information Systems and Management" },
            { "virtual", "Human-Computer Interaction" },
            { "wallet", "Computer Science Applications" },
            { "wireless", "Computer Networks and Communications" },
            { "zero knowledge proof", "Theoretical Computer Science" },
            { "zero-knowledge", "Theoretical Computer Science" }
        };

            var matchedSubjectsFromPool = keywords
                .Where(k => subjectKeywordMap.ContainsKey(k))
                .Select(k => subjectKeywordMap[k])
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (matchedSubjectsFromPool.Any())
            {
                // Return the most frequent mapped subject (or first)
                return matchedSubjectsFromPool.First();
            }

            // ✅ Step 2: Fallback to automatic subject matching
            var filteredSubjects = subjectList
                .Where(s => !excludedSubjects.Contains(s))
                .ToList();

            if (filteredSubjects.Count == 0)
                return null;

            var keywordSet = new HashSet<string>(keywords.Select(k => k.ToLowerInvariant()));

            var bestMatch = filteredSubjects
                .Select(subject => new
                {
                    Subject = subject,
                    Words = subject.Split(' ', StringSplitOptions.RemoveEmptyEntries),
                })
                .Select(s => new
                {
                    s.Subject,
                    MatchScore = s.Words.Count(w => keywordSet.Contains(w.ToLowerInvariant()))
                })
                .OrderByDescending(x => x.MatchScore)
                .ThenBy(x => subjectList.IndexOf(x.Subject))
                .FirstOrDefault();

            return bestMatch?.Subject ?? filteredSubjects.First();
        }
    }
}
