using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Scopus_Analysis.Helper
{
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
}
