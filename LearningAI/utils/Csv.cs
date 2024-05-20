using System.Globalization;
using CsvHelper;

namespace LearningAI.utils;

public class CsvHelper {
    public static List<CsvData> ReadCsv(string path) {
        using (var reader = new StreamReader("path\\to\\file.csv"))
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture)) {
            var records = csv.GetRecords<CsvData>();
            return records.ToList();
        }
    }

    public static void WriteCsv(string path, List<CsvData> data) {
        using var writer = new StreamWriter(path);
        using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
        csv.WriteRecords(data);
    }
}