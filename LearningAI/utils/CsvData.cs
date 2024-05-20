namespace LearningAI.utils;

public class CsvData {
    public CsvData(int id, string model, float accuracy, float tpr, float fpr, float tnr, float fnr) {
        Id = id;
        Model = model;
        Accuracy = accuracy;
        Tpr = tpr;
        Fpr = fpr;
        Tnr = tnr;
        Fnr = fnr;
    }

    public int Id { get; set; }
    public string Model { get; }
    public float Accuracy { get; set; }
    public float Tpr { get; set; }
    public float Fpr { get; set; }
    public float Tnr { get; set; }
    public float Fnr { get; set; }
}