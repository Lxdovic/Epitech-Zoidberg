namespace LearningAI.utils;

public class RoutineResult {
    public RoutineResult(Routine routine, float accuracy, float learningRate, float tpr, float fpr, float tnr,
        float fnr) {
        Routine = routine;
        Accuracy = accuracy;
        LearningRate = learningRate;
        Tpr = tpr;
        Fpr = fpr;
        Tnr = tnr;
        Fnr = fnr;
    }

    public Routine Routine { get; }
    public float Accuracy { get; }
    public float LearningRate { get; }
    public float Tpr { get; }
    public float Fpr { get; }
    public float Tnr { get; }
    public float Fnr { get; }

    public override string ToString() {
        return
            $"{Routine.GetType().Name} - Accuracy: {Accuracy} - Learning Rate: {LearningRate} - TPR: {Tpr} - FPR: {Fpr} - TNR: {Tnr} - FNR: {Fnr}";
    }
}