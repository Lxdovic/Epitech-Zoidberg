namespace LearningAI.utils;

public abstract class Routine {
    public Routine(int epochs) {
        Epochs = epochs;
    }

    public int Epochs { get; }
}