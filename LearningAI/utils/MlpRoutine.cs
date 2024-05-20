namespace LearningAI.utils;

public class MlpRoutine : Routine {
    public MlpRoutine(int epochs, int hiddenSize, Scheduler? scheduler) : base(epochs) {
        Scheduler = scheduler;
        HiddenSize = hiddenSize;
    }

    public Scheduler? Scheduler { get; }
    public int HiddenSize { get; }
}