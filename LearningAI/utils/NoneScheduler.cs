namespace LearningAI.utils;

public class NoneScheduler : Scheduler {
    public NoneScheduler(float learningRate) : base("None") {
        LearningRate = learningRate;
    }

    public float LearningRate { get; }
}