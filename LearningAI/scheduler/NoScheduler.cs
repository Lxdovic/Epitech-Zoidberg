namespace LearningAI.scheduler;

public class NoScheduler : LearningRateScheduler {
    public NoScheduler(float learningRate) : base("No Scheduler") {
        LearningRate = learningRate;
    }

    public double LearningRate { get; set; }

    public override double GetLearningRate(int epoch) {
        return LearningRate;
    }
}