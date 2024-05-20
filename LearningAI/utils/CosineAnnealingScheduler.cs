namespace LearningAI.utils;

public class CosineAnnealingScheduler : Scheduler {
    public CosineAnnealingScheduler(float learningRateMin, float learningRateMax) : base("Cosine Annealing") {
        LearningRateMin = learningRateMin;
        LearningRateMax = learningRateMax;
    }

    public float LearningRateMin { get; }
    public float LearningRateMax { get; }
}