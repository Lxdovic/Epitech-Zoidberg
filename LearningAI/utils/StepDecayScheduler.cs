namespace LearningAI.utils;

public class StepDecayScheduler : Scheduler {
    public StepDecayScheduler(float initialLearningRate, float decayFactor, int stepSize) : base("Step Decay") {
        InitialLearningRate = initialLearningRate;
        DecayFactor = decayFactor;
        StepSize = stepSize;
    }

    public float InitialLearningRate { get; }
    public float DecayFactor { get; }
    public int StepSize { get; }
}