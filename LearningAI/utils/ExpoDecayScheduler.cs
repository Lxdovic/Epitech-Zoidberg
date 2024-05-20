namespace LearningAI.utils;

public class ExpoDecayScheduler : Scheduler {
    public ExpoDecayScheduler(float initialLearningRate, float decayRate) : base("Expo Decay") {
        InitialLearningRate = initialLearningRate;
        DecayRate = decayRate;
    }

    public float InitialLearningRate { get; }
    public float DecayRate { get; }
}