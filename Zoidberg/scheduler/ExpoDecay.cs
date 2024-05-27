namespace Zoidberg.scheduler;

public class ExpoDecay : LearningRateScheduler {
    public ExpoDecay(double learningRate, double decayRate) : base("Exponential Decay") {
        LearningRate = learningRate;
        DecayRate = decayRate;
    }

    public double LearningRate { get; set; }
    public double DecayRate { get; set; }

    public override double GetLearningRate(int epoch) {
        return LearningRate * Math.Exp(-DecayRate * epoch);
    }
}