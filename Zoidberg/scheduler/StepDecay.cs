namespace Zoidberg.scheduler;

public class StepDecay : LearningRateScheduler {
    public StepDecay(double learningRate, double decayFactor, int stepSize) : base("Step Decay") {
        LearningRate = learningRate;
        DecayFactor = decayFactor;
        StepSize = stepSize;
    }

    public double LearningRate { get; set; }
    public double DecayFactor { get; set; }
    public int StepSize { get; set; }

    public override double GetLearningRate(int epoch) {
        return LearningRate * Math.Pow(DecayFactor, Math.Floor((1 + epoch) / (double)StepSize));
    }
}