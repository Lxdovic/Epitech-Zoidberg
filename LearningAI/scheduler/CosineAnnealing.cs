namespace LearningAI.scheduler;

public class CosineAnnealing : LearningRateScheduler {
    public CosineAnnealing(double learningRateMin, double learningRateMax) : base("Cosine Annealing") {
        LearningRateMin = learningRateMin;
        LearningRateMax = learningRateMax;
    }

    public double LearningRateMin { get; set; }
    public double LearningRateMax { get; set; }

    public override double GetLearningRate(int epoch, int epochs) {
        return LearningRateMin +
               0.5 * (LearningRateMax - LearningRateMin) * (1 + Math.Cos(epoch / (double)epochs * Math.PI));
    }
}