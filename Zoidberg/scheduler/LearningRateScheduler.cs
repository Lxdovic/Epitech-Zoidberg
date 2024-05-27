namespace Zoidberg.scheduler;

public abstract class LearningRateScheduler(string name) {
    public readonly string Name = name;

    public virtual double GetLearningRate(int epoch) {
        return 0;
    }

    public virtual double GetLearningRate(int epoch, int epochs) {
        return 0;
    }
}