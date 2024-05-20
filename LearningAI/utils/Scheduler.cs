namespace LearningAI.utils;

public abstract class Scheduler {
    public readonly string Name;

    protected Scheduler(string name) {
        Name = name;
    }
}