namespace LearningAI.utils;

public class PerceptronsRoutine : Routine {
    public PerceptronsRoutine(int epochs, Scheduler? scheduler) : base(epochs) {
        Scheduler = scheduler;
    }

    public Scheduler? Scheduler { get; }
}