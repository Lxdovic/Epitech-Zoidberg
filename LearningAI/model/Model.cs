using LearningAI.ui;

namespace LearningAI.model;

public abstract class Model(string name) {
    public readonly string Name = name;

    public abstract void StartTraining(TrainingSettings trainingSettings);
}