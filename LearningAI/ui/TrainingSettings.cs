using ImGuiNET;
using LearningAI.model;
using LearningAI.scheduler;

namespace LearningAI.ui;

public class TrainingSettings {
    private static readonly Model[] Models = [new PerceptronModel(), new MultiLayerPerceptronsModel()];
    private static readonly string[] ModelNames = Models.Select(m => m.Name).ToArray();

    private static readonly LearningRateScheduler[] LearningRateSchedulers = [
        new NoScheduler(0.01f),
        new StepDecay(0.01f, 0.1f, 10),
        new ExpoDecay(0.01f, 0.001f),
        new CosineAnnealing(0.01f, 0.1f)
    ];

    private static readonly string[] LearningRateSchedulerNames = LearningRateSchedulers.Select(m => m.Name).ToArray();

    private int _epochs;
    private int _selectedModel;
    private int _selectedScheduler;

    public Model SelectedModel => Models[_selectedModel];
    public LearningRateScheduler SelectedScheduler => LearningRateSchedulers[_selectedScheduler];
    public int Epochs => _epochs;

    public void Render() {
        var selectedModel = SelectedModel;
        var selectedScheduler = SelectedScheduler;

        ImGui.Combo("Model", ref _selectedModel, ModelNames, Models.Length);

        if (selectedModel is MultiLayerPerceptronsModel mlp) {
            var hiddenLayerSize = mlp.HiddenLayerSize;

            ImGui.SliderInt("Hidden Layer Size", ref hiddenLayerSize, 1, 32);

            mlp.HiddenLayerSize = hiddenLayerSize;
        }

        ImGui.DragInt("Epochs", ref _epochs, 0, 100);
        ImGui.Combo("Learning Rate Scheduler", ref _selectedScheduler, LearningRateSchedulerNames,
            LearningRateSchedulers.Length);

        switch (selectedScheduler) {
            case NoScheduler noScheduler: {
                var learningRate = (float)noScheduler.LearningRate;

                ImGui.DragFloat("Learning Rate", ref learningRate, 0.001f, 1f);

                noScheduler.LearningRate = learningRate;
                break;
            }

            case StepDecay stepDecay: {
                var learningRate = (float)stepDecay.LearningRate;
                var decayFactor = (float)stepDecay.DecayFactor;
                var stepSize = stepDecay.StepSize;

                ImGui.DragFloat("Learning Rate", ref learningRate, 0.001f, 1f);
                ImGui.DragFloat("Decay Factor", ref decayFactor, 0.1f, 1f);
                ImGui.DragInt("Step Size", ref stepSize, 1, 10);

                stepDecay.LearningRate = learningRate;
                stepDecay.DecayFactor = decayFactor;
                stepDecay.StepSize = stepSize;
                break;
            }

            case ExpoDecay expoDecay: {
                var learningRate = (float)expoDecay.LearningRate;
                var decayRate = (float)expoDecay.DecayRate;

                ImGui.DragFloat("Learning Rate", ref learningRate, 0.001f, 1f);
                ImGui.DragFloat("Decay Rate", ref decayRate, 0.001f, 1f);

                expoDecay.LearningRate = learningRate;
                expoDecay.DecayRate = decayRate;
                break;
            }

            case CosineAnnealing cosineAnnealing: {
                var learningRateMin = (float)cosineAnnealing.LearningRateMin;
                var learningRateMax = (float)cosineAnnealing.LearningRateMax;

                ImGui.DragFloat("Learning Rate Min", ref learningRateMin, 0.001f, 1f);
                ImGui.DragFloat("Learning Rate Max", ref learningRateMax, 0.001f, 1f);

                cosineAnnealing.LearningRateMin = learningRateMin;
                cosineAnnealing.LearningRateMax = learningRateMax;
                break;
            }
        }
    }
}