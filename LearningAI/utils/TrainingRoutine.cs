namespace LearningAI.utils;

public class TrainingRoutine {
    private static readonly List<RoutineResult> _routineResults = new();
    private static Stack<Routine> _routines = new();

    public static void StartRoutines(List<Routine> routines) {
        _routines = new Stack<Routine>(routines);

        StartNextRoutine();
    }

    private static void StartNextRoutine() {
        var routine = _routines.Peek();

        switch (routine) {
            case PerceptronsRoutine perceptronsRoutine:
                StartPerceptronsRoutine(perceptronsRoutine);
                break;

            case MlpRoutine mlpRoutine:
                StartMlpRoutine(mlpRoutine);
                break;
        }
    }

    private static void StartMlpRoutine(MlpRoutine mlpRoutine) {
        ImageClassification._epochs = mlpRoutine.Epochs;
        ImageClassification.MlpHiddenLayerSize = mlpRoutine.HiddenSize;
        ImageClassification._selectedModel = 1;

        switch (mlpRoutine.Scheduler) {
            case NoneScheduler noneScheduler:
                ImageClassification._learningRate = noneScheduler.LearningRate;
                ImageClassification._selectedScheduler = 0;
                break;

            case StepDecayScheduler stepDecayScheduler:
                ImageClassification._learningRate = stepDecayScheduler.InitialLearningRate;
                ImageClassification._decayFactor = stepDecayScheduler.DecayFactor;
                ImageClassification._stepSize = stepDecayScheduler.StepSize;
                ImageClassification._selectedScheduler = 1;
                break;

            case ExpoDecayScheduler expoDecayScheduler:
                ImageClassification._learningRate = expoDecayScheduler.InitialLearningRate;
                ImageClassification._decayRate = expoDecayScheduler.DecayRate;
                ImageClassification._selectedScheduler = 2;
                break;

            case CosineAnnealingScheduler cosineAnnealingScheduler:
                ImageClassification._learningRateMin = cosineAnnealingScheduler.LearningRateMin;
                ImageClassification._learningRateMax = cosineAnnealingScheduler.LearningRateMax;
                ImageClassification._selectedScheduler = 3;
                break;
        }

        var inputs = ImageClassification.ImageSize[0] * ImageClassification.ImageSize[1];

        MultiLayerPerceptronsTrainer.StartTraining(inputs, ImageClassification._epochs, OnFinish);
    }

    private static void StartPerceptronsRoutine(PerceptronsRoutine perceptronsRoutine) {
        ImageClassification._epochs = perceptronsRoutine.Epochs;
        ImageClassification._selectedModel = 0;

        switch (perceptronsRoutine.Scheduler) {
            case NoneScheduler noneScheduler:
                ImageClassification._learningRate = noneScheduler.LearningRate;
                ImageClassification._selectedScheduler = 0;
                break;

            case StepDecayScheduler stepDecayScheduler:
                ImageClassification._learningRate = stepDecayScheduler.InitialLearningRate;
                ImageClassification._decayFactor = stepDecayScheduler.DecayFactor;
                ImageClassification._stepSize = stepDecayScheduler.StepSize;
                ImageClassification._selectedScheduler = 1;
                break;

            case ExpoDecayScheduler expoDecayScheduler:
                ImageClassification._learningRate = expoDecayScheduler.InitialLearningRate;
                ImageClassification._decayRate = expoDecayScheduler.DecayRate;
                ImageClassification._selectedScheduler = 2;
                break;

            case CosineAnnealingScheduler cosineAnnealingScheduler:
                ImageClassification._learningRateMin = cosineAnnealingScheduler.LearningRateMin;
                ImageClassification._learningRateMax = cosineAnnealingScheduler.LearningRateMax;
                ImageClassification._selectedScheduler = 3;
                break;
        }

        var inputs = ImageClassification.ImageSize[0] * ImageClassification.ImageSize[1];

        PerceptronsTrainer.StartTraining(inputs, ImageClassification._epochs, OnFinish);
    }

    private static void OnFinish(List<float> accuracyHistory, List<float> learningRateHistory, List<float> tpr,
        List<float> fpr, List<float> tnr, List<float> fnr) {
        _routineResults.Add(new RoutineResult(
            _routines.Pop(),
            accuracyHistory.Last(),
            learningRateHistory.Last(),
            tpr.Last(),
            fpr.Last(),
            tnr.Last(),
            fnr.Last()
        ));

        if (_routines.Count > 0)
            StartNextRoutine();
        else
            WriteResults();
    }

    private static void WriteResults() {
        var csvData = new List<CsvData>();

        for (var i = 0; i < _routineResults.Count; i++) {
            var routineResult = _routineResults[i];
            var model = routineResult.Routine switch {
                PerceptronsRoutine => "Perceptrons",
                MlpRoutine => "MLP",
                _ => "Unknown"
            };

            csvData.Add(new CsvData(i, model
                , routineResult.Accuracy,
                routineResult.Tpr, routineResult.Fpr, routineResult.Tnr,
                routineResult.Fnr));
        }

        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

        CsvHelper.WriteCsv($"{home}/Desktop/results.csv", csvData);
    }

    public static List<Routine> CreateRoutines() {
        List<Routine> routines = [
            new PerceptronsRoutine(10, new NoneScheduler(0.1f)),
            new PerceptronsRoutine(10, new ExpoDecayScheduler(0.1f, 0.1f)),
            new PerceptronsRoutine(10, new StepDecayScheduler(0.1f, 0.1f, 5)),
            new PerceptronsRoutine(10, new CosineAnnealingScheduler(0.1f, 0.01f)),
            new MlpRoutine(10, 4, new NoneScheduler(0.1f)),
            new MlpRoutine(10, 4, new ExpoDecayScheduler(0.1f, 0.1f)),
            new MlpRoutine(10, 4, new StepDecayScheduler(0.1f, 0.1f, 5)),
            new MlpRoutine(10, 4, new CosineAnnealingScheduler(0.1f, 0.01f))
        ];

        return routines;
    }
}