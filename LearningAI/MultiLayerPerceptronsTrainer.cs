using System.ComponentModel;
using System.Numerics;
using ImGuiNET;
using LearningAI.utils;
using MultiLayerPerceptrons;

namespace LearningAI;

public class MultiLayerPerceptronsTrainer {
    public static BackgroundWorker TrainingWorker = new();
    public static NeuralNetwork NeuralNetwork = new(0, 0, 0, 0);
    public static readonly List<float> LearningRateHistory = new();
    public static readonly List<float> AccuracyHistory = new();
    public static readonly List<float> Tpr = new();
    public static readonly List<float> Fpr = new();
    public static readonly List<float> Tnr = new();
    public static readonly List<float> Fnr = new();

    public static void StartTraining(int inputs, int epochs) {
        if (TrainingWorker.IsBusy || ImageClassification.ImageloadingWorkers.Any(w => w.IsBusy)) return;

        AccuracyHistory.Clear();
        LearningRateHistory.Clear();
        Fpr.Clear();
        Tpr.Clear();
        Fnr.Clear();
        Tnr.Clear();

        TrainingWorker = new BackgroundWorker();

        ImageClassification.TrainLoad.Curr = 0;
        ImageClassification.TrainLoad.Max = epochs;

        TrainingWorker.WorkerReportsProgress = true;
        TrainingWorker.WorkerSupportsCancellation = true;
        TrainingWorker.DoWork += (_, _) => { TrainPerceptron(inputs, epochs); };
        TrainingWorker.ProgressChanged += (_, _) => ImageClassification.TrainLoad.Curr++;
        TrainingWorker.RunWorkerAsync();
    }

    public static void TrainPerceptron(int inputs, int epochs) {
        NeuralNetwork = new NeuralNetwork(inputs, ImageClassification.MlpHiddenLayerSize, 2, (float)ImageClassification.GetLearningRate(0));

        for (var i = 0; i < epochs; i++) {
            LearningRateHistory.Add((float)ImageClassification.GetLearningRate(i));
            NeuralNetwork.LearningRate = LearningRateHistory.Last();

            for (var j = 0; j < ImageClassification.TrainImages.Count; j++) {
                var (label, image) = ImageClassification.TrainImages[j];
                var pixels = new List<double>();

                for (var x = 0; x < image.Width; x++)
                for (var y = 0; y < image.Height; y++) {
                    var pixel = image[x, y];
                    var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0 / 255.0;
                    pixels.Add(grayscale);
                }

                double[] answers = label == "bacteria" || label == "virus" ? [1, 0] : [0, 1];
                NeuralNetwork.Train([..pixels], answers);
            }

            Console.WriteLine($"Epoch {i + 1} / {epochs}");

            Validate();

            TrainingWorker.ReportProgress(0);
        }
    }

    private static void Validate() {
        var correct = 0;
        var total = 0;
        var tp = 0;
        var fp = 0;
        var tn = 0;
        var fn = 0;

        for (var i = 0; i < ImageClassification.ValImages.Count; i++) {
            var (label, image) = ImageClassification.ValImages[i];
            var pixels = new List<double>();

            for (var x = 0; x < image.Width; x++)
            for (var y = 0; y < image.Height; y++) {
                var pixel = image[x, y];
                var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0 / 255.0;
                pixels.Add(grayscale);
            }

            var predict = NeuralNetwork.FeedForward([..pixels]);
            var guess = predict.ToList().IndexOf(predict.Max()) == 0 ? "positive" : "negative";

            if (guess == "positive" && (label == "bacteria" || label == "virus")) {
                tp++;
                correct++;
            }
            else if (guess == "positive" && label == "negative") {
                fp++;
            }
            else if (guess == "negative" && label == "negative") {
                tn++;
                correct++;
            }
            else if (guess == "negative" && (label == "bacteria" || label == "virus")) {
                fn++;
            }

            total++;
        }

        var acc = (float)Math.Round(correct / (double)total * 100, 3);
        var tpr = tp / (float)(tp + fn);
        var fpr = fp / (float)(fp + tn);
        var tnr = tn / (float)(tn + fp);
        var fnr = fn / (float)(fn + tp);

        AccuracyHistory.Add(acc);
        Tpr.Add(tpr);
        Fpr.Add(fpr);
        Tnr.Add(tnr);
        Fnr.Add(fnr);
    }

    public static void Render() {
        var displayWidth = ImGui.GetColumnWidth();

        if (AccuracyHistory.Count > 0) {
            var accuracy = AccuracyHistory.ToArray();

            Plot.Begin("Accuracy", new Vector2(displayWidth / 2, 250), AccuracyHistory.Min(),
                AccuracyHistory.Max(),
                $"Accuracy: {accuracy.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref accuracy, ImGui.ColorConvertFloat4ToU32(new Vector4(1f, .2f, 1f, 1f)));
            Plot.End();
        }

        if (LearningRateHistory.Count > 0) {
            var learningRate = LearningRateHistory.ToArray();

            ImGui.SameLine();

            Plot.Begin("Learning Rate", new Vector2(displayWidth / 2, 250), LearningRateHistory.Min(),
                LearningRateHistory.Max(), $"Learning Rate: {learningRate.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref learningRate, ImGui.ColorConvertFloat4ToU32(new Vector4(0f, .8f, 1f, 1f)));
            Plot.End();
        }

        if (Tpr.Count > 0) {
            var tpr = Tpr.ToArray();

            Plot.Begin("TPR", new Vector2(displayWidth / 2, 250), tpr.Min(), tpr.Max(), $"TPR: {tpr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Bar(ref tpr, ImGui.ColorConvertFloat4ToU32(new Vector4(.5f, 1f, .5f, 1f)),
                ImGui.ColorConvertFloat4ToU32(new Vector4(.5f, 1f, .5f, 0.5f)));
            Plot.End();
        }

        if (Fpr.Count > 0) {
            var fpr = Fpr.ToArray();

            ImGui.SameLine();

            Plot.Begin("FPR", new Vector2(displayWidth / 2, 250), fpr.Min(), fpr.Max(), $"FPR: {fpr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Bar(ref fpr, ImGui.ColorConvertFloat4ToU32(new Vector4(1f, .2f, .2f, 1f)),
                ImGui.ColorConvertFloat4ToU32(new Vector4(1f, .2f, .2f, .5f)));
            Plot.End();
        }

        if (Tnr.Count > 0) {
            var tnr = Tnr.ToArray();

            Plot.Begin("TNR", new Vector2(displayWidth / 2, 250), tnr.Min(), tnr.Max(), $"TNR: {tnr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref tnr, ImGui.ColorConvertFloat4ToU32(new Vector4(0.5f, 1f, .5f, 1f))
            );
            Plot.End();
        }

        if (Fnr.Count > 0) {
            var fnr = Fnr.ToArray();

            ImGui.SameLine();

            Plot.Begin("FNR", new Vector2(displayWidth / 2, 250), fnr.Min(), fnr.Max(), $"FNR: {fnr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref fnr, ImGui.ColorConvertFloat4ToU32(new Vector4(1f, .2f, .2f, 1f)));
            Plot.End();
        }
    }
}