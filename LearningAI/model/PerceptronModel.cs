using System.Text;
using LearningAI.ui;
using LearningAI.utils;
using Perceptrons;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LearningAI.model;

public class PerceptronModel : Model {
    public PerceptronModel() : base("Perceptron") {
        InitializePerceptron();
    }

    public List<float> AccuracyHistory { get; } = [];
    public Perceptron Perceptron { get; set; } = new(ImageClassification.InputSize, 0.01f);
    public List<(double[] values, double min, double max)> WeightsMapHistory { get; } = [];
    public List<float> Tpr { get; } = [];
    public List<float> Fpr { get; } = [];
    public List<float> Tnr { get; } = [];
    public List<float> Fnr { get; } = [];
    public List<float> LearningRateHistory { get; } = [];

    public int CurrentEpoch { get; private set; }

    public override void StartTraining(TrainingSettings trainingSettings) {
        Clear();

        var thread = new Thread(() => Train(trainingSettings));

        thread.Start();
    }

    private void InitializePerceptron(TrainingSettings? trainingSettings = null) {
        var learningRate = trainingSettings?.SelectedScheduler.GetLearningRate(0) ?? 0.01f;
        
        CustomConsole.Log($"Initializing perceptron");

        Perceptron = new Perceptron(ImageClassification.InputSize, learningRate);
        WeightsMapHistory.Add((Perceptron.Weights.ToArray(), Perceptron.Weights.Min(), Perceptron.Weights.Max()));
    }

    private void Clear() {
        CustomConsole.Log($"Clearing history");
        
        AccuracyHistory.Clear();
        WeightsMapHistory.Clear();
        Tpr.Clear();
        Fpr.Clear();
        Tnr.Clear();
        Fnr.Clear();
        LearningRateHistory.Clear();

        InitializePerceptron();
    }

    public string Predict(Image<Rgba32> image) {
        var pixels = new List<double>();

        for (var x = 0; x < image.Width; x++)
        for (var y = 0; y < image.Height; y++) {
            var pixel = image[x, y];
            var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0;
            pixels.Add(grayscale);
        }

        var guess = Perceptron.Activate([..pixels, 1]) == 1 ? "positive" : "negative";

        return guess;
    }

    private void Train(TrainingSettings trainingSettings) {
        CustomConsole.Log($"Starting training with {trainingSettings.Epochs} epochs");
        
        InitializePerceptron(trainingSettings);

        for (var i = 0; i < trainingSettings.Epochs; i++) {
            Perceptron.Learnc = trainingSettings.SelectedScheduler.GetLearningRate(i);

            foreach (var (label, image) in ImageLoader.TrainImages) {
                var pixels = new List<double>();

                for (var x = 0; x < image.Width; x++)
                for (var y = 0; y < image.Height; y++) {
                    var pixel = image[x, y];
                    var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0;
                    pixels.Add(grayscale);
                }

                Perceptron.Train([..pixels, 1], label is "bacteria" or "virus" ? 1 : 0);
            }

            Validate();

            LearningRateHistory.Add((float)Perceptron.Learnc);
            CurrentEpoch = i;
        }
    }

    private void Validate() {
        var correct = 0;
        var total = 0;
        var tp = 0;
        var fp = 0;
        var tn = 0;
        var fn = 0;

        foreach (var (label, image) in ImageLoader.ValImages) {
            var pixels = new List<double>();

            for (var x = 0; x < image.Width; x++)
            for (var y = 0; y < image.Height; y++) {
                var pixel = image[x, y];
                var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0;
                pixels.Add(grayscale);
            }

            var guess = Perceptron.Activate([..pixels, 1]) == 1 ? "positive" : "negative";

            switch (guess) {
                case "positive" when label is "bacteria" or "virus":
                    tp++;
                    correct++;
                    break;
                case "positive" when label == "negative":
                    fp++;
                    break;
                case "negative" when label == "negative":
                    tn++;
                    correct++;
                    break;
                case "negative" when label is "bacteria" or "virus":
                    fn++;
                    break;
            }

            total++;
        }

        var acc = (float)Math.Round(correct / (double)total * 100, 3);
        var tpr = tp / (float)(tp + fn);
        var fpr = fp / (float)(fp + tn);
        var tnr = tn / (float)(tn + fp);
        var fnr = fn / (float)(fn + tp);

        CustomConsole.Log($"Epoch: {CurrentEpoch} Accuracy: {acc}% TPR: {tpr} FPR: {fpr} TNR: {tnr} FNR: {fnr}");

        WeightsMapHistory.Add((Perceptron.Weights.ToArray(), Perceptron.Weights.Min(),
            Perceptron.Weights.Max()));
        AccuracyHistory.Add(acc);
        Tpr.Add(tpr);
        Fpr.Add(fpr);
        Tnr.Add(tnr);
        Fnr.Add(fnr);
    }
    
    public string ExportResults() {
        var sb = new StringBuilder();
        
        sb.AppendLine("Epoch,Accuracy,TPR,FPR,TNR,FNR");
        
        for (var i = 0; i < AccuracyHistory.Count; i++) {
            sb.AppendLine($"{i},{AccuracyHistory[i]},{Tpr[i]},{Fpr[i]},{Tnr[i]},{Fnr[i]}");
        }

        return sb.ToString();
    }
}