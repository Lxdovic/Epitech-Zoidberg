using LearningAI.ui;
using LearningAI.utils;
using MultiLayerPerceptrons;

namespace LearningAI.model;

public class MultiLayerPerceptronsModel() : Model("Multi Layer Perceptrons") {
    public List<float> AccuracyHistory { get; } = [];
    public List<float> Tpr { get; } = [];
    public List<float> Fpr { get; } = [];
    public List<float> Tnr { get; } = [];
    public List<float> Fnr { get; } = [];
    public List<float> LearningRateHistory { get; } = [];
    public int HiddenLayerSize { get; set; } = 4;
    public NeuralNetwork NeuralNetwork { get; set; } = new(0, 0, 0, 0);

    public int CurrentEpoch { get; private set; }

    public override void StartTraining(TrainingSettings trainingSettings) {
        Clear();

        var thread = new Thread(() => Train(trainingSettings));

        thread.Start();
    }

    private void Clear() {
        AccuracyHistory.Clear();
        Tpr.Clear();
        Fpr.Clear();
        Tnr.Clear();
        Fnr.Clear();
        LearningRateHistory.Clear();
    }

    private void Train(TrainingSettings trainingSettings) {
        NeuralNetwork = new NeuralNetwork(ImageClassification.InputSize,
            HiddenLayerSize, 2, (float)trainingSettings.SelectedScheduler.GetLearningRate(0));

        for (var i = 0; i < trainingSettings.Epochs; i++) {
            LearningRateHistory.Add((float)trainingSettings.SelectedScheduler.GetLearningRate(i));
            NeuralNetwork.LearningRate = LearningRateHistory.Last();

            foreach (var (label, image) in ImageLoader.TrainImages) {
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

            Validate();

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
}