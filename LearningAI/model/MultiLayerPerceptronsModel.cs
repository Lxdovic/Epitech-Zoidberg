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
        CustomConsole.Log($"Clearing history");
        
        AccuracyHistory.Clear();
        Tpr.Clear();
        Fpr.Clear();
        Tnr.Clear();
        Fnr.Clear();
        LearningRateHistory.Clear();
    }

    private void Train(TrainingSettings trainingSettings) {
        CustomConsole.Log($"Starting training with {trainingSettings.Epochs} epochs");
        CustomConsole.Log($"Initializing neural network with {HiddenLayerSize} hidden layers");
        
        NeuralNetwork = new NeuralNetwork(ImageClassification.InputSize,
            HiddenLayerSize, 2, (float)trainingSettings.SelectedScheduler.GetLearningRate(0));

        for (var i = 0; i < trainingSettings.Epochs; i++) {
            NeuralNetwork.LearningRate = (float)trainingSettings.SelectedScheduler.GetLearningRate(i);

            foreach (var (label, image) in ImageLoader.TrainImages) {
                var pixels = new List<double>();

                for (var x = 0; x < image.Width; x++)
                for (var y = 0; y < image.Height; y++) {
                    var pixel = image[x, y];
                    var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0 / 255.0;
                    pixels.Add(grayscale);
                }

                double[] answers = label is "bacteria" or "virus" ? [1, 0] : [0, 1];
                NeuralNetwork.Train([..pixels], answers);
            }

            Validate();

            LearningRateHistory.Add((float)NeuralNetwork.LearningRate);
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
                case "negative" when label == "bacteria" || label == "virus":
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

        AccuracyHistory.Add(acc);
        Tpr.Add(tpr);
        Fpr.Add(fpr);
        Tnr.Add(tnr);
        Fnr.Add(fnr);
    }
}