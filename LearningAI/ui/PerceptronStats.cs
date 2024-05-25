using System.Numerics;
using ImGuiNET;
using LearningAI.model;
using LearningAI.utils;

namespace LearningAI.ui;

public static class PerceptronStats {
    private static bool _showVisualization;
    private static int _selectedVisualization;
    private static int _predictedImageIndex;
    private static string _prediction = "";
    private static readonly string[] VisualizationOptions = ["Weights", "Predicted Image", "Weights Overlay Image"];

    public static void Render(PerceptronModel perceptron) {
        var displayWidth = ImGui.GetColumnWidth();
        var style = ImGui.GetStyle();

        if (perceptron.AccuracyHistory.Count > 0) {
            var accuracy = perceptron.AccuracyHistory.ToArray();

            Plot.Begin("Accuracy", new Vector2(displayWidth / 2, 250), perceptron.AccuracyHistory.Min(),
                perceptron.AccuracyHistory.Max(),
                $"Accuracy: {accuracy.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref accuracy, ImGui.ColorConvertFloat4ToU32(style.Colors[(int)ImGuiCol.SliderGrabActive]));
            Plot.End();
        }

        if (perceptron.LearningRateHistory.Count > 0) {
            var learningRate = perceptron.LearningRateHistory.ToArray();

            ImGui.SameLine();

            Plot.Begin("Learning Rate", new Vector2(displayWidth / 2, 250), perceptron.LearningRateHistory.Min(),
                perceptron.LearningRateHistory.Max(), $"Learning Rate: {learningRate.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref learningRate, ImGui.ColorConvertFloat4ToU32(style.Colors[(int)ImGuiCol.PlotLinesHovered]));
            Plot.End();
        }

        if (perceptron.Tpr.Count > 0) {
            var tpr = perceptron.Tpr.ToArray();

            Plot.Begin("TPR", new Vector2(displayWidth / 2, 250), tpr.Min(), tpr.Max(), $"TPR: {tpr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref tpr, ImGui.ColorConvertFloat4ToU32(style.Colors[(int)ImGuiCol.DragDropTarget]));
            Plot.End();
        }

        if (perceptron.Fpr.Count > 0) {
            var fpr = perceptron.Fpr.ToArray();

            ImGui.SameLine();

            Plot.Begin("FPR", new Vector2(displayWidth / 2, 250), fpr.Min(), fpr.Max(), $"FPR: {fpr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref fpr, ImGui.ColorConvertFloat4ToU32(new Vector4(.75f, .01f, .4f, 1f)));
            Plot.End();
        }

        if (perceptron.Tnr.Count > 0) {
            var tnr = perceptron.Tnr.ToArray();

            Plot.Begin("TNR", new Vector2(displayWidth / 2, 250), tnr.Min(), tnr.Max(), $"TNR: {tnr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref tnr, ImGui.ColorConvertFloat4ToU32(style.Colors[(int)ImGuiCol.DragDropTarget]));
            Plot.End();
        }

        if (perceptron.Fnr.Count > 0) {
            var fnr = perceptron.Fnr.ToArray();

            ImGui.SameLine();

            Plot.Begin("FNR", new Vector2(displayWidth / 2, 250), fnr.Min(), fnr.Max(), $"FNR: {fnr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref fnr, ImGui.ColorConvertFloat4ToU32(new Vector4(.75f, .01f, .4f, 1f)));
            Plot.End();
        }
    }

    private static void PreviousImage(PerceptronModel perceptron) {
        if (ImageLoader.IsLoading) return;

        _predictedImageIndex = Math.Max(0, _predictedImageIndex - 1);
        _prediction = perceptron.Predict(ImageLoader.ValImages[_predictedImageIndex].image);
    }

    private static void NextImage(PerceptronModel perceptron) {
        if (ImageLoader.IsLoading) return;

        _predictedImageIndex = Math.Min(ImageLoader.ValImages.Count - 1, _predictedImageIndex + 1);
        _prediction = perceptron.Predict(ImageLoader.ValImages[_predictedImageIndex].image);
    }

    public static void RenderAdditional(PerceptronModel perceptron) {
        var width = ImGui.GetColumnWidth();
        var style = ImGui.GetStyle();

        ImGui.SeparatorText("Weights");
        ImGui.Checkbox("Show Visualization", ref _showVisualization);

        if (!_showVisualization || ImageLoader.ValImages.Count <= 0) return;

        ImGui.Combo("Visualization", ref _selectedVisualization, VisualizationOptions, VisualizationOptions.Length);

        if (ImGui.ArrowButton("Left", ImGuiDir.Left)) PreviousImage(perceptron);

        var label = ImageLoader.ValImages[_predictedImageIndex].label;

        ImGui.SameLine();
        ImGui.TextColored(style.Colors[(int)ImGuiCol.DragDropTarget], $"{_prediction}");
        ImGui.SameLine();
        ImGui.TextColored(style.Colors[(int)ImGuiCol.PlotHistogram],
            label == "bacteria" || label == "virus" ? "positive" : "negative");
        ImGui.SameLine();

        if (ImGui.ArrowButton("Right", ImGuiDir.Right)) NextImage(perceptron);

        switch (VisualizationOptions[_selectedVisualization]) {
            case "Weights" when perceptron.WeightsMapHistory.Count > 0:
                ImageClassification.RenderHeatMap("Heatmap", perceptron.WeightsMapHistory.Last(),
                    (int)(width / ImageLoader.ImageSize.X));
                break;

            case "Predicted Image":
                ImageClassification.RenderImage("Predicted Image", ImageLoader.ValImages[_predictedImageIndex].image,
                    (int)(width / ImageLoader.ImageSize.X));
                break;

            case "Weights Overlay Image" when
                perceptron.WeightsMapHistory.Count > 0:
                ImageClassification.RenderImageDiff("Weights Overlay Image",
                    ImageLoader.ValImages[_predictedImageIndex].image, perceptron.WeightsMapHistory.Last(),
                    (int)(width / ImageLoader.ImageSize.X));
                break;
        }
    }
}