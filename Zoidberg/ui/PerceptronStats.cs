using System.Numerics;
using ImGuiNET;
using Zoidberg.model;
using Zoidberg.utils;

namespace Zoidberg.ui;

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

    private static void NextRandomImage(PerceptronModel perceptron) {
        if (ImageLoader.IsLoading) return;

        var random = new Random();

        _predictedImageIndex = random.Next(0, ImageLoader.ValImages.Count);
        _prediction = perceptron.Predict(ImageLoader.ValImages[_predictedImageIndex].image);
    }

    public static void RenderAdditional(PerceptronModel perceptron) {
        var style = ImGui.GetStyle();
        var width = ImGui.GetColumnWidth();

        ImGui.SeparatorText("Weights");
        ImGui.Checkbox("Show Visualization", ref _showVisualization);

        if (!_showVisualization || ImageLoader.ValImages.Count <= 0) return;

        ImGui.Combo("Visualization", ref _selectedVisualization, VisualizationOptions, VisualizationOptions.Length);

        if (ImGui.Button("Random Image")) NextRandomImage(perceptron);

        var label = ImageLoader.ValImages[_predictedImageIndex].label is "bacteria" or "virus"
            ? "positive"
            : "negative";

        ImGui.SameLine();
        ImGui.TextColored(
            label == _prediction
                ? style.Colors[(int)ImGuiCol.DragDropTarget]
                : style.Colors[(int)ImGuiCol.PlotHistogram], $"{_prediction}");

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

        if (ImGui.Button("Export Results")) {
            ImGui.OpenPopup("Export Results");
            
            Console.WriteLine(perceptron.ExportResults());
        }
        
        var results = perceptron.ExportResults();
        
        if (ImGui.BeginPopupModal("Export Results")) {
            ImGui.InputTextMultiline("##Results", ref results, 1000, new Vector2(ImGui.GetWindowWidth() - 20, ImGui.GetWindowHeight() - 80), ImGuiInputTextFlags.ReadOnly);
            
            if (ImGui.Button("Close")) { ImGui.CloseCurrentPopup(); }
            ImGui.EndPopup();
        }
    }
}
