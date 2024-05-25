using System.Numerics;
using ImGuiNET;
using LearningAI.model;
using LearningAI.utils;

namespace LearningAI.ui;

public static class MultiLayerPerceptronsStats {
    public static void Render(MultiLayerPerceptronsModel mlp) {
        var displayWidth = ImGui.GetColumnWidth();

        if (mlp.AccuracyHistory.Count > 0) {
            var accuracy = mlp.AccuracyHistory.ToArray();

            Plot.Begin("Accuracy", new Vector2(displayWidth / 2, 250), mlp.AccuracyHistory.Min(),
                mlp.AccuracyHistory.Max(),
                $"Accuracy: {accuracy.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref accuracy, ImGui.ColorConvertFloat4ToU32(new Vector4(1f, .2f, 1f, 1f)));
            Plot.End();
        }

        if (mlp.LearningRateHistory.Count > 0) {
            var learningRate = mlp.LearningRateHistory.ToArray();

            ImGui.SameLine();

            Plot.Begin("Learning Rate", new Vector2(displayWidth / 2, 250), mlp.LearningRateHistory.Min(),
                mlp.LearningRateHistory.Max(), $"Learning Rate: {learningRate.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref learningRate, ImGui.ColorConvertFloat4ToU32(new Vector4(0f, .8f, 1f, 1f)));
            Plot.End();
        }

        if (mlp.Tpr.Count > 0) {
            var tpr = mlp.Tpr.ToArray();

            Plot.Begin("TPR", new Vector2(displayWidth / 2, 250), tpr.Min(), tpr.Max(), $"TPR: {tpr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Bar(ref tpr, ImGui.ColorConvertFloat4ToU32(new Vector4(.5f, 1f, .5f, 1f)),
                ImGui.ColorConvertFloat4ToU32(new Vector4(.5f, 1f, .5f, 0.5f)));
            Plot.End();
        }

        if (mlp.Fpr.Count > 0) {
            var fpr = mlp.Fpr.ToArray();

            ImGui.SameLine();

            Plot.Begin("FPR", new Vector2(displayWidth / 2, 250), fpr.Min(), fpr.Max(), $"FPR: {fpr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Bar(ref fpr, ImGui.ColorConvertFloat4ToU32(new Vector4(1f, .2f, .2f, 1f)),
                ImGui.ColorConvertFloat4ToU32(new Vector4(1f, .2f, .2f, .5f)));
            Plot.End();
        }

        if (mlp.Tnr.Count > 0) {
            var tnr = mlp.Tnr.ToArray();

            Plot.Begin("TNR", new Vector2(displayWidth / 2, 250), tnr.Min(), tnr.Max(), $"TNR: {tnr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref tnr, ImGui.ColorConvertFloat4ToU32(new Vector4(0.5f, 1f, .5f, 1f)));
            Plot.End();
        }

        if (mlp.Fnr.Count > 0) {
            var fnr = mlp.Fnr.ToArray();

            ImGui.SameLine();

            Plot.Begin("FNR", new Vector2(displayWidth / 2, 250), fnr.Min(), fnr.Max(), $"FNR: {fnr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref fnr, ImGui.ColorConvertFloat4ToU32(new Vector4(1f, .2f, .2f, 1f)));
            Plot.End();
        }
    }
}