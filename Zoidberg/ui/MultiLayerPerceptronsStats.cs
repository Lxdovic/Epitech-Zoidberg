using System.Numerics;
using ImGuiNET;
using MultiLayerPerceptrons;
using Zoidberg.model;
using Zoidberg.utils;

namespace Zoidberg.ui;

public static class MultiLayerPerceptronsStats {
    private static bool _showVisualization;
    private static int _predictedImageIndex;
    private static string _prediction;

    public static void Render(MultiLayerPerceptronsModel mlp) {
        var displayWidth = ImGui.GetColumnWidth();
        var style = ImGui.GetStyle();

        if (mlp.AccuracyHistory.Count > 0) {
            var accuracy = mlp.AccuracyHistory.ToArray();

            Plot.Begin("Accuracy", new Vector2(displayWidth / 2, 250), mlp.AccuracyHistory.Min(),
                mlp.AccuracyHistory.Max(),
                $"Accuracy: {accuracy.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref accuracy, ImGui.ColorConvertFloat4ToU32(style.Colors[(int)ImGuiCol.SliderGrabActive]));
            Plot.End();
        }

        if (mlp.LearningRateHistory.Count > 0) {
            var learningRate = mlp.LearningRateHistory.ToArray();

            ImGui.SameLine();

            Plot.Begin("Learning Rate", new Vector2(displayWidth / 2, 250), mlp.LearningRateHistory.Min(),
                mlp.LearningRateHistory.Max(), $"Learning Rate: {learningRate.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref learningRate, ImGui.ColorConvertFloat4ToU32(style.Colors[(int)ImGuiCol.PlotLinesHovered]));
            Plot.End();
        }

        if (mlp.Tpr.Count > 0) {
            var tpr = mlp.Tpr.ToArray();

            Plot.Begin("TPR", new Vector2(displayWidth / 2, 250), tpr.Min(), tpr.Max(), $"TPR: {tpr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref tpr, ImGui.ColorConvertFloat4ToU32(style.Colors[(int)ImGuiCol.DragDropTarget]));
            Plot.End();
        }

        if (mlp.Fpr.Count > 0) {
            var fpr = mlp.Fpr.ToArray();

            ImGui.SameLine();

            Plot.Begin("FPR", new Vector2(displayWidth / 2, 250), fpr.Min(), fpr.Max(), $"FPR: {fpr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref fpr, ImGui.ColorConvertFloat4ToU32(new Vector4(.75f, .01f, .4f, 1f)));
            Plot.End();
        }

        if (mlp.Tnr.Count > 0) {
            var tnr = mlp.Tnr.ToArray();

            Plot.Begin("TNR", new Vector2(displayWidth / 2, 250), tnr.Min(), tnr.Max(), $"TNR: {tnr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref tnr, ImGui.ColorConvertFloat4ToU32(style.Colors[(int)ImGuiCol.DragDropTarget]));
            Plot.End();
        }

        if (mlp.Fnr.Count > 0) {
            var fnr = mlp.Fnr.ToArray();

            ImGui.SameLine();

            Plot.Begin("FNR", new Vector2(displayWidth / 2, 250), fnr.Min(), fnr.Max(), $"FNR: {fnr.Last()} %");
            Plot.Annotations(new Vector2(4, 4));
            Plot.Line(ref fnr, ImGui.ColorConvertFloat4ToU32(new Vector4(.75f, .01f, .4f, 1f)));
            Plot.End();
        }
    }

    public static void RenderAdditional(MultiLayerPerceptronsModel mlp) {
        var style = ImGui.GetStyle();
        var width = ImGui.GetColumnWidth();
        var drawList = ImGui.GetWindowDrawList();

        if (ImageLoader.ValImages.Count <= 0 || ImageLoader.IsLoading) return;

        ImGui.SeparatorText("Weights");
        ImGui.Checkbox("Show Visualization", ref _showVisualization);

        if (!_showVisualization) return;

        if (ImGui.Button("Random Image", new Vector2(ImGui.CalcItemWidth(), ImGui.GetFrameHeight())))
            NextRandomImage(mlp);

        var label = ImageLoader.ValImages[_predictedImageIndex].label is "bacteria" or "virus"
            ? "positive"
            : "negative";

        ImGui.SameLine();
        ImGui.TextColored(
            label == _prediction
                ? style.Colors[(int)ImGuiCol.DragDropTarget]
                : style.Colors[(int)ImGuiCol.PlotHistogram], $"{_prediction}");

        ImageClassification.RenderImage("Predicted Image", ImageLoader.ValImages[_predictedImageIndex].image,
            (int)(width / ImageLoader.ImageSize.X));

        RenderNetwork(mlp, drawList);
    }

    private static void NextRandomImage(MultiLayerPerceptronsModel mlp) {
        if (ImageLoader.IsLoading) return;

        var random = new Random();

        _predictedImageIndex = random.Next(0, ImageLoader.ValImages.Count);
        _prediction = mlp.Predict(ImageLoader.ValImages[_predictedImageIndex].image);
    }

    public static void RenderNetwork(MultiLayerPerceptronsModel mlp, ImDrawListPtr drawList) {
        var buttonSize = new Vector2(ImGui.GetColumnWidth(), 400);

        var neuronRadius = 10.0f;
        var neuronDistance = buttonSize.Y / mlp.NeuralNetwork.HiddenNodes;
        var padding = 60.0f;

        var cursorPos = ImGui.GetCursorScreenPos();
        var style = ImGui.GetStyle();

        ImGui.InvisibleButton("Network", buttonSize);

        drawList.AddRectFilled(cursorPos, buttonSize + cursorPos,
            ImGui.ColorConvertFloat4ToU32(style.Colors[(int)ImGuiCol.FrameBg]), style.FrameRounding);

        var layerDistance = buttonSize.X - padding * 2 - neuronRadius * 2;

        var hiddenLayerPos = cursorPos + new Vector2(padding + neuronRadius, neuronDistance / 2);
        var outputLayerPos = hiddenLayerPos + new Vector2(
            layerDistance,
            neuronDistance * mlp.HiddenLayerSize / 2 - neuronDistance * mlp.NeuralNetwork.OutputNodes / 2);

        var hiddenWeights = mlp.NeuralNetwork.WeightsIh;
        var outputWeights = mlp.NeuralNetwork.WeightsHo;

        DrawConnections(mlp.NeuralNetwork.HiddenNodes, mlp.NeuralNetwork.OutputNodes, hiddenLayerPos, outputLayerPos,
            neuronDistance, drawList, hiddenWeights);
        DrawNeurons(mlp.NeuralNetwork.HiddenNodes, hiddenLayerPos, neuronDistance, neuronRadius, drawList,
            hiddenWeights.ToArray());
        DrawNeurons(mlp.NeuralNetwork.OutputNodes, outputLayerPos, neuronDistance, neuronRadius, drawList,
            outputWeights.ToArray());
    }

    private static void DrawNeurons(int neurons, Vector2 layerPos, float neuronDistance, float neuronRadius,
        ImDrawListPtr drawList, double[] weights) {
        for (var neuronIndex = 0; neuronIndex < neurons; neuronIndex++) {
            var neuronPos = layerPos + new Vector2(0, neuronDistance * neuronIndex);
            var weight = weights[neuronIndex];
            var color = new Vector4(1, 1, 1, Math.Max(0.1f, (float)weight));

            drawList.AddCircleFilled(neuronPos, neuronRadius, ImGui.ColorConvertFloat4ToU32(color));
            drawList.AddCircle(neuronPos, neuronRadius, ImGui.ColorConvertFloat4ToU32(new Vector4(1, 1, 1, 0.4f)));
        }
    }

    private static void DrawConnections(int fromLayer, int toLayer, Vector2 fromLayerPos, Vector2 toLayerPos,
        float neuronDistance, ImDrawListPtr drawList, Matrix weights) {
        for (var fromNeuronIndex = 0; fromNeuronIndex < fromLayer; fromNeuronIndex++) {
            var fromNeuronPos = fromLayerPos + new Vector2(0, neuronDistance * fromNeuronIndex);

            for (var toNeuronIndex = 0; toNeuronIndex < toLayer; toNeuronIndex++) {
                var toNeuronPos = toLayerPos + new Vector2(0, neuronDistance * toNeuronIndex);
                var weight = weights[fromNeuronIndex, toNeuronIndex];
                var color = new Vector4(1, 1, 1, Math.Max(0.1f, (float)weight));

                drawList.AddLine(fromNeuronPos, toNeuronPos, ImGui.ColorConvertFloat4ToU32(color));
            }
        }
    }
}