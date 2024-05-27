using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using ImGuiNET;
using Zoidberg.model;
using Zoidberg.ui;
using Zoidberg.utils;
using Raylib_cs;
using rlImGui_cs;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Zoidberg;

public struct Load {
    public int Curr;
    public int Max;
}

public static class ImageClassification {
    private static readonly (int width, int height) ScreenSize = (1200, 780);
    private static readonly Load TrainLoad = new() { Curr = 0, Max = 0 };
    private static readonly TrainingSettings TrainingSettings = new();

    public static int InputSize => (int)(ImageLoader.ImageSize.X * ImageLoader.ImageSize.Y);

    public static void Run() {
        ImageLoader.InitDatasets();

        Raylib.SetWindowState(ConfigFlags.ResizableWindow);
        Raylib.InitWindow(ScreenSize.width, ScreenSize.height, "Zoidberg");

        Theme.SetupFonts();
        rlImGui.Setup(true, true);
        Theme.ApplyTheme();

        while (!Raylib.WindowShouldClose()) Render();

        rlImGui.Shutdown();
        Raylib.CloseWindow();
    }

    private static void StartTraining() {
        switch (TrainingSettings.SelectedModel) {
            case PerceptronModel perceptron:
                perceptron.StartTraining(TrainingSettings);
                break;
            case MultiLayerPerceptronsModel mlp:
                mlp.StartTraining(TrainingSettings);
                break;
        }
    }

    public static void PredictImage(Image<Rgba32> image) {
        switch (TrainingSettings.SelectedModel) {
            case PerceptronModel perceptron:
                var predicted = perceptron.Predict(image);
                Console.WriteLine($"Predicted: {predicted}");
                break;
        }
    }

    public static void RenderImageDiff(string id, Image<Rgba32> image,
        (double[] values, double min, double max) diff,
        int renderSize) {
        var pos = ImGui.GetCursorScreenPos();
        var drawList = ImGui.GetWindowDrawList();

        ImGui.InvisibleButton(id, new Vector2(renderSize * image.Width, renderSize * image.Height));

        for (var i = 0; i < image.Width; i++)
        for (var j = 0; j < image.Height; j++) {
            var index = i * image.Width + j;
            var pixel = image[i, j];
            var normalizedDiff = (diff.values[index] - diff.min) / (diff.max - diff.min) - 1;

            var color = ImGui.ColorConvertFloat4ToU32(
                new Vector4(
                    pixel.R / 255f + (float)normalizedDiff,
                    pixel.G / 255f + (float)normalizedDiff,
                    pixel.B / 255f + (float)normalizedDiff,
                    1
                )
            );


            drawList.AddRectFilled(
                new Vector2(pos.X + i * renderSize, pos.Y + j * renderSize),
                new Vector2(pos.X + (i + 1) * renderSize, pos.Y + (j + 1) * renderSize),
                color
            );
        }
    }

    public static void RenderImage(string id, Image<Rgba32> image, int renderSize) {
        var pos = ImGui.GetCursorScreenPos();
        var drawList = ImGui.GetWindowDrawList();

        ImGui.InvisibleButton(id, new Vector2(renderSize * image.Width, renderSize * image.Height));

        for (var i = 0; i < image.Width; i++)
        for (var j = 0; j < image.Height; j++) {
            var pixel = image[i, j];
            var color = ImGui.ColorConvertFloat4ToU32(
                new Vector4(pixel.R / 255f, pixel.G / 255f, pixel.B / 255f, pixel.A / 255f));

            drawList.AddRectFilled(
                new Vector2(pos.X + i * renderSize, pos.Y + j * renderSize),
                new Vector2(pos.X + (i + 1) * renderSize, pos.Y + (j + 1) * renderSize),
                color
            );
        }
    }

    public static void RenderHeatMap(string id, (double[] values, double min, double max) heatmap, int renderSize) {
        var pos = ImGui.GetCursorScreenPos();
        var drawList = ImGui.GetWindowDrawList();

        ImGui.InvisibleButton(id,
            new Vector2(renderSize * ImageLoader.ImageSize.X, renderSize * ImageLoader.ImageSize.Y));

        for (var i = 0; i < ImageLoader.ImageSize.X; i++)
        for (var j = 0; j < ImageLoader.ImageSize.Y; j++) {
            var index = i * (int)ImageLoader.ImageSize.X + j;
            var value = heatmap.values[index];

            var normalizedValue = (value - heatmap.min) / (heatmap.max - heatmap.min);

            drawList.AddRectFilled(new Vector2(pos.X + i * renderSize, pos.Y + j * renderSize),
                new Vector2(pos.X + (i + 1) * renderSize, pos.Y + (j + 1) * renderSize),
                ImGui.ColorConvertFloat4ToU32(
                    new Vector4((float)normalizedValue, (float)normalizedValue, (float)normalizedValue, 1f)));
        }
    }

    private static void Render() {
        rlImGui.Begin();

        ImGui.PushFont(Theme.Fonts["Poppins-Regular"]);
        // ImGui.ShowDemoWindow();

        ImGui.SetNextWindowPos(Vector2.Zero, ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(Raylib.GetScreenWidth(), Raylib.GetScreenHeight()), ImGuiCond.Always);

        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(0, 0));
        ImGui.Begin("AI",
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove |
            ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse |
            ImGuiWindowFlags.NoBringToFrontOnFocus | ImGuiWindowFlags.NoNavFocus | ImGuiWindowFlags.NoNav);
        ImGui.PopStyleVar();

        ImGui.BeginChild("Left Menu", new Vector2(400, ImGui.GetWindowHeight()),
            ImGuiChildFlags.ResizeX | ImGuiChildFlags.AlwaysUseWindowPadding);

        ImGui.BeginChild("Training Settings", new Vector2(ImGui.GetColumnWidth(), 500),
            ImGuiChildFlags.ResizeY);
        ImGui.SeparatorText("Image Loading");
        
        int[] resolution = [(int)ImageLoader.ImageSize.X, (int)ImageLoader.ImageSize.Y];

        if (ImageLoader.IsLoading) {
            ImGui.BeginDisabled();
        }

        ImGui.DragInt2("Resolution", ref resolution[0], 1, 25, 300);

        ImageLoader.ImageSize = new Vector2(resolution[0], resolution[1]);
    
        if (ImGui.Button("Load datasets", new Vector2(ImGui.CalcItemWidth(), ImGui.GetFrameHeight()))) ImageLoader.LoadDatasets();
    
        
        if (ImageLoader.IsLoading) {
            ImGui.EndDisabled();
        }
            
        if (ImageLoader.IsLoading) {
            ImGui.ProgressBar(ImageLoader._imageLoad.Curr / (float)ImageLoader._imageLoad.Max,
                new Vector2(ImGui.CalcItemWidth(), 20));
        }
        
        ImGui.SeparatorText("Training");
        
        TrainingSettings.Render();
        
        if (TrainLoad.Curr == 0 || TrainLoad.Curr == TrainLoad.Max) {
            if (ImGui.Button("Start Training",new Vector2(ImGui.CalcItemWidth(), ImGui.GetFrameHeight()))) StartTraining();
        }
        
        else {
            ImGui.SameLine();
            ImGui.ProgressBar(TrainLoad.Curr / (float)TrainLoad.Max, new Vector2(ImGui.GetColumnWidth(), 20));
        }
        
        ImGui.EndChild();

        CustomConsole.Render();

        ImGui.EndChild();
        ImGui.SameLine();
        ImGui.BeginChild("Stats", new Vector2(ImGui.GetColumnWidth() / 2, ImGui.GetWindowHeight()),
            ImGuiChildFlags.AlwaysUseWindowPadding | ImGuiChildFlags.ResizeX);

        switch (TrainingSettings.SelectedModel) {
            case PerceptronModel perceptron:
                PerceptronStats.Render(perceptron);
                break;

            case MultiLayerPerceptronsModel mlp:
                MultiLayerPerceptronsStats.Render(mlp);
                break;
        }

        ImGui.EndChild();
        ImGui.SameLine();

        ImGui.BeginChild("Other", new Vector2(ImGui.GetColumnWidth(), ImGui.GetWindowHeight()),
            ImGuiChildFlags.AlwaysUseWindowPadding);

        switch (TrainingSettings.SelectedModel) {
            case PerceptronModel perceptron:
                PerceptronStats.RenderAdditional(perceptron);
                break;
        }

        // if (ImGui.Button("Reload Theme")) Theme.ApplyTheme();

        ImGui.EndChild();

        Raylib.BeginDrawing();
        Raylib.EndDrawing();

        ImGui.PopFont();
        ImGui.End();

        rlImGui.End();
    }
}