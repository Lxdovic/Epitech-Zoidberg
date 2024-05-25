using System.ComponentModel;
using System.Numerics;
using ImGuiNET;
using LearningAI.model;
using LearningAI.ui;
using Raylib_cs;
using rlImGui_cs;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Image = SixLabors.ImageSharp.Image;

namespace LearningAI;

public struct Load {
    public int Curr;
    public int Max;
}

public static class ImageClassification {
    private static readonly (int width, int height) ScreenSize = (1200, 780);
    public static readonly int[] ImageSize = [50, 50];
    private static readonly int[] ImageLoadingWorkerCount = [1, 1, 2];
    private static Load _imageLoad = new() { Curr = 0, Max = 0 };
    public static Load TrainLoad = new() { Curr = 0, Max = 0 };
    private static List<string> _trainImagesPaths = new();
    private static List<string> _valImagesPaths = new();
    private static List<string> _testImagesPaths = new();

    public static BackgroundWorker[] ImageloadingWorkers =
        new BackgroundWorker[ImageLoadingWorkerCount.Sum()];

    public static readonly List<(string label, Image<Rgba32> image)> TrainImages = new();
    public static readonly List<(string label, Image<Rgba32> image)> ValImages = new();
    public static readonly List<(string label, Image<Rgba32> image)> TestImages = new();

    private static readonly TrainingSettings TrainingSettings = new();

    public static int InputSize => ImageSize[0] * ImageSize[1];

    public static void Run() {
        InitDatasets();

        Raylib.SetWindowState(ConfigFlags.ResizableWindow);
        Raylib.InitWindow(ScreenSize.width, ScreenSize.height, "Image Classification");
        rlImGui.Setup(true, true);

        ImGui.StyleColorsClassic();

        while (!Raylib.WindowShouldClose()) Render();

        rlImGui.Shutdown();
        Raylib.CloseWindow();
    }

    private static void InitDatasets() {
        var trainDatasetFolder = Path.Combine(Environment.CurrentDirectory, "resources/dataset/TRAIN");
        var valDatasetFolder = Path.Combine(Environment.CurrentDirectory, "resources/dataset/VALIDATION");
        var testDatasetFolder = Path.Combine(Environment.CurrentDirectory, "resources/dataset/TEST");

        if (!Directory.Exists(trainDatasetFolder)) Console.WriteLine("Dataset folder not found for train.");
        if (!Directory.Exists(valDatasetFolder)) Console.WriteLine("Dataset folder not found for val.");
        if (!Directory.Exists(testDatasetFolder)) Console.WriteLine("Dataset folder not found for test.");

        _trainImagesPaths = Directory.GetFiles(trainDatasetFolder).ToList();
        _valImagesPaths = Directory.GetFiles(valDatasetFolder).ToList();
        _testImagesPaths = Directory.GetFiles(testDatasetFolder).ToList();
    }

    private static void InitImageLoadingWorkers() {
        ImageloadingWorkers = new BackgroundWorker[ImageLoadingWorkerCount.Sum()];

        for (var i = 0; i < ImageLoadingWorkerCount.Sum(); i++) {
            var workerIndex = i;

            ImageloadingWorkers[workerIndex] = new BackgroundWorker();
            ImageloadingWorkers[workerIndex].WorkerReportsProgress = true;

            if (workerIndex >= 0 && workerIndex < ImageLoadingWorkerCount[0]) {
                var from = workerIndex * _valImagesPaths.Count;
                var to = (workerIndex + 1) * _valImagesPaths.Count;

                ImageloadingWorkers[workerIndex].DoWork += (_, _) => { LoadValidationDataset(workerIndex, from, to); };
            }

            else if (workerIndex >= ImageLoadingWorkerCount[0] &&
                     workerIndex < ImageLoadingWorkerCount[0] + ImageLoadingWorkerCount[1]) {
                var from = (workerIndex - ImageLoadingWorkerCount[0]) * _testImagesPaths.Count /
                           ImageLoadingWorkerCount[1];
                var to = (workerIndex - ImageLoadingWorkerCount[0] + 1) * _testImagesPaths.Count /
                         ImageLoadingWorkerCount[1];

                ImageloadingWorkers[workerIndex].DoWork += (_, _) => { LoadTestDataset(workerIndex, from, to); };
            }

            else {
                var from = (workerIndex - ImageLoadingWorkerCount[0] - ImageLoadingWorkerCount[1]) *
                    _trainImagesPaths.Count / ImageLoadingWorkerCount[2];
                var to = (workerIndex - ImageLoadingWorkerCount[0] - ImageLoadingWorkerCount[1] + 1) *
                    _trainImagesPaths.Count / ImageLoadingWorkerCount[2];

                ImageloadingWorkers[workerIndex].DoWork += (_, _) => { LoadTrainDataset(workerIndex, from, to); };
            }

            ImageloadingWorkers[workerIndex].ProgressChanged += (_, _) => _imageLoad.Curr++;
        }
    }

    private static void RunImageLoadingWorkers() {
        for (var i = 0; i < ImageLoadingWorkerCount.Sum(); i++) {
            if (ImageloadingWorkers[i].IsBusy) continue;

            ImageloadingWorkers[i].RunWorkerAsync();
        }
    }

    private static void LoadDatasets() {
        TrainImages.Clear();
        ValImages.Clear();
        TestImages.Clear();

        _imageLoad.Curr = 0;
        _imageLoad.Max = _trainImagesPaths.Count + _valImagesPaths.Count + _testImagesPaths.Count;

        InitImageLoadingWorkers();
        RunImageLoadingWorkers();
    }

    private static void LoadTrainDataset(int workerIndex, int from, int to) {
        Console.WriteLine($"train workerIndex = {workerIndex} from = {from} to = {to}");

        for (var index = from; index < to; index++) {
            var path = _trainImagesPaths[index];
            var label = path.Contains("bacteria") ? "bacteria" : path.Contains("virus") ? "virus" : "negative";

            TrainImages.Add((label,
                LoadImage(path, new Vector2(ImageSize[0], ImageSize[1]))));

            ImageloadingWorkers[workerIndex].ReportProgress(0);
        }
    }

    private static void LoadTestDataset(int workerIndex, int from, int to) {
        Console.WriteLine($"test workerIndex = {workerIndex} from = {from} to = {to}");

        for (var index = from; index < to; index++) {
            var path = _testImagesPaths[index];
            var label = path.Contains("bacteria") ? "bacteria" : path.Contains("virus") ? "virus" : "negative";

            TestImages.Add((label,
                LoadImage(path, new Vector2(ImageSize[0], ImageSize[1]))));

            ImageloadingWorkers[workerIndex].ReportProgress(0);
        }
    }

    private static void LoadValidationDataset(int workerIndex, int from, int to) {
        Console.WriteLine($"val workerIndex = {workerIndex} from = {from} to = {to}");

        for (var index = from; index < to; index++) {
            var path = _valImagesPaths[index];
            var label = path.Contains("bacteria") ? "bacteria" : path.Contains("virus") ? "virus" : "negative";

            ValImages.Add((label,
                LoadImage(path, new Vector2(ImageSize[0], ImageSize[1]))));

            ImageloadingWorkers[workerIndex].ReportProgress(0);
        }
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

    private static Image<Rgba32> LoadImage(string path, Vector2 size) {
        var image = Image.Load<Rgba32>(path);

        image.Mutate(x => x.Resize((int)size.X, (int)size.Y));

        return image;
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
            var unnormalizedDiff = diff.values[index];
            var normalizedDiff = (unnormalizedDiff - diff.min) / (diff.max - diff.min);
            var color = ImGui.ColorConvertFloat4ToU32(
                new Vector4(pixel.R / 255f - (float)normalizedDiff, pixel.G / 255f - (float)normalizedDiff,
                    pixel.B / 255f - (float)normalizedDiff, pixel.A / 255f - (float)normalizedDiff));


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

        ImGui.InvisibleButton(id, new Vector2(renderSize * ImageSize[0], renderSize * ImageSize[1]));

        for (var i = 0; i < ImageSize[0]; i++)
        for (var j = 0; j < ImageSize[1]; j++) {
            var index = i * ImageSize[0] + j;
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

        // ImGui.ShowDemoWindow();

        ImGui.SetNextWindowPos(Vector2.Zero, ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(Raylib.GetScreenWidth(), Raylib.GetScreenHeight()), ImGuiCond.Always);

        ImGui.PushStyleVar(ImGuiStyleVar.WindowPadding, new Vector2(0, 0));
        ImGui.PushStyleVar(ImGuiStyleVar.FramePadding, new Vector2(4, 4));
        ImGui.PushStyleVar(ImGuiStyleVar.FrameRounding, 2f);
        ImGui.Begin("Settings",
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove |
            ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse |
            ImGuiWindowFlags.NoBringToFrontOnFocus | ImGuiWindowFlags.NoNavFocus | ImGuiWindowFlags.NoNav);

        ImGui.PushStyleColor(ImGuiCol.ChildBg, new Vector4(0.1f, 0.1f, 0.1f, 1));

        ImGui.BeginChild("Training Settings", new Vector2(400, ImGui.GetWindowHeight()), ImGuiChildFlags.ResizeX);

        ImGui.Text("Image Loading");
        ImGui.SliderInt2("Resolution", ref ImageSize[0], 25, 300);
        ImGui.Text("Image Loading Worker Distribution");
        ImGui.SliderInt("Validation", ref ImageLoadingWorkerCount[0], 1, 10);
        ImGui.SliderInt("Test", ref ImageLoadingWorkerCount[1], 1, 10);
        ImGui.SliderInt("Train", ref ImageLoadingWorkerCount[2], 1, 10);

        ImGui.Text("Training");

        TrainingSettings.Render();

        if (ImGui.Button("Load datasets")) LoadDatasets();

        if (_imageLoad.Curr > 0 && _imageLoad.Curr < _imageLoad.Max) {
            ImGui.SameLine();
            ImGui.ProgressBar(_imageLoad.Curr / (float)_imageLoad.Max, new Vector2(ImGui.GetColumnWidth(), 20));
            ImGui.Text($"Loading images... {_imageLoad.Curr}/{_imageLoad.Max}");
        }

        if (TrainLoad.Curr == 0 || TrainLoad.Curr == TrainLoad.Max) {
            if (ImGui.Button("Start Training")) StartTraining();
        }

        else {
            ImGui.SameLine();
            ImGui.ProgressBar(TrainLoad.Curr / (float)TrainLoad.Max, new Vector2(ImGui.GetColumnWidth(), 20));
        }

        ImGui.EndChild();
        ImGui.PopStyleColor();
        ImGui.SameLine();

        ImGui.BeginChild("Stats");

        switch (TrainingSettings.SelectedModel) {
            case PerceptronModel perceptron:
                PerceptronStats.Render(perceptron);
                break;

            case MultiLayerPerceptronsModel mlp:
                MultiLayerPerceptronsStats.Render(mlp);
                break;
        }

        ImGui.EndChild();
        ImGui.PopStyleVar(3);

        Raylib.BeginDrawing();
        Raylib.EndDrawing();

        ImGui.End();

        rlImGui.End();
    }
}