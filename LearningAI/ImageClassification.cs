using System.ComponentModel;
using System.Numerics;
using ImGuiNET;
using LearningAI.utils;
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
    public static float _learningRate = 0.1f;
    public static float _learningRateMin = 0.001f;
    public static float _learningRateMax = 0.1f;
    public static float _decayRate = 0.05f;
    public static float _decayFactor = 0.5f;
    public static int MlpHiddenLayerSize = 4;
    private static readonly int[] ImageLoadingWorkerCount = [1, 1, 2];
    public static int _epochs = 10;
    public static int _stepSize = 4;
    public static int _selectedScheduler;
    public static int _selectedModel;
    private static Load _imageLoad = new() { Curr = 0, Max = 0 };
    public static Load TrainLoad = new() { Curr = 0, Max = 0 };
    private static List<string> _trainImagesPaths = new();
    private static List<string> _valImagesPaths = new();
    private static List<string> _testImagesPaths = new();

    public static BackgroundWorker[] ImageloadingWorkers =
        new BackgroundWorker[ImageLoadingWorkerCount.Sum()];

    public static readonly string[] LearningRateSchedulers =
        { "None", "Step Decay", "Expo Decay", "Cosine Annealing" };

    public static readonly string[] Models =
        { "Perceptron", "Multi-Layer Perceptrons" };

    public static readonly List<(string label, Image<Rgba32> image)> TrainImages = new();
    public static readonly List<(string label, Image<Rgba32> image)> ValImages = new();
    public static readonly List<(string label, Image<Rgba32> image)> TestImages = new();

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
        if (PerceptronsTrainer.TrainingWorker.IsBusy) return;
        if (MultiLayerPerceptronsTrainer.TrainingWorker.IsBusy) return;

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

    public static double GetLearningRate(int epoch) {
        return _selectedScheduler switch {
            0 => _learningRate,
            1 => _learningRate * Math.Pow(_decayFactor, Math.Floor((1 + epoch) / (double)_stepSize)),
            2 => _learningRate * Math.Exp(-_decayRate * epoch),
            3 => _learningRateMin + 0.5 * (_learningRateMax - _learningRateMin) *
                (1 + Math.Cos(epoch / (double)_epochs * Math.PI)),
            _ => throw new ArgumentOutOfRangeException()
        };
    }

    private static void StartTraining() {
        switch (_selectedModel) {
            case 0:
                PerceptronsTrainer.StartTraining(ImageSize[0] * ImageSize[1], _epochs);
                break;
            case 1:
                MultiLayerPerceptronsTrainer.StartTraining(ImageSize[0] * ImageSize[1], _epochs);
                break;
        }
    }

    private static Image<Rgba32> LoadImage(string path, Vector2 size) {
        var image = Image.Load<Rgba32>(path);

        image.Mutate(x => x.Resize((int)size.X, (int)size.Y));

        return image;
    }

    public static void RenderHeatMap(string id, (double[] values, double min, double max) heatmap, int renderSize) {
        var amount = (int)Math.Sqrt(heatmap.values.Length);
        var pos = ImGui.GetCursorScreenPos();
        var drawList = ImGui.GetWindowDrawList();

        ImGui.InvisibleButton(id, new Vector2(renderSize * amount, renderSize * amount));

        for (var i = 0; i < amount; i++)
        for (var j = 0; j < amount; j++) {
            var index = i * amount + j;
            var value = heatmap.values[index];

            var normalizedValue = (value - heatmap.min) / (heatmap.max - heatmap.min);

            if (i == 0) normalizedValue = heatmap.max;

            drawList.AddRectFilled(new Vector2(pos.X + i * renderSize, pos.Y + j * renderSize),
                new Vector2(pos.X + (i + renderSize) * renderSize, pos.Y + (j + renderSize) * renderSize),
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

        ImGui.BeginChild("Settings", new Vector2(400, ImGui.GetWindowHeight()), ImGuiChildFlags.ResizeX);

        ImGui.Combo("Model", ref _selectedModel, Models,
            Models.Length);
        if (Models[_selectedModel] == "Multi-Layer Perceptrons")
            ImGui.SliderInt("Hidden Layer Size", ref MlpHiddenLayerSize, 1, 32);
        ImGui.Text("Image Loading");
        ImGui.SliderInt2("Resolution", ref ImageSize[0], 25, 300);
        ImGui.Text("Image Loading Worker Distribution");
        ImGui.SliderInt("Validation", ref ImageLoadingWorkerCount[0], 1, 10);
        ImGui.SliderInt("Test", ref ImageLoadingWorkerCount[1], 1, 10);
        ImGui.SliderInt("Train", ref ImageLoadingWorkerCount[2], 1, 10);

        ImGui.Text("Training");
        ImGui.SliderInt("Epochs", ref _epochs, 0, 100);
        ImGui.Combo("Learning Rate Scheduler", ref _selectedScheduler, LearningRateSchedulers,
            LearningRateSchedulers.Length);

        switch (_selectedScheduler) {
            case 0:
                ImGui.SliderFloat("Learning Rate", ref _learningRate, 0.001f, 1f);
                break;
            case 1:
                ImGui.SliderFloat("Learning Rate", ref _learningRate, 0.001f, 1f);
                ImGui.SliderFloat("Decay Factor", ref _decayFactor, 0.1f, 1f);
                ImGui.SliderInt("Step Size", ref _stepSize, 1, 10);
                break;

            case 2:
                ImGui.SliderFloat("Learning Rate", ref _learningRate, 0.001f, 1f);
                ImGui.SliderFloat("Decay Rate", ref _decayRate, 0.001f, 1f);
                break;

            case 3:
                ImGui.SliderFloat("Learning Rate Min", ref _learningRateMin, 0.001f, _learningRateMax);
                ImGui.SliderFloat("Learning Rate Max", ref _learningRateMax, _learningRateMin, 1f);
                ImGui.Text("Cosine Annealing");
                break;
        }

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

        if (ImGui.Button("Start Routines")) TrainingRoutine.StartRoutines(TrainingRoutine.CreateRoutines());


        ImGui.EndChild();
        ImGui.PopStyleColor();
        ImGui.SameLine();

        ImGui.BeginChild("Images");

        switch (_selectedModel) {
            case 0:
                PerceptronsTrainer.Render();
                break;
            case 1:
                MultiLayerPerceptronsTrainer.Render();
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