using System.ComponentModel;
using System.Numerics;
using ImGuiNET;
using Raylib_cs;
using rlImGui_cs;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Color = Raylib_cs.Color;
using Image = SixLabors.ImageSharp.Image;

namespace RosenblattPerceptrons;

public struct Load {
    public int Curr;
    public int Max;
}

public static class ImageClassification {
    private static readonly (int width, int height) ScreenSize = (1200, 780);
    private static readonly Vector2 ImageSize = new(50, 50);
    private static float _learningRate = 0.1f;
    private static readonly int[] ImageLoadingWorkerCount = [1, 1, 2];
    private static int _epochs = 10;
    private static bool _showBatch;
    private static int _displayImageMultiplier = 3;
    private static int _batchSize = 9;
    private static float _averageAccuracy;
    private static Load _imageLoad = new() { Curr = 0, Max = 0 };
    private static Load _trainLoad = new() { Curr = 0, Max = 0 };
    private static List<string> _trainImagesPaths = new();
    private static List<string> _valImagesPaths = new();
    private static List<string> _testImagesPaths = new();
    private static Perceptron _perceptron = new((int)(ImageSize.X * ImageSize.Y), _learningRate);

    private static BackgroundWorker[] _imageloadingWorkers =
        new BackgroundWorker[ImageLoadingWorkerCount.Sum()];

    private static BackgroundWorker _trainingWorker = new();
    private static readonly Random Random = new();
    private static readonly List<float> AccuracyHistory = new();
    private static readonly List<string> LastGuesses = new();
    private static readonly List<(string label, Image<Rgba32> image)> TrainImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> ValImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> TestImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> ImageBatch = new();

    public static void Run() {
        InitDatasets();

        Raylib.SetWindowState(ConfigFlags.ResizableWindow);
        Raylib.InitWindow(ScreenSize.width, ScreenSize.height, "Rosenblatt Perceptrons");
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
        _imageloadingWorkers = new BackgroundWorker[ImageLoadingWorkerCount.Sum()];

        for (var i = 0; i < ImageLoadingWorkerCount.Sum(); i++) {
            var workerIndex = i;

            _imageloadingWorkers[workerIndex] = new BackgroundWorker();
            _imageloadingWorkers[workerIndex].WorkerReportsProgress = true;

            if (workerIndex >= 0 && workerIndex < ImageLoadingWorkerCount[0]) {
                var from = workerIndex * _valImagesPaths.Count;
                var to = (workerIndex + 1) * _valImagesPaths.Count;

                _imageloadingWorkers[workerIndex].DoWork += (_, _) => { LoadValidationDataset(workerIndex, from, to); };
            }

            else if (workerIndex >= ImageLoadingWorkerCount[0] &&
                     workerIndex < ImageLoadingWorkerCount[0] + ImageLoadingWorkerCount[1]) {
                var from = (workerIndex - ImageLoadingWorkerCount[0]) * _testImagesPaths.Count /
                           ImageLoadingWorkerCount[1];
                var to = (workerIndex - ImageLoadingWorkerCount[0] + 1) * _testImagesPaths.Count /
                         ImageLoadingWorkerCount[1];

                _imageloadingWorkers[workerIndex].DoWork += (_, _) => { LoadTestDataset(workerIndex, from, to); };
            }

            else {
                var from = (workerIndex - ImageLoadingWorkerCount[0] - ImageLoadingWorkerCount[1]) *
                    _trainImagesPaths.Count / ImageLoadingWorkerCount[2];
                var to = (workerIndex - ImageLoadingWorkerCount[0] - ImageLoadingWorkerCount[1] + 1) *
                    _trainImagesPaths.Count / ImageLoadingWorkerCount[2];

                _imageloadingWorkers[workerIndex].DoWork += (_, _) => { LoadTrainDataset(workerIndex, from, to); };
            }

            _imageloadingWorkers[workerIndex].ProgressChanged += (_, _) => _imageLoad.Curr++;
        }
    }

    private static void RunImageLoadingWorkers() {
        for (var i = 0; i < ImageLoadingWorkerCount.Sum(); i++) {
            if (_imageloadingWorkers[i].IsBusy) continue;

            _imageloadingWorkers[i].RunWorkerAsync();
        }
    }

    private static void LoadDatasets() {
        if (_trainingWorker.IsBusy) return;

        _imageLoad.Curr = 0;
        _imageLoad.Max = _trainImagesPaths.Count + _valImagesPaths.Count + _testImagesPaths.Count;

        InitImageLoadingWorkers();
        RunImageLoadingWorkers();
    }

    private static void LoadTrainDataset(int workerIndex, int from, int to) {
        Console.WriteLine($"train workerIndex = {workerIndex} from = {from} to = {to}");

        for (var index = from; index < to; index++) {
            var path = _trainImagesPaths[index];
            var isPositive = path.Contains("bacteria") || path.Contains("virus");

            TrainImages.Add((isPositive ? "positive" : "negative",
                LoadImage(path, ImageSize)));

            _imageloadingWorkers[workerIndex].ReportProgress(0);
        }
    }

    private static void LoadTestDataset(int workerIndex, int from, int to) {
        Console.WriteLine($"test workerIndex = {workerIndex} from = {from} to = {to}");

        for (var index = from; index < to; index++) {
            var path = _testImagesPaths[index];
            var isPositive = path.Contains("bacteria") || path.Contains("virus");

            TestImages.Add((isPositive ? "positive" : "negative",
                LoadImage(path, ImageSize)));

            _imageloadingWorkers[workerIndex].ReportProgress(0);
        }
    }

    private static void LoadValidationDataset(int workerIndex, int from, int to) {
        Console.WriteLine($"val workerIndex = {workerIndex} from = {from} to = {to}");

        for (var index = from; index < to; index++) {
            var path = _valImagesPaths[index];
            var isPositive = path.Contains("bacteria") || path.Contains("virus");

            ValImages.Add((isPositive ? "positive" : "negative",
                LoadImage(path, ImageSize)));

            _imageloadingWorkers[workerIndex].ReportProgress(0);
        }
    }

    private static void StartTraining() {
        Console.WriteLine("Training...");

        AccuracyHistory.Clear();
        _perceptron = new Perceptron((int)(ImageSize.X * ImageSize.Y), _learningRate / 100);

        for (var i = 0; i < _epochs; i++) {
            _trainingWorker.ReportProgress(0);

            for (var j = 0; j < TrainImages.Count; j++) {
                var (label, image) = TrainImages[j];
                var pixels = new List<double>();

                for (var x = 0; x < image.Width; x++)
                for (var y = 0; y < image.Height; y++) {
                    var pixel = image[x, y];
                    var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0;
                    pixels.Add(grayscale);
                }

                _perceptron.Train([..pixels, 1], label == "positive" ? 1 : 0);
            }

            Validate();
        }

        Console.WriteLine("Finished training.");
    }

    private static void Validate() {
        var correct = 0;
        var total = 0;

        for (var i = 0; i < ValImages.Count; i++) {
            var (label, image) = ValImages[i];
            var pixels = new List<double>();

            for (var x = 0; x < image.Width; x++)
            for (var y = 0; y < image.Height; y++) {
                var pixel = image[x, y];
                var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0;
                pixels.Add(grayscale);
            }

            var guess = _perceptron.Activate([..pixels, 1]) == 1 ? "positive" : "negative";

            if (guess == label) correct++;

            total++;
        }

        var acc = (float)Math.Round(correct / (double)total * 100, 3);

        AccuracyHistory.Add(acc);
        _averageAccuracy = AccuracyHistory.Average();
    }

    private static void GuessBatch(int amount) {
        LastGuesses.Clear();
        ImageBatch.Clear();

        for (var i = 0; i < amount; i++) {
            var (label, image) = ValImages[Random.Next(0, TestImages.Count)];
            var pixels = new List<double>();

            for (var x = 0; x < image.Width; x++)
            for (var y = 0; y < image.Height; y++) {
                var pixel = image[x, y];
                var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0;
                pixels.Add(grayscale);
            }

            LastGuesses.Add(_perceptron.Activate([..pixels, 1]) == 1 ? "positive" : "negative");
            ImageBatch.Add((label, image));
        }

        _showBatch = true;
    }

    private static Image<Rgba32> LoadImage(string path, Vector2 size) {
        var image = Image.Load<Rgba32>(path);

        image.Mutate(x => x.Resize((int)size.X, (int)size.Y));

        return image;
    }

    private static void RenderImage(Image<Rgba32>? image, Vector2 pos, int size) {
        if (image == null) return;

        for (var i = 0; i < image.Width * size; i++)
        for (var j = 0; j < image.Height * size; j++) {
            var pixel = image[i / size, j / size];
            Raylib.DrawPixel((int)pos.X + i, (int)pos.Y + j, new Color(pixel.R, pixel.G, pixel.B, pixel.A));
        }
    }

    private static void RenderBatch(List<(string label, Image<Rgba32> image)> images, int size) {
        var inGuiDrawPos = ImGui.GetCursorScreenPos();
        var nextY = inGuiDrawPos.Y;

        for (var i = 0; i < images.Count; i++) {
            var (label, image) = images[i];
            var guessLabel = LastGuesses[i];
            var x = inGuiDrawPos.X + (float)(i % Math.Sqrt(images.Count)) * ImageSize.X * size;
            var y = inGuiDrawPos.Y + (float)Math.Floor(i / Math.Sqrt(images.Count)) * ImageSize.Y * size;
            var color = Color.Black;

            if (label == "negative" && guessLabel == "negative") color = Color.Green;
            else if (label == "positive" && guessLabel == "positive") color = Color.Blue;
            else if (label == "negative" && guessLabel == "positive") color = Color.Red;
            else if (label == "positive" && guessLabel == "negative") color = Color.Yellow;

            RenderImage(image, new Vector2(x, y), size);

            Raylib.DrawText(guessLabel, (int)x + 2, (int)y + 2, 20, color);

            nextY = y + image.Height * size + image.Height * size + 6;
        }

        ImGui.SetCursorScreenPos(inGuiDrawPos with { Y = nextY });
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
            ImGuiWindowFlags.MenuBar | ImGuiWindowFlags.NoBringToFrontOnFocus | ImGuiWindowFlags.NoNavFocus |
            ImGuiWindowFlags.NoNav);

        ImGui.PushStyleColor(ImGuiCol.ChildBg, new Vector4(0.1f, 0.1f, 0.1f, 1));
        ImGui.BeginChild("Settings", new Vector2(400, ImGui.GetWindowHeight()), ImGuiChildFlags.ResizeX);
        ImGui.SliderInt("Epochs", ref _epochs, 0, 100);
        ImGui.SliderFloat("Learning rate", ref _learningRate, 0.001f, 1f);
        ImGui.SliderInt("Batch size", ref _batchSize, 1, 16);
        ImGui.SliderInt("Display Size", ref _displayImageMultiplier, 1, 10);
        ImGui.Text("Image Loading Worker Distribution");
        ImGui.SliderInt("Validation", ref ImageLoadingWorkerCount[0], 1, 10);
        ImGui.SliderInt("Test", ref ImageLoadingWorkerCount[1], 1, 10);
        ImGui.SliderInt("Train", ref ImageLoadingWorkerCount[2], 1, 10);

        if (ImGui.Button("Load datasets")) LoadDatasets();

        if (_imageLoad.Curr > 0 && _imageLoad.Curr < _imageLoad.Max) {
            ImGui.SameLine();
            ImGui.ProgressBar(_imageLoad.Curr / (float)_imageLoad.Max, new Vector2(ImGui.GetColumnWidth(), 20));
            ImGui.Text($"Loading images... {_imageLoad.Curr}/{_imageLoad.Max}");
        }

        if (_trainLoad.Curr == 0 || _trainLoad.Curr == _trainLoad.Max) {
            if (ImGui.Button("Start Training")) {
                if (_trainingWorker.IsBusy || _imageloadingWorkers.Any(w => w.IsBusy)) return;

                _trainingWorker = new BackgroundWorker();

                _trainLoad.Curr = 0;
                _trainLoad.Max = _epochs;

                _trainingWorker.WorkerReportsProgress = true;
                _trainingWorker.DoWork += (_, _) => { StartTraining(); };
                _trainingWorker.ProgressChanged += (_, _) => _trainLoad.Curr++;
                _trainingWorker.RunWorkerAsync();
            }
        }

        else {
            if (ImGui.Button("Abort Training")) {
                _trainingWorker.CancelAsync();
                _trainLoad.Curr = 0;
            }

            ImGui.SameLine();
            ImGui.ProgressBar(_trainLoad.Curr / (float)_trainLoad.Max, new Vector2(ImGui.GetColumnWidth(), 20));
        }

        ImGui.EndChild();
        ImGui.PopStyleColor();
        ImGui.SameLine();

        ImGui.BeginChild("Images", new Vector2(ImGui.GetWindowWidth(), ImGui.GetWindowHeight()));

        if (AccuracyHistory.Count > 0) {
            var displayWidth = (float)Math.Sqrt(_batchSize) * ImageSize.X * _displayImageMultiplier;
            var accuracy = AccuracyHistory.ToArray();

            ImGui.PlotLines("##AccuracyHistory", ref accuracy[0], AccuracyHistory.Count, 0, null, AccuracyHistory.Min(),
                100,
                new Vector2(displayWidth, 100));

            ImGui.SameLine();
            ImGui.SetWindowFontScale(5);
            ImGui.TextColored(new Vector4(1, 1, 1, 1), $"{Math.Round(_averageAccuracy, 2)}%%");
            ImGui.SetWindowFontScale(1);
        }

        Raylib.BeginDrawing();

        if (_showBatch) RenderBatch(ImageBatch, _displayImageMultiplier);

        Raylib.EndDrawing();

        if (ImGui.Button("Guess")) GuessBatch(_batchSize);

        ImGui.EndChild();
        ImGui.PopStyleVar(3);
        ImGui.End();

        rlImGui.End();
    }
}