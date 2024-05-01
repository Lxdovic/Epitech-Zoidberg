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
    private static float _learningRate = 0.001f;
    private static int _epochs = 10;
    private static bool _showBatch;
    private static Load _imageLoad = new() { Curr = 0, Max = 0 };
    private static Load _trainLoad = new() { Curr = 0, Max = 0 };
    private static List<string> _trainImagesPaths = new();
    private static List<string> _valImagesPaths = new();
    private static List<string> _testImagesPaths = new();
    private static Perceptron _perceptron = new(50 * 50, _learningRate);
    private static BackgroundWorker _imageLoadingWorker = new();
    private static BackgroundWorker _trainingWorker = new();
    private static readonly Vector2 ImageSize = new(50, 50);
    private static readonly Random Random = new();
    private static readonly List<float> Accuracy = new();
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

        while (!Raylib.WindowShouldClose()) Render();

        rlImGui.Shutdown();
        Raylib.CloseWindow();
    }

    private static void InitDatasets() {
        var trainDatasetFolder = Path.Combine(Environment.CurrentDirectory, "dataset/TRAIN");
        var valDatasetFolder = Path.Combine(Environment.CurrentDirectory, "dataset/VALIDATION");
        var testDatasetFolder = Path.Combine(Environment.CurrentDirectory, "dataset/TEST");

        if (!Directory.Exists(trainDatasetFolder)) Console.WriteLine("Dataset folder not found for train.");
        if (!Directory.Exists(valDatasetFolder)) Console.WriteLine("Dataset folder not found for val.");
        if (!Directory.Exists(testDatasetFolder)) Console.WriteLine("Dataset folder not found for test.");

        _trainImagesPaths = Directory.GetFiles(trainDatasetFolder).ToList();
        _valImagesPaths = Directory.GetFiles(valDatasetFolder).ToList();
        _testImagesPaths = Directory.GetFiles(testDatasetFolder).ToList();
    }

    private static void LoadDatasets() {
        Console.WriteLine("Loading datasets...");

        for (var index = 0; index < _trainImagesPaths.Count; index++) {
            var path = _trainImagesPaths[index];
            var isPositive = path.Contains("bacteria") || path.Contains("virus");

            TrainImages.Add((isPositive ? "positive" : "negative",
                LoadImage(path, ImageSize)));

            _imageLoadingWorker.ReportProgress(0);
        }

        for (var index = 0; index < _valImagesPaths.Count; index++) {
            var path = _valImagesPaths[index];
            var isPositive = path.Contains("bacteria") || path.Contains("virus");

            ValImages.Add((isPositive ? "positive" : "negative",
                LoadImage(path, ImageSize)));

            _imageLoadingWorker.ReportProgress(0);
        }

        for (var index = 0; index < _testImagesPaths.Count; index++) {
            var path = _testImagesPaths[index];
            var isPositive = path.Contains("bacteria") || path.Contains("virus");

            TestImages.Add((isPositive ? "positive" : "negative",
                LoadImage(path, ImageSize)));

            _imageLoadingWorker.ReportProgress(0);
        }

        Console.WriteLine("Finished loading datasets.");
    }

    private static void StartTraining() {
        Console.WriteLine("Training...");

        Accuracy.Clear();
        _perceptron = new Perceptron((int)(ImageSize.X * ImageSize.Y), _learningRate);

        for (var i = 0; i < _epochs; i++) {
            _trainingWorker.ReportProgress(0);

            for (var j = 0; j < TrainImages.Count; j++) {
                if (j % 1000 == 0) Validate();

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

        Accuracy.Add((float)Math.Round(correct / (double)total * 100, 3));
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

    private static void RenderImageWithLabels(Image<Rgba32>? image, Vector2 pos, int size, List<string> labels) {
        if (image == null) return;

        RenderImage(image, pos, size);

        for (var i = 0; i < labels.Count; i++)
            Raylib.DrawText(labels[i], (int)pos.X, (int)pos.Y + 20 * i, 20, i % 2 == 0 ? Color.Blue : Color.Red);
    }

    private static void RenderBatch(List<(string label, Image<Rgba32> image)> images, int size) {
        var inGuiDrawPos = ImGui.GetCursorScreenPos();

        for (var i = 0; i < images.Count; i++) {
            var (label, image) = images[i];
            var guessLabel = LastGuesses[i];
            var x = inGuiDrawPos.X + (float)(i % Math.Sqrt(images.Count) - 1) * ImageSize.X * size;
            var y = inGuiDrawPos.Y + (float)Math.Floor(i / Math.Sqrt(images.Count) - 1) * ImageSize.Y * size;

            RenderImageWithLabels(image, new Vector2(x + image.Width * size, y + image.Height * size), size,
                [label, guessLabel]);
        }
    }

    private static void Render() {
        rlImGui.Begin();

        // ImGui.ShowDemoWindow();

        ImGui.SetNextWindowPos(Vector2.Zero, ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(Raylib.GetScreenWidth(), Raylib.GetScreenHeight()), ImGuiCond.Always);

        ImGui.Begin("Settings",
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove |
            ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse |
            ImGuiWindowFlags.MenuBar | ImGuiWindowFlags.NoBringToFrontOnFocus | ImGuiWindowFlags.NoNavFocus |
            ImGuiWindowFlags.NoNav);

        ImGui.BeginChild("Settings", new Vector2(400, ImGui.GetWindowHeight()), ImGuiChildFlags.ResizeX);
        ImGui.SliderInt("Epochs", ref _epochs, 0, 100);
        ImGui.SliderFloat("Learning rate", ref _learningRate, 0.001f, 0.1f);

        if (ImGui.Button("Load datasets")) {
            if (_imageLoadingWorker.IsBusy) return;

            _imageLoadingWorker = new BackgroundWorker();

            _imageLoad.Curr = 0;
            _imageLoad.Max = _trainImagesPaths.Count + _valImagesPaths.Count + _testImagesPaths.Count;

            _imageLoadingWorker.WorkerReportsProgress = true;
            _imageLoadingWorker.DoWork += (_, _) => { LoadDatasets(); };
            _imageLoadingWorker.ProgressChanged += (_, _) => _imageLoad.Curr++;
            _imageLoadingWorker.RunWorkerAsync();
        }

        if (_imageLoad.Curr > 0 && _imageLoad.Curr < _imageLoad.Max) {
            ImGui.SameLine();
            ImGui.ProgressBar(_imageLoad.Curr / (float)_imageLoad.Max, new Vector2(ImGui.GetColumnWidth(), 20));
        }

        if (ImGui.Button("Start Training")) {
            if (_trainingWorker.IsBusy || _imageLoadingWorker.IsBusy) return;

            _trainingWorker = new BackgroundWorker();

            _trainLoad.Curr = 0;
            _trainLoad.Max = _epochs;

            _trainingWorker.WorkerReportsProgress = true;
            _trainingWorker.DoWork += (_, _) => { StartTraining(); };
            _trainingWorker.ProgressChanged += (_, _) => _trainLoad.Curr++;
            _trainingWorker.RunWorkerAsync();
        }

        if (_trainLoad.Curr > 0 && _trainLoad.Curr < _trainLoad.Max) {
            ImGui.SameLine();
            ImGui.ProgressBar(_trainLoad.Curr / (float)_trainLoad.Max, new Vector2(ImGui.GetColumnWidth(), 20));
        }

        if (ImGui.Button("Guess")) GuessBatch(9);

        ImGui.EndChild();
        ImGui.SameLine();

        ImGui.BeginChild("Images", new Vector2(ImGui.GetWindowWidth(), ImGui.GetWindowHeight()));

        ImGui.Text("Batch");
        Raylib.BeginDrawing();

        if (_showBatch) RenderBatch(ImageBatch, 3);

        Raylib.EndDrawing();

        if (Accuracy.Count > 0) {
            var accuracy = Accuracy.ToArray();

            ImGui.PlotLines("Accuracy", ref accuracy[0], Accuracy.Count, 0, null, 0, 100, new Vector2(500, 100));
        }

        ImGui.EndChild();
        ImGui.End();

        rlImGui.End();
    }
}