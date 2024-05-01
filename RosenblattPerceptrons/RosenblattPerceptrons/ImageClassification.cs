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
    private static float _learningRate = 0.001f;
    private static int _epochs = 10;
    private static bool _showBatch;
    private static int _displayImageMultiplier = 3;
    private static int _batchSize = 16;
    private static Load _imageLoad = new() { Curr = 0, Max = 0 };
    private static Load _trainLoad = new() { Curr = 0, Max = 0 };
    private static List<string> _trainImagesPaths = new();
    private static List<string> _valImagesPaths = new();
    private static List<string> _testImagesPaths = new();
    private static Perceptron _perceptron = new((int)(ImageSize.X * ImageSize.Y), _learningRate);
    private static BackgroundWorker _trainImageLoadingWorker = new();
    private static BackgroundWorker _valImageLoadingWorker = new();
    private static BackgroundWorker _testImageLoadingWorker = new();
    private static BackgroundWorker _trainingWorker = new();
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
        if (_trainingWorker.IsBusy || _valImageLoadingWorker.IsBusy || _testImageLoadingWorker.IsBusy) return;

        _trainImageLoadingWorker = new BackgroundWorker();
        _valImageLoadingWorker = new BackgroundWorker();
        _testImageLoadingWorker = new BackgroundWorker();

        _imageLoad.Curr = 0;
        _imageLoad.Max = _trainImagesPaths.Count + _valImagesPaths.Count + _testImagesPaths.Count;

        _trainImageLoadingWorker.WorkerReportsProgress = true;
        _valImageLoadingWorker.WorkerReportsProgress = true;
        _testImageLoadingWorker.WorkerReportsProgress = true;
        _trainImageLoadingWorker.DoWork += (_, _) => { LoadTrainDataset(); };
        _valImageLoadingWorker.DoWork += (_, _) => { LoadValidationDataset(); };
        _testImageLoadingWorker.DoWork += (_, _) => { LoadTestDataset(); };
        _trainImageLoadingWorker.ProgressChanged += (_, _) => _imageLoad.Curr++;
        _valImageLoadingWorker.ProgressChanged += (_, _) => _imageLoad.Curr++;
        _testImageLoadingWorker.ProgressChanged += (_, _) => _imageLoad.Curr++;

        _trainImageLoadingWorker.RunWorkerAsync();
        _valImageLoadingWorker.RunWorkerAsync();
        _testImageLoadingWorker.RunWorkerAsync();
    }

    private static void LoadTrainDataset() {
        for (var index = 0; index < _trainImagesPaths.Count; index++) {
            var path = _trainImagesPaths[index];
            var isPositive = path.Contains("bacteria") || path.Contains("virus");

            TrainImages.Add((isPositive ? "positive" : "negative",
                LoadImage(path, ImageSize)));

            _trainImageLoadingWorker.ReportProgress(0);
        }
    }

    private static void LoadTestDataset() {
        for (var index = 0; index < _testImagesPaths.Count; index++) {
            var path = _testImagesPaths[index];
            var isPositive = path.Contains("bacteria") || path.Contains("virus");

            TestImages.Add((isPositive ? "positive" : "negative",
                LoadImage(path, ImageSize)));

            _testImageLoadingWorker.ReportProgress(0);
        }
    }

    private static void LoadValidationDataset() {
        for (var index = 0; index < _valImagesPaths.Count; index++) {
            var path = _valImagesPaths[index];
            var isPositive = path.Contains("bacteria") || path.Contains("virus");

            ValImages.Add((isPositive ? "positive" : "negative",
                LoadImage(path, ImageSize)));

            _valImageLoadingWorker.ReportProgress(0);
        }
    }

    private static void StartTraining() {
        Console.WriteLine("Training...");

        Accuracy.Clear();
        _perceptron = new Perceptron((int)(ImageSize.X * ImageSize.Y), _learningRate / 1000);

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
        ImGui.Begin("Settings",
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove |
            ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse |
            ImGuiWindowFlags.MenuBar | ImGuiWindowFlags.NoBringToFrontOnFocus | ImGuiWindowFlags.NoNavFocus |
            ImGuiWindowFlags.NoNav);

        ImGui.PushStyleColor(ImGuiCol.ChildBg, new Vector4(0.1f, 0.1f, 0.1f, 1));
        ImGui.BeginChild("Settings", new Vector2(400, ImGui.GetWindowHeight()), ImGuiChildFlags.ResizeX);
        ImGui.SliderInt("Epochs", ref _epochs, 0, 100);
        ImGui.SliderFloat("Learning rate", ref _learningRate, 0.000001f, 0.1f);
        ImGui.SliderInt("Batch size", ref _batchSize, 1, 16);
        ImGui.SliderInt("Image display multiplier", ref _displayImageMultiplier, 1, 10);

        if (ImGui.Button("Load datasets")) LoadDatasets();

        if (_imageLoad.Curr > 0 && _imageLoad.Curr < _imageLoad.Max) {
            ImGui.SameLine();
            ImGui.ProgressBar(_imageLoad.Curr / (float)_imageLoad.Max, new Vector2(ImGui.GetColumnWidth(), 20));
        }

        if (ImGui.Button("Start Training")) {
            if (_trainingWorker.IsBusy || _valImageLoadingWorker.IsBusy || _testImageLoadingWorker.IsBusy ||
                _trainImageLoadingWorker.IsBusy) return;

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

        if (ImGui.Button("Guess")) GuessBatch(_batchSize);

        ImGui.EndChild();
        ImGui.PopStyleColor();
        ImGui.SameLine();

        ImGui.BeginChild("Images", new Vector2(ImGui.GetWindowWidth(), ImGui.GetWindowHeight()));

        var displayWidth = (float)Math.Sqrt(ImageBatch.Count) * ImageSize.X * _displayImageMultiplier;

        if (Accuracy.Count > 0) {
            var accuracy = Accuracy.ToArray();

            ImGui.PlotLines("Accuracy", ref accuracy[0], Accuracy.Count, 0, null, 0, 100,
                new Vector2(displayWidth, 100));
        }

        Raylib.BeginDrawing();

        if (_showBatch) RenderBatch(ImageBatch, _displayImageMultiplier);

        Raylib.EndDrawing();

        ImGui.EndChild();
        ImGui.PopStyleVar(2);
        ImGui.End();

        rlImGui.End();
    }
}