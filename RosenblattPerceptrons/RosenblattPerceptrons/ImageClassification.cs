using System.ComponentModel;
using System.Numerics;
using System.Runtime.InteropServices;
using ImGuiNET;
using Raylib_cs;
using rlImGui_cs;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Color = Raylib_cs.Color;
using Image = SixLabors.ImageSharp.Image;

namespace RosenblattPerceptrons;

public static class ImageClassification {
    private static readonly (int width, int height) ScreenSize = (1200, 780);
    private static float _learningRate = 0.0001f;
    private static int _epochs = 10;
    private static int _loadingProgress;
    private static bool _showBatch;
    private static int _totalImagesToLoad = 1;
    private static string _accuracy = "";
    private static readonly BackgroundWorker Worker = new();
    private static readonly List<string> LastGuesses = new();
    private static Perceptron _perceptron = new(50 * 50, _learningRate);
    private static readonly List<(string label, Image<Rgba32> image)> TrainImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> ValImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> TestImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> _imageBatch = new();

    public static void Run() {
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

        var trainImagesPaths = Directory.GetFiles(trainDatasetFolder);
        var valImagesPaths = Directory.GetFiles(valDatasetFolder);
        var testImagesPaths = Directory.GetFiles(testDatasetFolder);

        _totalImagesToLoad = trainImagesPaths.Length + valImagesPaths.Length + testImagesPaths.Length;

        for (var i = 0; i < trainImagesPaths.Length; i++) {
            var isPositive = trainImagesPaths[i].Contains("bacteria") || trainImagesPaths[i].Contains("virus");

            TrainImages.Add((isPositive ? "positive" : "negative",
                LoadImage(trainImagesPaths[i], new Vector2(50, 50))));

            Worker.ReportProgress(0);
        }

        for (var i = 0; i < valImagesPaths.Length; i++) {
            var isPositive = valImagesPaths[i].Contains("bacteria") || valImagesPaths[i].Contains("virus");

            ValImages.Add((isPositive ? "positive" : "negative",
                LoadImage(valImagesPaths[i], new Vector2(50, 50))));

            Worker.ReportProgress(0);
        }

        for (var i = 0; i < testImagesPaths.Length; i++) {
            var isPositive = testImagesPaths[i].Contains("bacteria") || testImagesPaths[i].Contains("virus");

            TestImages.Add((isPositive ? "positive" : "negative",
                LoadImage(testImagesPaths[i], new Vector2(50, 50))));

            Worker.ReportProgress(0);
        }
    }

    private static void StartTraining() {
        _perceptron = new Perceptron(50 * 50, _learningRate);

        for (var i = 0; i < _epochs; i++)
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

        _accuracy = $"{correct} / {total} correct. ({Math.Round(correct / (double)total * 100, 3)}%)";
    }

    private static void GuessBatch(int amount) {
        var random = new Random();

        LastGuesses.Clear();
        _imageBatch.Clear();

        for (var i = 0; i < amount; i++) {
            var (label, image) = TestImages[random.Next(0, TestImages.Count)];
            var pixels = new List<double>();

            for (var x = 0; x < image.Width; x++)
            for (var y = 0; y < image.Height; y++) {
                var pixel = image[x, y];
                var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0;
                pixels.Add(grayscale);
            }

            LastGuesses.Add(_perceptron.Activate([..pixels, 1]) == 1 ? "positive" : "negative");
            _imageBatch.Add((label, image));
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
            var x = inGuiDrawPos.X + (float)(i % Math.Sqrt(images.Count) - 1) * 50 * size;
            var y = inGuiDrawPos.Y + (float)Math.Floor(i / Math.Sqrt(images.Count) - 1) * 50 * size;

            RenderImageWithLabels(image, new Vector2(x + image.Width * size, y + image.Height * size), size,
                [label, guessLabel]);
        }
    }

    private static void Render() {
        rlImGui.Begin();

        ImGui.SetNextWindowPos(Vector2.Zero, ImGuiCond.Always);
        ImGui.SetNextWindowSize(new Vector2(Raylib.GetScreenWidth(), Raylib.GetScreenHeight()), ImGuiCond.Always);

        ImGui.Begin("Settings",
            ImGuiWindowFlags.NoCollapse | ImGuiWindowFlags.NoResize | ImGuiWindowFlags.NoMove |
            ImGuiWindowFlags.NoTitleBar | ImGuiWindowFlags.NoScrollbar | ImGuiWindowFlags.NoScrollWithMouse |
            ImGuiWindowFlags.MenuBar | ImGuiWindowFlags.NoBringToFrontOnFocus | ImGuiWindowFlags.NoNavFocus |
            ImGuiWindowFlags.NoNav);

        ImGui.BeginChild("Settings", new Vector2(400, ImGui.GetWindowHeight()), ImGuiChildFlags.ResizeX);
        ImGui.SliderInt("Epochs", ref _epochs, 0, 100);
        ImGui.SliderFloat("Learning rate", ref _learningRate, 0.0001f, 0.1f);

        if (ImGui.Button("Load Data")) {
            Worker.WorkerReportsProgress = true;
            Worker.DoWork += (sender, args) => InitDatasets();
            Worker.ProgressChanged += (sender, args) => { _loadingProgress++; };
            Worker.RunWorkerAsync();
        }

        if (_loadingProgress > 0 && _loadingProgress < _totalImagesToLoad) {
            ImGui.SameLine();
            ImGui.ProgressBar(_loadingProgress / (float)_totalImagesToLoad, new Vector2(200, 20));
        }

        if (ImGui.Button("Start Training")) StartTraining();

        if (_accuracy != "") {
            ImGui.SameLine();
            ImGui.Text(_accuracy);
        }

        if (ImGui.Button("Guess")) GuessBatch(9);

        ImGui.EndChild();
        ImGui.SameLine();


        ImGui.BeginChild("Images", new Vector2(ImGui.GetWindowWidth(), ImGui.GetWindowHeight()));

        ImGui.Text("Batch");
        Raylib.BeginDrawing();
        if (_showBatch) RenderBatch(_imageBatch, 3);
        Raylib.EndDrawing();
        ImGui.EndChild();
        ImGui.End();

        rlImGui.End();
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void RenderDelegate(IntPtr data);
}