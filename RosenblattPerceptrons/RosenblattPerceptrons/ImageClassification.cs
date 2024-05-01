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

public static class ImageClassification {
    private static readonly (int width, int height) ScreenSize = (1200, 780);
    private static float _learningRate = 0.0001f;
    private static int _epochs = 10;
    private static readonly List<(string label, Image<Rgba32> image)> TrainImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> ValImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> TestImages = new();
    private static Perceptron Perceptron;
    private static (string label, Image<Rgba32> image) LastImage;
    private static double LastGuess;

    public static void Run() {
        InitDatasets();

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

        for (var i = 0; i < trainImagesPaths.Length; i++) {
            var isPositive = trainImagesPaths[i].Contains("bacteria") || trainImagesPaths[i].Contains("virus");

            Console.WriteLine($"Image {trainImagesPaths[i]} is {(isPositive ? "positive" : "negative")}.");

            TrainImages.Add((isPositive ? "positive" : "negative",
                LoadImage(trainImagesPaths[i], new Vector2(50, 50))));
        }

        for (var i = 0; i < valImagesPaths.Length; i++) {
            var isPositive = valImagesPaths[i].Contains("bacteria") || valImagesPaths[i].Contains("virus");

            Console.WriteLine($"Image {valImagesPaths[i]} is {(isPositive ? "positive" : "negative")}.");

            ValImages.Add((isPositive ? "positive" : "negative",
                LoadImage(valImagesPaths[i], new Vector2(50, 50))));
        }

        for (var i = 0; i < testImagesPaths.Length; i++) {
            var isPositive = testImagesPaths[i].Contains("bacteria") || testImagesPaths[i].Contains("virus");

            Console.WriteLine($"Image {testImagesPaths[i]} is {(isPositive ? "positive" : "negative")}.");

            TestImages.Add((isPositive ? "positive" : "negative",
                LoadImage(testImagesPaths[i], new Vector2(50, 50))));
        }

        Console.WriteLine($"Loaded {testImagesPaths.Length} images.");
    }

    private static void StartTraining() {
        Perceptron = new Perceptron(50 * 50, _learningRate);

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

            Perceptron.Train([..pixels, 1], label == "positive" ? 1 : 0);
        }
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

            LastGuess = Perceptron.Activate([..pixels, 1]);

            if (LastGuess == (label == "positive" ? 1 : 0)) correct++;

            total++;
        }

        Console.WriteLine($"The perceptron got {correct} out of {total} correct. ({correct / (double)total * 100}%)");
    }

    private static void Guess() {
        var random = new Random();
        var chosenImage = TestImages[random.Next(0, TestImages.Count)];

        LastImage = chosenImage;

        var pixels = new List<double>();

        for (var x = 0; x < chosenImage.image.Width; x++)
        for (var y = 0; y < chosenImage.image.Height; y++) {
            var pixel = chosenImage.image[x, y];
            var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0;
            pixels.Add(grayscale);
        }

        LastGuess = Perceptron.Activate([..pixels, 1]);

        Console.WriteLine($"The perceptron guessed {LastGuess} and the image is {chosenImage.label}.");
    }

    private static Image<Rgba32> LoadImage(string path, Vector2 size) {
        var image = Image.Load<Rgba32>(path);

        image.Mutate(x => x.Resize((int)size.X, (int)size.Y));

        return image;
    }

    private static void RenderImage(Image<Rgba32> image, Vector2 pos, int size) {
        for (var i = 0; i < image.Width * size; i++)
        for (var j = 0; j < image.Height * size; j++) {
            var pixel = image[i / size, j / size];
            Raylib.DrawPixel((int)pos.X + i, (int)pos.Y + j, new Color(pixel.R, pixel.G, pixel.B, pixel.A));
        }
    }

    private static void Render() {
        Raylib.BeginDrawing();
        Raylib.ClearBackground(Color.White);

        rlImGui.Begin();

        ImGui.SliderInt("Epochs", ref _epochs, 0, 100);
        ImGui.SliderFloat("Learning rate", ref _learningRate, 0.0001f, 0.1f);
        if (ImGui.Button("Start Training")) StartTraining();
        if (Perceptron != null && ImGui.Button("Guess")) Guess();
        if (Perceptron != null && ImGui.Button("Validate")) Validate();

        rlImGui.End();

        if (LastImage.image != null && LastImage.label != null) {
            RenderImage(LastImage.image, new Vector2(0, 0), 8);
            Raylib.DrawText(LastGuess == 1 ? "positive" : "negative", 0, 0, 20, Color.Blue);
            Raylib.DrawText(LastImage.label, 0, 20, 20, Color.Red);
        }

        Raylib.EndDrawing();
    }
}