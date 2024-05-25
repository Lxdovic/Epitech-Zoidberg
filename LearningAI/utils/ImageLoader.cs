using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Image = SixLabors.ImageSharp.Image;

namespace LearningAI.utils;

public static class ImageLoader {
    private static List<string> _trainImagesPaths = new();
    private static List<string> _valImagesPaths = new();
    private static List<string> _testImagesPaths = new();
    public static readonly List<(string label, Image<Rgba32> image)> TrainImages = new();
    public static readonly List<(string label, Image<Rgba32> image)> ValImages = new();
    public static readonly List<(string label, Image<Rgba32> image)> TestImages = new();
    public static Vector2 ImageSize = new(64, 64);
    public static Load _imageLoad = new() { Curr = 0, Max = 0 };

    public static void InitDatasets() {
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

    public static void ClearDatasets() {
        TrainImages.Clear();
        ValImages.Clear();
        TestImages.Clear();
    }

    public static void LoadDatasets() {
        ClearDatasets();

        _imageLoad.Curr = 0;
        _imageLoad.Max = _trainImagesPaths.Count + _valImagesPaths.Count + _testImagesPaths.Count - 1;

        new Thread(() => LoadTrainImages(0, _trainImagesPaths.Count)).Start();
        new Thread(() => LoadValImages(0, _valImagesPaths.Count)).Start();
        new Thread(() => LoadTestImages(0, _testImagesPaths.Count)).Start();
    }

    public static void LoadTrainImages(int from, int to) {
        Parallel.For(from, to, index => {
            var path = _trainImagesPaths[index];
            var label = path.Contains("bacteria") ? "bacteria" : path.Contains("virus") ? "virus" : "negative";

            TrainImages.Add((label,
                LoadImage(path, ImageSize)));

            _imageLoad.Curr++;
        });
    }

    public static void LoadValImages(int from, int to) {
        Parallel.For(from, to, index => {
            var path = _valImagesPaths[index];
            var label = path.Contains("bacteria") ? "bacteria" : path.Contains("virus") ? "virus" : "negative";

            ValImages.Add((label,
                LoadImage(path, ImageSize)));

            _imageLoad.Curr++;
        });
    }

    public static void LoadTestImages(int from, int to) {
        Parallel.For(from, to, index => {
            var path = _testImagesPaths[index];
            var label = path.Contains("bacteria") ? "bacteria" : path.Contains("virus") ? "virus" : "negative";

            TestImages.Add((label,
                LoadImage(path, ImageSize)));

            _imageLoad.Curr++;
        });
    }

    public static Image<Rgba32> LoadImage(string path, Vector2 size) {
        var image = Image.Load<Rgba32>(path);

        image.Mutate(x => x.Resize((int)size.X, (int)size.Y));

        return image;
    }
}