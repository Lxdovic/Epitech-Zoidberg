using System.Numerics;
using Zoidberg.ui;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Image = SixLabors.ImageSharp.Image;

namespace Zoidberg.utils;

public static class ImageLoader {
    private static List<string> _trainImagesPaths = new();
    private static List<string> _valImagesPaths = new();
    private static List<string> _testImagesPaths = new();
    private static Thread _trainThread = new(() => LoadTrainImages(0, _trainImagesPaths.Count));
    private static Thread _valThread = new(() => LoadValImages(0, _valImagesPaths.Count));
    private static Thread _testThread = new(() => LoadTestImages(0, _testImagesPaths.Count));
    public static readonly List<(string label, Image<Rgba32> image)> TrainImages = new();
    public static readonly List<(string label, Image<Rgba32> image)> ValImages = new();
    public static readonly List<(string label, Image<Rgba32> image)> TestImages = new();
    public static Vector2 ImageSize = new(64, 64);
    public static Load _imageLoad = new() { Curr = 0, Max = 0 };

    public static bool IsLoading => _trainThread.IsAlive || _valThread.IsAlive || _testThread.IsAlive;

    public static void InitDatasets() {
        CustomConsole.Log($"Initializing datasets...");
        
        var trainDatasetFolder = Path.Combine(Environment.CurrentDirectory, "resources/dataset/TRAIN");
        var valDatasetFolder = Path.Combine(Environment.CurrentDirectory, "resources/dataset/VALIDATION");
        var testDatasetFolder = Path.Combine(Environment.CurrentDirectory, "resources/dataset/TEST");

        if (!Directory.Exists(trainDatasetFolder)) CustomConsole.Log("Training dataset folder not found.");
        if (!Directory.Exists(valDatasetFolder)) CustomConsole.Log("Validation dataset folder not found.");
        if (!Directory.Exists(testDatasetFolder)) CustomConsole.Log("Test dataset folder not found.");

        _trainImagesPaths = Directory.GetFiles(trainDatasetFolder).ToList();
        _valImagesPaths = Directory.GetFiles(valDatasetFolder).ToList();
        _testImagesPaths = Directory.GetFiles(testDatasetFolder).ToList();
    }

    public static void ClearDatasets() {
        CustomConsole.Log($"Clearing Datasets.");
        
        TrainImages.Clear();
        ValImages.Clear();
        TestImages.Clear();
    }

    public static void LoadDatasets() {
        ClearDatasets();

        _imageLoad.Curr = 0;
        _imageLoad.Max = _trainImagesPaths.Count - 1 + _valImagesPaths.Count - 1 + _testImagesPaths.Count - 1;
        
        CustomConsole.Log($"Loading datasets, total images: {_imageLoad.Max}.");
        
        if (IsLoading) CustomConsole.Log($"WARNING: Loading threads were still busy.", LogType.Warning);
        
        _trainThread = new Thread(() => LoadTrainImages(0, _trainImagesPaths.Count));
        _valThread = new Thread(() => LoadValImages(0, _valImagesPaths.Count));
        _testThread = new Thread(() => LoadTestImages(0, _testImagesPaths.Count));

        _trainThread.Start();
        _valThread.Start();
        _testThread.Start();
    }

    public static void LoadTrainImages(int from, int to) {
        Parallel.For(from, to, index => {
            var path = _trainImagesPaths[index];
            var label = path.Contains("bacteria") ? "bacteria" : path.Contains("virus") ? "virus" : "negative";

            if (index % 100 == 0) CustomConsole.Log($"Loading train image {index} of {_trainImagesPaths.Count}.");

            TrainImages.Add((label,
                LoadImage(path, ImageSize)));

            _imageLoad.Curr++;
        });
        
        CustomConsole.Log($"Image loading complete for training dataset.");
    }

    public static void LoadValImages(int from, int to) {
        Parallel.For(from, to, index => {
            var path = _valImagesPaths[index];
            var label = path.Contains("bacteria") ? "bacteria" : path.Contains("virus") ? "virus" : "negative";
            
            if (index % 100 == 0) CustomConsole.Log($"Loading val image {index} of {_trainImagesPaths.Count}.");

            ValImages.Add((label,
                LoadImage(path, ImageSize)));

            _imageLoad.Curr++;
        });
        
        CustomConsole.Log($"Image loading complete for validation dataset.");
    }

    public static void LoadTestImages(int from, int to) {
        Parallel.For(from, to, index => {
            var path = _testImagesPaths[index];
            var label = path.Contains("bacteria") ? "bacteria" : path.Contains("virus") ? "virus" : "negative";
            
            if (index % 100 == 0) CustomConsole.Log($"Loading test image {index} of {_trainImagesPaths.Count}.");

            TestImages.Add((label,
                LoadImage(path, ImageSize)));

            _imageLoad.Curr++;
        });
        
        CustomConsole.Log($"Image loading complete for test dataset.");
    }

    public static Image<Rgba32> LoadImage(string path, Vector2 size) {
        var image = Image.Load<Rgba32>(path);

        image.Mutate(x => x.Resize((int)size.X, (int)size.Y));

        return image;
    }
}