using System.ComponentModel;
using System.Numerics;
using ImGuiNET;
using Raylib_cs;
using rlImGui_cs;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Image = SixLabors.ImageSharp.Image;

namespace MultiLayerPerceptrons;

public struct Load {
    public int Curr;
    public int Max;
}

public static class Program {
    private static readonly (int width, int height) ScreenSize = (1200, 780);
    private static readonly int[] ImageSize = [50, 50];
    private static float _learningRate = 0.1f;
    private static float _learningRateMin = 0.001f;
    private static float _learningRateMax = 0.1f;
    private static float _decayRate = 0.05f;
    private static float _decayFactor = 0.5f;
    private static int _stepSize = 4;
    private static int _selectedScheduler;
    private static readonly int[] ImageLoadingWorkerCount = [1, 1, 2];
    private static int _epochs = 10;
    private static Load _imageLoad = new() { Curr = 0, Max = 0 };
    private static Load _trainLoad = new() { Curr = 0, Max = 0 };

    private static BackgroundWorker[] _imageloadingWorkers =
        new BackgroundWorker[ImageLoadingWorkerCount.Sum()];

    private static BackgroundWorker _trainingWorker = new();
    private static List<string> _trainImagesPaths = new();
    private static List<string> _valImagesPaths = new();
    private static List<string> _testImagesPaths = new();

    private static readonly string[] LearningRateSchedulers =
        { "None", "Step Decay", "Expo Decay", "Cosine Annealing" };

    private static readonly List<(string label, Image<Rgba32> image)> TrainImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> ValImages = new();
    private static readonly List<(string label, Image<Rgba32> image)> TestImages = new();
    private static NeuralNetwork.NeuralNetwork network = new(ImageSize.Length, 16, 2, _learningRate);

    public static void Main() {
        InitDatasets();

        Raylib.SetWindowState(ConfigFlags.ResizableWindow);
        Raylib.InitWindow(ScreenSize.width, ScreenSize.height, "Multi Layer Perceptrons");
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

    private static void LoadDatasets() {
        if (_trainingWorker.IsBusy) return;

        TrainImages.Clear();
        ValImages.Clear();
        TestImages.Clear();

        _imageLoad.Curr = 0;
        _imageLoad.Max = _trainImagesPaths.Count + _valImagesPaths.Count + _testImagesPaths.Count;

        InitImageLoadingWorkers();
        RunImageLoadingWorkers();
    }

    private static Image<Rgba32> LoadImage(string path, Vector2 size) {
        var image = Image.Load<Rgba32>(path);

        image.Mutate(x => x.Resize((int)size.X, (int)size.Y));

        return image;
    }

    private static void LoadTrainDataset(int workerIndex, int from, int to) {
        Console.WriteLine($"train workerIndex = {workerIndex} from = {from} to = {to}");

        for (var index = from; index < to; index++) {
            var path = _trainImagesPaths[index];
            var label = path.Contains("bacteria") || path.Contains("virus") ? "positive" : "negative";

            TrainImages.Add((label,
                LoadImage(path, new Vector2(ImageSize[0], ImageSize[1]))));

            _imageloadingWorkers[workerIndex].ReportProgress(0);
        }
    }

    private static void LoadTestDataset(int workerIndex, int from, int to) {
        Console.WriteLine($"test workerIndex = {workerIndex} from = {from} to = {to}");

        for (var index = from; index < to; index++) {
            var path = _testImagesPaths[index];
            var label = path.Contains("bacteria") || path.Contains("virus") ? "positive" : "negative";

            TestImages.Add((label,
                LoadImage(path, new Vector2(ImageSize[0], ImageSize[1]))));

            _imageloadingWorkers[workerIndex].ReportProgress(0);
        }
    }

    private static void LoadValidationDataset(int workerIndex, int from, int to) {
        Console.WriteLine($"val workerIndex = {workerIndex} from = {from} to = {to}");

        for (var index = from; index < to; index++) {
            var path = _valImagesPaths[index];
            var label = path.Contains("bacteria") || path.Contains("virus") ? "positive" : "negative";

            ValImages.Add((label,
                LoadImage(path, new Vector2(ImageSize[0], ImageSize[1]))));

            _imageloadingWorkers[workerIndex].ReportProgress(0);
        }
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

    private static void StartTraining() {
        Console.WriteLine("Training...");

        network = new NeuralNetwork.NeuralNetwork(ImageSize[0] * ImageSize[1], 8, 2, _learningRate);

        for (var i = 0; i < _epochs; i++) {
            network.LearningRate = (float)GetLearningRate(_selectedScheduler, i);

            for (var j = 0; j < TrainImages.Count; j++) {
                var (label, image) = TrainImages[j];
                var pixels = new List<double>();

                for (var x = 0; x < image.Width; x++)
                for (var y = 0; y < image.Height; y++) {
                    var pixel = image[x, y];
                    var grayscale = (pixel.R + pixel.G + pixel.B) / 3.0 / 255.0;
                    pixels.Add(grayscale);
                }
                
                double[] answers = label == "positive" ? [1, 0] : [0, 1];

                network.Train([..pixels], answers);
            }

            Validate();

            _trainingWorker.ReportProgress(0);
        }

        Console.WriteLine("Training finished.");
    }

    private static double GetLearningRate(int scheduler, int epoch) {
        return scheduler switch {
            0 => _learningRate,
            1 => _learningRate * Math.Pow(_decayFactor, Math.Floor((1 + epoch) / (double)_stepSize)),
            2 => _learningRate * Math.Exp(-_decayRate * epoch),
            3 => _learningRateMin + 0.5 * (_learningRateMax - _learningRateMin) *
                (1 + Math.Cos(epoch / (double)_epochs * Math.PI)),
            _ => throw new ArgumentOutOfRangeException()
        };
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

            var prediction = network.FeedForward([..pixels]);
            var guess = prediction.ToList().IndexOf(prediction.Max()) == 0 ? "positive" : "negative";

            if (guess == label) correct++;

            total++;
        }

        var acc = (float)Math.Round(correct / (double)total * 100, 3);

        Console.WriteLine($"Validation accuracy: {acc}%");
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

        if (_trainLoad.Curr == 0 || _trainLoad.Curr == _trainLoad.Max) {
            if (ImGui.Button("Start Training")) {
                if (_trainingWorker.IsBusy || _imageloadingWorkers.Any(w => w.IsBusy)) return;

                _trainingWorker = new BackgroundWorker();

                _trainLoad.Curr = 0;
                _trainLoad.Max = _epochs;

                _trainingWorker.WorkerReportsProgress = true;
                _trainingWorker.WorkerSupportsCancellation = true;
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

        ImGui.BeginChild("Images");

        Raylib.BeginDrawing();
        Raylib.EndDrawing();

        ImGui.EndChild();
        ImGui.PopStyleVar(3);
        ImGui.End();

        rlImGui.End();
    }
}