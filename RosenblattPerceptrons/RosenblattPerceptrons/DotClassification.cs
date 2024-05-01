using ImGuiNET;
using Raylib_cs;
using rlImGui_cs;

namespace RosenblattPerceptrons;

internal class DotClassification {
    private static readonly List<(double x, double y)> Points = new();
    private static readonly List<double> CorrectAnswers = new();
    private static readonly Random Random = new();
    private static readonly (int width, int height) ScreenSize = (1200, 780);
    private static float _learningRate = 0.0001f;
    private static int _bias = 80;
    private static int _oldBias = 80;
    private static float _weight = 1.2f;
    private static float _oldWeight = 1.2f;
    private static int _numberOfPoints = 1000;
    private static int _oldNumberOfPoints = 1000;
    private static int _epochs = 10;
    private static bool _showGuesses;
    private static Perceptron _perceptron = new(2, _learningRate);


    public static void Run() {
        GeneratePoints();


        Raylib.InitWindow(ScreenSize.width, ScreenSize.height, "Rosenblatt Perceptrons");
        rlImGui.Setup(true, true);

        while (!Raylib.WindowShouldClose()) Render();

        rlImGui.Shutdown();
        Raylib.CloseWindow();
    }

    private static void StartTraining() {
        _perceptron = new Perceptron(2, _learningRate);

        for (var i = 0; i < _epochs; i++)
        for (var j = 0; j < Points.Count; j++) {
            var (x, y) = Points[j];

            _perceptron.Train([x, y, _bias], CorrectAnswers[j]);
        }
    }

    private static void GeneratePoints() {
        Points.Clear();

        for (var i = 0; i < _numberOfPoints; i++) {
            var x = Random.NextDouble() * ScreenSize.width;
            var y = Random.NextDouble() * ScreenSize.height;

            Points.Add((x, y));
        }

        ComputeCorrectAnswers();
    }

    private static void ToggleShowGuesses() {
        _showGuesses = !_showGuesses;
    }

    private static void Render() {
        Raylib.BeginDrawing();
        Raylib.ClearBackground(Color.White);

        rlImGui.Begin();

        ImGui.SliderInt("Number of points", ref _numberOfPoints, 0, 10000);
        ImGui.SliderInt("Epochs", ref _epochs, 0, 10000);
        ImGui.SliderFloat("Learning rate", ref _learningRate, 0.0001f, 0.1f);
        ImGui.SliderInt("Bias", ref _bias, 0, ScreenSize.height);
        ImGui.SliderFloat("Weight", ref _weight, -10, 10);

        if (ImGui.Button("Start Training")) StartTraining();
        if (ImGui.Button(_showGuesses ? "Show Answers" : "Show Guesses")) ToggleShowGuesses();

        if (_oldNumberOfPoints != _numberOfPoints) {
            GeneratePoints();

            _oldNumberOfPoints = _numberOfPoints;
        }

        if (_oldBias != _bias) {
            ComputeCorrectAnswers();

            _oldBias = _bias;
        }

        if (_oldWeight != _weight) {
            ComputeCorrectAnswers();

            _oldWeight = _weight;
        }

        for (var i = 0; i < Points.Count; i++) {
            var (x, y) = Points[i];
            var guess = _showGuesses ? _perceptron.Activate([x, y, _bias]) : CorrectAnswers[i];
            var color = guess >= 1 ? Color.Red : Color.Green;

            Raylib.DrawCircle((int)x, (int)y, 2, color);
        }

        Raylib.DrawLine(0, (int)LineFunction(0), ScreenSize.width, (int)LineFunction(ScreenSize.width), Color.Blue);

        rlImGui.End();
        Raylib.EndDrawing();
    }

    private static double LineFunction(double x) {
        return x * _weight + _bias;
    }

    private static void ComputeCorrectAnswers() {
        CorrectAnswers.Clear();

        foreach (var (x, y) in Points) {
            var lineY = LineFunction(x);

            CorrectAnswers.Add(y > lineY ? 1 : 0);
        }
    }
}