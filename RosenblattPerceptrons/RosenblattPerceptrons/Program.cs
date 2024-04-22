namespace RosenblattPerceptrons;

internal class Program {
    private static void Main() {
        var inputsWithLabels = new List<(string label, bool value)> {
            ("Artists is Good", true),
            ("Weather is Good", true),
            ("Friend will Come", false),
            ("Food is Served", true),
            ("Alcohol is Served", true)
        };

        var inputs = new List<bool>();

        foreach (var (_, value) in inputsWithLabels) inputs.Add(value);

        var weights = new List<double> {
            0.6, 0.5, 0.6, 0.2, 0.0
        };

        var threshold = 1.2;

        var perceptron = new Perceptron(inputs, weights, threshold);

        Console.WriteLine(perceptron.CalculateOutput());
    }
}