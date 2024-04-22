namespace RosenblattPerceptrons;

internal sealed class Perceptron {
    public Perceptron(int numberOfInputs, double learningRate = 0.0001) {
        NumberOfInputs = numberOfInputs;
        Learnc = learningRate;
        Weights = GenerateRandomWeights();
    }

    public int NumberOfInputs { get; }
    public double Learnc { get; }
    public List<double> Weights { get; }

    public double Activate(List<double> inputs) {
        double sum = 0;

        for (var i = 0; i < inputs.Count; i++) sum += inputs[i] * Weights[i];

        return sum > 0 ? 1 : 0;
    }

    public void Train(List<double> inputs, double desired) {
        var output = Activate(inputs);
        var error = desired - output;

        if (error != 0)
            for (var i = 0; i < inputs.Count; i++)
                Weights[i] += Learnc * error * inputs[i];
    }

    public List<double> GenerateRandomWeights() {
        var random = new Random();
        var weights = new List<double>();

        for (var i = 0; i <= NumberOfInputs; i++) {
            weights.Add(random.NextDouble() * 2 - 1);

            Console.WriteLine(weights[i]);
        }

        return weights;
    }
}