namespace RosenblattPerceptrons;

internal sealed class Perceptron {
    public Perceptron(List<bool> inputs, List<double> weights, double threshold) {
        if (inputs.Count != weights.Count) throw new ArgumentException("Inputs and weights must have the same length.");

        Inputs = inputs;
        Weights = weights;
        Threshold = threshold;
    }

    public List<bool> Inputs { get; } = new();
    public List<double> Weights { get; }
    public double Threshold { get; set; }

    public bool CalculateOutput() {
        double sum = 0;

        for (var i = 0; i < Inputs.Count; i++) sum += (Inputs[i] ? 1 : 0) * Weights[i];

        return sum >= Threshold;
    }
}