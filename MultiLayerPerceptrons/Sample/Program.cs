using MultiLayerPerceptrons;

namespace Sample;

public static class Program {
    public static void Main() {
        var neuralNetwork = new NeuralNetwork(2, 2, 1);

        // train the neural network to learn the XOR function

        var inputs = new[] {
            new double[] { 0, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 1, 1 }
        };

        var answers = new[] {
            new double[] { 0 },
            new double[] { 1 },
            new double[] { 1 },
            new double[] { 0 }
        };

        for (var i = 0; i < 10000; i++)
        for (var j = 0; j < inputs.Length; j++)
            neuralNetwork.Train(inputs[j], answers[j]);

        // test the neural network

        for (var i = 0; i < inputs.Length; i++) {
            var output = neuralNetwork.FeedForward(inputs[i]);
            Console.WriteLine($"Input: {inputs[i][0]} {inputs[i][1]}");
            Console.WriteLine($"Output: {output[0]}");
            Console.WriteLine();
        }
    }
}