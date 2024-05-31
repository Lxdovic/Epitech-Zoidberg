namespace MultiLayerPerceptrons;

public class NeuralNetwork {
    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate) {
        InputNodes = inputNodes;
        HiddenNodes = hiddenNodes;
        OutputNodes = outputNodes;

        WeightsIh = new Matrix(hiddenNodes, inputNodes);
        WeightsHo = new Matrix(outputNodes, hiddenNodes);
        BiasH = new Matrix(hiddenNodes, 1);
        BiasO = new Matrix(outputNodes, 1);

        WeightsIh.Randomize();
        WeightsHo.Randomize();
        BiasH.Randomize();
        BiasO.Randomize();

        LearningRate = learningRate;
    }

    public double LearningRate { get; set; }
    public int InputNodes { get; }
    public int HiddenNodes { get; }
    public int OutputNodes { get; }
    public Matrix WeightsIh { get; }
    public Matrix WeightsHo { get; }
    public Matrix BiasH { get; }
    public Matrix BiasO { get; }

    public double[] FeedForward(double[] inputsArray) {
        var inputs = Matrix.FromArray(inputsArray);
        var hidden = Matrix.Multiply(WeightsIh, inputs);

        hidden.Add(BiasH);
        hidden.Map(Sigmoid);

        var output = Matrix.Multiply(WeightsHo, hidden);

        output.Add(BiasO);
        output.Map(Sigmoid);

        return output.ToArray();
    }

    public void Train(double[] inputsArray, double[] answers) {
        var inputs = Matrix.FromArray(inputsArray);
        var hidden = Matrix.Multiply(WeightsIh, inputs);

        hidden.Add(BiasH);
        hidden.Map(Sigmoid);

        var outputs = Matrix.Multiply(WeightsHo, hidden);

        outputs.Add(BiasO);
        outputs.Map(Sigmoid);

        var targets = Matrix.FromArray(answers);
        var outputErrors = targets - outputs;
        var gradients = Matrix.Map(outputs, DSigmoid);

        gradients.Multiply(outputErrors);
        gradients.Multiply(LearningRate);

        var hiddenT = hidden.Transpose();
        var deltaWeightsHo = Matrix.Multiply(gradients, hiddenT);

        WeightsHo.Add(deltaWeightsHo);
        BiasO.Add(gradients);

        var hiddenErrors = Matrix.Multiply(WeightsHo.Transpose(), outputErrors);
        var hiddenGradients = Matrix.Map(hidden, DSigmoid);

        hiddenGradients.Multiply(hiddenErrors);
        hiddenGradients.Multiply(LearningRate);

        var inputsT = inputs.Transpose();
        var deltaWeightsIh = Matrix.Multiply(hiddenGradients, inputsT);

        WeightsIh.Add(deltaWeightsIh);
        BiasH.Add(hiddenGradients);
    }

    private static double Sigmoid(double x) {
        return 1.0f / (1.0f + (float)Math.Exp(-x));
    }

    private static double DSigmoid(double y) {
        return y * (1 - y);
    }
}