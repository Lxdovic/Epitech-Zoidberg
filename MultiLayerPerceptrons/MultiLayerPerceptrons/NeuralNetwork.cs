namespace MultiLayerPerceptrons;

public class NeuralNetwork {
    public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
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
    }

    private double LearningRate { get; } = 0.1;
    private int InputNodes { get; }
    private int HiddenNodes { get; }
    private int OutputNodes { get; }
    private Matrix WeightsIh { get; set; }
    private Matrix WeightsHo { get; set; }
    private Matrix BiasH { get; set; }
    private Matrix BiasO { get; set; }

    public double[] FeedForward(double[] input) {
        var hidden = Matrix.Multiply(WeightsIh, Matrix.FromArray(input));

        hidden += BiasH;
        hidden.Map(Sigmoid);

        var output = Matrix.Multiply(WeightsHo, hidden);

        output += BiasO;
        output.Map(Sigmoid);

        return output.ToArray();
    }

    public void Train(double[] inputsArray, double[] answers) {
        var inputs = Matrix.FromArray(inputsArray);
        var hidden = Matrix.Multiply(WeightsIh, inputs);

        hidden += BiasH;
        hidden.Map(Sigmoid);

        var outputs = Matrix.Multiply(WeightsHo, hidden);

        outputs += BiasO;
        outputs.Map(Sigmoid);

        var targets = Matrix.FromArray(answers);
        var outputErrors = targets - outputs;
        var gradients = Matrix.Map(outputs, DSigmoid);

        gradients.Multiply(outputErrors);
        gradients.Multiply(LearningRate);

        var hiddenT = hidden.Transpose();
        var deltaWeightsHo = Matrix.Multiply(gradients, hiddenT);

        WeightsHo += deltaWeightsHo;
        BiasO += gradients;

        var hiddenErrors = Matrix.Multiply(WeightsHo.Transpose(), outputErrors);
        var hiddenGradients = Matrix.Map(hidden, DSigmoid);

        hiddenGradients.Multiply(hiddenErrors);
        hiddenGradients.Multiply(LearningRate);

        var inputsT = inputs.Transpose();
        var deltaWeightsIh = Matrix.Multiply(hiddenGradients, inputsT);

        WeightsIh += deltaWeightsIh;
        BiasH += hiddenGradients;
    }

    private double Sigmoid(double x) {
        return 1 / (1 + Math.Exp(-x));
    }

    private double DSigmoid(double y) {
        return y * (1 - y);
    }
}