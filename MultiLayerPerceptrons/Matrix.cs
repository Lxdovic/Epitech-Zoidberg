using System.Text;

namespace MultiLayerPerceptrons;

public class Matrix(int rows, int columns) {
    private double[,] Data { get; } = new double[rows, columns];
    public int Rows { get; } = rows;
    public int Columns { get; } = columns;

    public double this[int row, int column] {
        get => Data[row, column];
        set => Data[row, column] = value;
    }

    public void Randomize() {
        var rand = new Random();

        Parallel.For(0, Rows, i => {
            for (var j = 0; j < Columns; j++)
                Data[i, j] = rand.NextDouble() * 2 - 1;
        });
    }

    public void Multiply(double n) {
        Parallel.For(0, Rows, i => {
            for (var j = 0; j < Columns; j++)
                Data[i, j] *= n;
        });
    }

    public void Multiply(Matrix matrix) {
        Parallel.For(0, Rows, i => {
            for (var j = 0; j < Columns; j++)
                Data[i, j] *= matrix[i, j];
        });
    }

    public static Matrix Multiply(Matrix matrix1, Matrix matrix2) {
        if (matrix1.Columns != matrix2.Rows)
            throw new ArgumentException(
                "The number of columns in the first matrix must be equal to the number of rows in the second matrix for multiplication.");

        var result = new Matrix(matrix1.Rows, matrix2.Columns);

        Parallel.For(0, result.Rows, i => {
            for (var j = 0; j < result.Columns; j++)
            for (var k = 0; k < matrix1.Columns; k++)
                result.Data[i, j] += matrix1[i, k] * matrix2[k, j];
        });

        return result;
    }

    public Matrix Transpose() {
        var result = new Matrix(Columns, Rows);

        Parallel.For(0, Rows, i => {
            for (var j = 0; j < Columns; j++)
                result[j, i] = Data[i, j];
        });

        return result;
    }

    public void Map(Func<double, double> func) {
        Parallel.For(0, Rows, i => {
            for (var j = 0; j < Columns; j++) {
                var value = Data[i, j];
                Data[i, j] = func.Invoke(value);
            }
        });
    }

    public static Matrix Map(Matrix matrix, Func<double, double> func) {
        var result = new Matrix(matrix.Rows, matrix.Columns);

        Parallel.For(0, matrix.Rows, i => {
            for (var j = 0; j < matrix.Columns; j++)
                result[i, j] = func(matrix[i, j]);
        });

        return result;
    }

    public static Matrix operator +(Matrix matrix1, Matrix matrix2) {
        if (matrix1.Rows != matrix2.Rows || matrix1.Columns != matrix2.Columns)
            throw new ArgumentException(
                "The number of rows and columns in both matrices must be equal for addition.");

        var result = new Matrix(matrix1.Rows, matrix1.Columns);

        for (var i = 0; i < result.Rows; i++)
        for (var j = 0; j < result.Columns; j++)
            result[i, j] = matrix1[i, j] + matrix2[i, j];

        return result;
    }

    public static Matrix operator -(Matrix matrix1, Matrix matrix2) {
        if (matrix1.Rows != matrix2.Rows || matrix1.Columns != matrix2.Columns)
            throw new ArgumentException(
                "The number of rows and columns in both matrices must be equal for subtraction.");

        var result = new Matrix(matrix1.Rows, matrix1.Columns);

        for (var i = 0; i < result.Rows; i++)
        for (var j = 0; j < result.Columns; j++)
            result[i, j] = matrix1[i, j] - matrix2[i, j];

        return result;
    }

    public override string ToString() {
        var sb = new StringBuilder();

        for (var i = 0; i < Rows; i++) {
            for (var j = 0; j < Columns; j++) {
                sb.Append(Data[i, j]);
                sb.Append(' ');
            }

            sb.AppendLine();
        }

        return sb.ToString();
    }

    public static Matrix FromArray(double[] input) {
        var result = new Matrix(input.Length, 1);

        for (var i = 0; i < input.Length; i++)
            result[i, 0] = input[i];

        return result;
    }

    public double[] ToArray() {
        var result = new double[Rows * Columns];

        for (var i = 0; i < Rows; i++)
        for (var j = 0; j < Columns; j++)
            result[i * Columns + j] = Data[i, j];

        return result;
    }

    public void Add(Matrix matrix) {
        for (var i = 0; i < Rows; i++)
        for (var j = 0; j < Columns; j++)
            Data[i, j] += matrix[i, j];
    }
}