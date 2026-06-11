namespace Day17ReflectionClassifier.Services;

/// <summary>Row-major matrix parameter with gradient and Adam moment buffers.</summary>
public sealed class Tensor
{
    public readonly int Rows;
    public readonly int Cols;
    public readonly double[] Data;
    public readonly double[] Grad;
    internal readonly double[] M; // Adam first moment
    internal readonly double[] V; // Adam second moment

    public Tensor(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        Data = new double[rows * cols];
        Grad = new double[rows * cols];
        M = new double[rows * cols];
        V = new double[rows * cols];
    }

    public double this[int r, int c]
    {
        get => Data[r * Cols + c];
        set => Data[r * Cols + c] = value;
    }
}
