package model;

import java.util.ArrayList;
import java.util.Arrays;

public class Network {

    public final int layers;
    private final int[] input_shape;
    public int nodes;
    ArrayList<double[][]> weights = new ArrayList<double[][]>();
    ArrayList<double[][]> dweights = new ArrayList<double[][]>();
    ArrayList<double[][]> biases = new ArrayList<double[][]>();
    ArrayList<double[][]> dbiases = new ArrayList<double[][]>();
    ArrayList<double[][]> A = new ArrayList<double[][]>();
    private int[] output_shape;

    // TODO add additional constructors
    public Network(int layers, int nodes, int[] input_shape, int[] output_shape) {
        this.output_shape = output_shape;
        this.input_shape = input_shape;
        this.layers = layers;
        this.nodes = nodes;


        for (int i = 0; i < layers; i++) {

            System.out.println(output_shape[0]);
            System.out.println(output_shape[1]);
            if (i + 1 == layers) {
                double[][] w = np.random(output_shape[1], nodes);
                this.weights.add(i, w);
                double[][] b = new double[output_shape[1]][output_shape[0]];
                this.biases.add(i, b);
            }
            else {
                double[][] w = np.random(nodes, input_shape[1]);
                this.weights.add(i, w);
                double[][] b = new double[nodes][output_shape[0]];
                this.biases.add(i, b);
            }
        }
    }

    public double forwardProp(double[][] X, double[][] Y) {
        // iterate over total layers forward
        for (int i = 0; i < this.weights.size(); i++) {
            if (i == 0) {
                double[][] Z = np.add(np.dot(this.weights.get(i), X), this.biases.get(i));
                // add assumptions to list
                this.A.add(np.sigmoid(Z));
            }

            else {
                double[][] Z = np.add(np.dot(this.weights.get(i), this.A.get(i-1)), this.biases.get(i));
                // add assumptions to list
                this.A.add(np.sigmoid(Z));
            }
        }
        // returns cost of epoch
        return np.cross_entropy(this.A.size(),Y, this.A.get(A.size()-1));
    }

    public void backProp(double[][] X, double[][] Y) {

        this.dbiases = (ArrayList<double[][]>) this.biases.clone();
        this.dweights = (ArrayList<double[][]>) this.weights.clone();

        for (int i = this.layers - 1; i >= 0; i--) {

            double[][] dZ;
            double[][] dW;
            double[][] db;

            // last layer
            if (i == this.layers - 1) {
                System.out.printf("Last layer: %d\n", i);
                dZ = np.subtract(this.A.get(A.size() - 1), Y);
                dW = np.divide(np.dot(dZ, np.T(this.A.get(A.size() - 1))), this.output_shape[0]);
                db = np.divide(dZ, this.output_shape[0]);
                this.dweights.add(i, dW);
                this.dbiases.add(i, db);
                System.out.println(np.shape(this.weights.get(i)));
            }
            // first layer
            else if (i == 0) {
                System.out.printf("First layer: %d\n", i);
                System.out.println(np.shape(this.weights.get(i+1)));
                System.out.println(np.shape(np.dot(np.T(this.weights.get(i+1)),this.dweights.get(i+1))));
                System.out.println(np.shape(np.subtract(1.0, np.power(this.A.get(i), 2))));
                dZ = np.multiply(
                        np.dot(
                                np.T(this.weights.get(i+1)),
                                this.dweights.get(i+1)),
                        np.subtract(1.0, np.power(this.A.get(i), 2)));
                dW = np.divide(np.dot(dZ, np.T(X)), this.output_shape[0]);
                db = np.divide(dZ, this.output_shape[0]);
                this.dweights.add(i, dW);
                this.dbiases.add(i, db);
            }
            else {
                System.out.printf("Hidden layer: %d\n", i);
                // TODO add logic for hidden layers
                dZ = np.multiply(np.dot(np.T(this.weights.get(i+1)), this.dweights.get(i+1)), np.subtract(1.0, np.power(this.A.get(i), 2)));
                dW = np.divide(np.dot(dZ, np.T(this.A.get(i+1))), this.output_shape[0]);
                db = np.divide(dZ, this.output_shape[0]);
                this.dweights.add(i, dW);
                this.dbiases.add(i, db);
            }
        }
    }

    public void gradientDescent(double [][] dW, double[][] db, int i) {
        double [][] W = np.subtract(this.weights.get(i), np.multiply(0.01, dW));
        this.weights.set(i, W);

        double[][] b = np.subtract(this.biases.get(i), np.multiply(0.01, db));
        this.biases.set(i, b);
    }
}
