package model;

import java.util.ArrayList;
import java.util.Arrays;

public class Network {

    public final int layers;
    private final int[] input_shape;
    public int nodes;
    ArrayList<double[][]> weights = new ArrayList<double[][]>();
    ArrayList<double[][]> dweights = new ArrayList<double[][]>();
    ArrayList<double[][]> dzeights = new ArrayList<double[][]>();
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

            if (i + 1 == layers) {
                double[][] w = np.random(output_shape[1], nodes);
                this.weights.add(i, w);
                double[][] b = new double[output_shape[1]][output_shape[0]];
                this.biases.add(i, b);
            }
            else {
                double[][] w = np.random(nodes, input_shape[1]);
                this.weights.add(i, w);
                double[][] b = new double[nodes][input_shape[0]];
                this.biases.add(i, b);
            }
        }


        this.dbiases = (ArrayList<double[][]>) this.biases.clone();
        this.dweights = (ArrayList<double[][]>) this.weights.clone();
        this.dzeights = (ArrayList<double[][]>) this.weights.clone();
    }

    public double forwardProp(double[][] X, double[][] Y, int epoch) {
        // iterate over total layers forward
        for (int i = 0; i < this.weights.size(); i++) {
            // first layer
            if (i == 0) {
                double[][] Z = np.add(np.dot(this.weights.get(i), X), this.biases.get(i));
                // add assumptions to list
                if (epoch == 0) {
                    this.A.add(i, np.sigmoid(Z));
                }
                else {
                    this.A.set(i, np.sigmoid(Z));
                }
            }
            // hidden or last layers
            else {
                double[][] Z = np.add(np.dot(this.weights.get(i), this.A.get(i-1)), this.biases.get(i));
                // add assumptions to list
                if (epoch == 0) {
                    this.A.add(i, np.sigmoid(Z));
                }
                else {
                    this.A.set(i, np.sigmoid(Z));
                }
            }
        }
        // returns cost of epoch
        return np.cross_entropy(output_shape[0],Y, this.A.get(A.size()-1));
    }

    public void backProp(double[][] X, double[][] Y) {

        for (int i = this.layers - 1; i >= 0; i--) {

            double[][] dZ;
            double[][] dW;
            double[][] db;

            // last layer
            if (i == this.layers - 1) {
//                System.out.printf("Last layer: %d\n", i);
//                System.out.println(np.shape(dZ));
//                System.out.println(np.shape(np.T(this.A.get(i- 1))));
//                System.out.println(this.output_shape[0]);
//                System.out.println(np.shape(dW));

                /*
                (1,4)
                (4,512)
                4
                (1,512)
                 */
                dZ = np.subtract(this.A.get(i), Y);
                // check output dimensions
                dW = np.divide(np.dot(dZ, np.T(this.A.get(i- 1))), this.output_shape[0]);
                db = np.divide(dZ, this.output_shape[0]);
                this.dweights.set(i, dW);
                this.dzeights.set(i, dZ);
                this.dbiases.set(i, db);
            }
            // first layer
            else if (i == 0) {
//                System.out.printf("First layer: %d\n", i);
//                System.out.println(np.shape(np.dot(np.T(this.weights.get(i+1)),this.dzeights.get(i+1)))); // incorrect shape
//                System.out.println(np.shape(np.T(this.weights.get(i+1))));
//                System.out.println(np.shape(this.dzeights.get(i+1)));
//                System.out.println(np.shape(np.subtract(1.0, np.power(this.A.get(i), 2))));
//                System.out.println(np.shape(this.A.get(i)));

                /* supposed to output:
                (512,4)
                (512,1)
                (1,4)
                (512,4)
                (512,4)
                 */
                dZ = np.multiply(
                        np.dot(
                                np.T(this.weights.get(i+1)),
                                this.dzeights.get(i+1)),
                        np.subtract(1.0, np.power(this.A.get(i), 2)));
                dW = np.divide(np.dot(dZ, np.T(X)), this.output_shape[0]);
                db = np.divide(dZ, this.output_shape[0]);
                this.dweights.set(i, dW);
                this.dbiases.set(i, db);
                this.dzeights.set(i, dZ);
            }
            else {
//                System.out.printf("Hidden layer: %d\n", i);
                // TODO add logic for hidden layers
                dZ = np.multiply(np.dot(np.T(this.weights.get(i+1)), this.dweights.get(i+1)), np.subtract(1.0, np.power(this.A.get(i), 2)));
                dW = np.divide(np.dot(dZ, np.T(this.A.get(i+1))), this.output_shape[0]);
                db = np.divide(dZ, this.output_shape[0]);
                this.dweights.set(i, dW);
                this.dbiases.set(i, db);
                this.dzeights.set(i, dZ);
            }
        }
    }

    public void gradientDescent() {
        for (int i = 0; i < this.layers - 1; i++) {
            double [][] W = np.subtract(this.weights.get(i), np.multiply(0.01, this.dweights.get(i)));
            this.weights.set(i, W);

            double[][] b = np.subtract(this.biases.get(i), np.multiply(0.01, this.dbiases.get(i)));
            this.biases.set(i, b);
        }
    }
}
