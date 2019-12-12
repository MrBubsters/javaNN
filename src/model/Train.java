package model;

import java.util.Arrays;

import static stdio_lib.StdOut.printf;

public class Train {

    public static void main(String[] args) {
        // training data set
        double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] Y = {{0}, {1}, {1}, {0}};

        int[] input_shape = {X.length, X[0].length};
        int[] output_shape = {Y.length, Y[0].length};

        printf("input (%d,%d)\n", input_shape[0], input_shape[1]);
        printf("output (%d,%d)\n", output_shape[0], output_shape[1]);

        // transform data set for model
        X = np.T(X);
        Y = np.T(Y);

        // initialize model from Network()
        Network model = new Network(2, 512, input_shape, output_shape);


        // loop for 10,000 epochs of training
        for (int i = 0; i < 10000; i++) {

            double cost = model.forwardProp(X, Y, i);
            model.backProp(X, Y);
            model.gradientDescent(0.01);

            if (i % 100 == 0) {
                printf("__________________\n");
                printf("Cost: %s\n", cost);
                printf("Predictions = %s", Arrays.deepToString(model.A.get(model.layers-1)));
            }

        }
    }
}
