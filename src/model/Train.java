package model;

import java.util.Arrays;

import static stdio_lib.StdOut.printf;

public class Train {

    public static void main(String[] args) {
        double[][] X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double[][] Y = {{0}, {1}, {1}, {0}};

        int[] input_shape = {X.length, X[0].length};
        int[] output_shape = {Y.length, Y[0].length};

        printf("input (%d,%d)\n", input_shape[0], input_shape[1]);
        printf("output (%d,%d)\n", output_shape[0], output_shape[1]);

        int nodes = 400;

        X = np.T(X);
        Y = np.T(Y);

        Network model = new Network(2, 512, input_shape, output_shape);


        for (int i = 0; i < 4000; i++) {

            double cost = model.forwardProp(X, Y);
            model.backProp(X, Y);

            if (i % 400 == 0) {
                printf("__________________\n");
                printf("Cost: %s\n", cost);
                printf("Predictions = %s", Arrays.deepToString(model.A.get(model.layers-1)));
            }

        }
    }
}
