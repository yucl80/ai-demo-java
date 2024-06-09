package com.yucl.demo.djl.test;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public class NDArrayExample {
    public static void main(String[] args) {
        // Create an NDManager to manage NDArrays
        try (NDManager manager = NDManager.newBaseManager()) {
            // Create a 2D NDArray of shape [3, 3]
            float[][] data = {
                    { 1.0f, 2.0f, 3.0f },
                    { 4.0f, 5.0f, 6.0f },
                    { 7.0f, 8.0f, 9.0f }
            };
            NDArray array = manager.create(data);

            // Print the NDArray
            System.out.println("Original NDArray:");
            System.out.println(array);

            // Get the shape of the NDArray
            long[] shape = array.getShape().getShape();
            System.out.println("Shape: " + java.util.Arrays.toString(shape));

            // Access a specific element
            float element = array.getFloat(1, 2); // Element at row 1, column 2
            System.out.println("Element at (1,2): " + element);

            // Slice the NDArray (get a sub-array)
            NDArray subArray = array.get(":2", ":2"); // First 2 rows and first 2 columns
            System.out.println("Sliced NDArray (first 2 rows and columns):");
            System.out.println(subArray);

            // Perform operations on the NDArray
            NDArray result = array.add(10); // Add 10 to each element
            System.out.println("NDArray after adding 10 to each element:");
            System.out.println(result);
        }
    }
}
