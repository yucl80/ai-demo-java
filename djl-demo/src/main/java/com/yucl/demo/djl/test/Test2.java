package com.yucl.demo.djl.test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Base64;
import java.util.HashMap;
import java.util.Map;

public class Test2 {
    public static void main(String[] args) {
        Map<String, Integer> mergeableRanks = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new FileReader("D:\\llm\\glm4-tokenizer\\tokenizer.model"))) {
            String line;
            int i = 0;
            while ((line = reader.readLine()) != null && i < 40000) {
                i++;
                String[] parts = line.trim().split(" ");
                int rank = Integer.parseInt(parts[1]);
                byte[] token = Base64.getDecoder().decode(parts[0]);
                System.out.print(new String(token) + " ; ");
                mergeableRanks.put(new String(token), rank);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
