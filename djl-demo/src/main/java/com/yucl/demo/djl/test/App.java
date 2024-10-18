package com.yucl.demo.djl.test;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import ai.djl.Device;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.djl.ndarray.NDManager;

/**
 * Hello world!
 *
 */
public class App {
    public static void main(String[] args) throws OrtException, IOException {

        OrtEnvironment env = OrtEnvironment.getEnvironment();

        System.out.println(OrtEnvironment.getAvailableProviders());

        var sessionOptions = new OrtSession.SessionOptions();

        // sessionOptions.addDirectML(0);

        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get("D:/llm/bge-m3/tokenizer.json"));
        tokenizer.encode("test");
        var session = env.createSession("D:\\llm\\bge-m3\\model.onnx", sessionOptions);
        Map<String, NodeInfo> inputInfoList = session.getInputInfo();
        System.out.println(inputInfoList);
        Map<String, OnnxTensor> inputMap = new HashMap<>();
        OnnxTensor a = OnnxTensor.createTensor(env, new float[] { 2.0f });
        OnnxTensor b = OnnxTensor.createTensor(env, new float[] { 3.0f });

        String[] arrValues = new String[] { "this", "is", "a", "single", "dimensional", "string" };
        try (OnnxTensor t = OnnxTensor.createTensor(env, arrValues)) {

            String[] output = (String[]) t.getValue();

        }

        String[][] stringValues = new String[][] { { "this", "is", "a" }, { "multi", "dimensional", "string" } };
        try (OnnxTensor t = OnnxTensor.createTensor(env, stringValues)) {

            String[][] output = (String[][]) t.getValue();

        }

        String[][][] deepStringValues = new String[][][] {
                { { "this", "is", "a" }, { "multi", "dimensional", "string" } },
                { { "with", "lots", "more" }, { "dimensions", "than", "before" } }
        };
        try (OnnxTensor t = OnnxTensor.createTensor(env, deepStringValues)) {

            String[][][] output = (String[][][]) t.getValue();

        }

    }
    
}
