package com.yucl.demo.djl.test;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;

public class ONNXSample7 {

    public static void main(String[] args) throws Exception {
        String TOKENIZER_URI = "D:\\llm\\qwen2-0.5B-onnx\\tokenizer.json";
        String MODEL_URI = "D:\\llm\\qwen2-0.5B-onnx\\model.onnx";
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL);
        sessionOptions.setSessionLogVerbosityLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL.getValue());
        sessionOptions.setMemoryPatternOptimization(true);
        OrtSession session = env.createSession(MODEL_URI, sessionOptions);
        // Set<String> onnxModelOutputs = session.getOutputNames();
        // Stream<String> strs = onnxModelOutputs.stream();
        // System.out.println("Model output names: " + (String)
        // strs.collect(Collectors.joining(", ")));
        // System.out.println(session.getNumOutputs());
        // System.out.println(session.getOutputInfo());

        // System.out.println(session.getInputInfo());

        String sentences = "How to learn ai program ?";
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(TOKENIZER_URI), Map.of());

        Encoding encodings = tokenizer.encode(sentences);
        long[] input_ids = encodings.getIds();
        List<Long> generatedIds = new ArrayList<>();
        for (long id : input_ids) {
            generatedIds.add(id);
        }
        int totalLength = 200;
        while (generatedIds.size() < totalLength) {
            long[] currentInputIds = new long[generatedIds.size()];
            long[] currentPositionIds = new long[generatedIds.size()];
            long[] attentionMask = new long[generatedIds.size()];
            for (int i = 0; i < generatedIds.size(); i++) {
                currentInputIds[i] = generatedIds.get(i);
                currentPositionIds[i] = i;
                attentionMask[i] = 1;
            }

            // 创建 ONNX 输入
            try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(currentInputIds),
                    new long[] { 1, currentInputIds.length });
                    OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask),
                            new long[] { 1, attentionMask.length });
                    OnnxTensor positionTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(currentPositionIds),
                            new long[] { 1, currentPositionIds.length });) {

                Map<String, OnnxTensor> inputs = new HashMap<>();
                inputs.put("input_ids", inputTensor);
                inputs.put("attention_mask", attentionMaskTensor);
                inputs.put("position_ids", positionTensor);
                long[] shape = { 1, 2, 1, 64 };
                for (int i = 3; i < 22; i += 2) {
                    int order = (i - 3) / 2;
                    FloatBuffer kbuffer = FloatBuffer.allocate(1 * 2 * 1 * 64);
                    for (int j = 0; j < kbuffer.capacity(); j++) {
                        kbuffer.put(0f);
                    }
                    kbuffer.flip();
                    OnnxTensor ktensor = OnnxTensor.createTensor(env, kbuffer, shape);
                    inputs.put(String.format("past_key_values.%s.key", order), ktensor);

                    FloatBuffer vbuffer = FloatBuffer.allocate(1 * 2 * 1 * 64);
                    for (int j = 0; j < vbuffer.capacity(); j++) {
                        vbuffer.put(0f);
                    }
                    vbuffer.flip();
                    OnnxTensor vtensor = OnnxTensor.createTensor(env, vbuffer, shape);
                    inputs.put(String.format("past_key_values.%s.key", order), vtensor);
                }

                try (OrtSession.Result results = session.run(inputs)) {
                    OnnxValue lastHiddenState = results.get(0);
                    float[][][] logits = (float[][][]) lastHiddenState.getValue();
                    // 选择最大概率对应的 token ID (贪心搜索)
                    // long[] outputIds = new long[logits[0].length];
                    long nextTokenId = argmax(logits[0][logits[0].length - 1]);
                    generatedIds.add(nextTokenId);
                }
                inputs.clear();
            }
            // inputTensor.close();
            // attentionMaskTensor.close();
            // positionTensor.close();
        }
        session.close();
        env.close();

        long[] gen_tokens = new long[generatedIds.size()];
        for (int i = 0; i < generatedIds.size(); i++) {
            gen_tokens[i] = generatedIds.get(i);
        }

        String outputText = tokenizer.decode(gen_tokens);
        System.out.println("Generated text: " + outputText);

    }

    public static NDArray create(float[][][] data, NDManager manager) {
        FloatBuffer buffer = FloatBuffer.allocate(data.length * data[0].length * data[0][0].length);
        for (float[][] data2 : data) {
            for (float[] d : data2) {
                buffer.put(d);
            }
        }
        buffer.rewind();
        return manager.create(buffer, new Shape(data.length, data[0].length, data[0][0].length));
    }

    // 选择最大值的索引
    private static int argmax(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

}