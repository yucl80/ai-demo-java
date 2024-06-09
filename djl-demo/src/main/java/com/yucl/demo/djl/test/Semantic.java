package com.yucl.demo.djl.test;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtUtil;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.*;
import java.util.HashMap;
import java.util.Map;

public class Semantic {
    private HuggingFaceTokenizer tokenizer;
    private OrtSession session;
    private OrtEnvironment env;

    public Semantic(HuggingFaceTokenizer tokenizer, OrtSession session, OrtEnvironment env) {
        this.tokenizer = tokenizer;
        this.session = session;
        this.env = env;
    }

    public float[] embed(String sequence) throws Exception {
        var tokenized = tokenizer.encode(sequence);

        var inputIds = tokenized.getIds();
        var attentionMask = tokenized.getAttentionMask();
        var typeIds = tokenized.getTypeIds();

        var tensorInput = OrtUtil.reshape(inputIds, new long[] { 1, inputIds.length });
        var tensorAttentionMask = OrtUtil.reshape(attentionMask, new long[] { 1, attentionMask.length });
        var tensorTypeIds = OrtUtil.reshape(typeIds, new long[] { 1, typeIds.length });

        Map<String, OnnxTensor> inputMap = new HashMap<>();
        inputMap.put("input_ids", OnnxTensor.createTensor(env, tensorInput));
        inputMap.put("attention_mask", OnnxTensor.createTensor(env, tensorAttentionMask));
        inputMap.put("token_type_ids", OnnxTensor.createTensor(env, tensorTypeIds));

        var result = session.run(inputMap);
        var outputTensor = (OnnxTensor) result.get(0);
        var outputBuffer = outputTensor.getFloatBuffer();
        float[] output = new float[outputBuffer.remaining()];
        outputBuffer.get(output);

        return output;
    }

    public static Semantic create() throws OrtException, IOException {
        var classLoader = Thread.currentThread().getContextClassLoader();

        try (InputStream tokenizerStream = classLoader.getResourceAsStream("model/tokenizer.json");
                InputStream onnxStream = classLoader.getResourceAsStream("model/model.onnx")) {

            if (tokenizerStream == null || onnxStream == null) {
                throw new IOException("Resource not found");
            }

            var tokenizer = HuggingFaceTokenizer.newInstance(tokenizerStream, null);
            var ortEnv = OrtEnvironment.getEnvironment();
            var sessionOptions = new OrtSession.SessionOptions();

            byte[] onnxPathAsByteArray = onnxStream.readAllBytes();
            var session = ortEnv.createSession(onnxPathAsByteArray, sessionOptions);

            return new Semantic(tokenizer, session, ortEnv);
        }
    }
}
