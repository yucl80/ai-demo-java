package com.yucl.demo.djl;

import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class OnnxEmbeddingDemo {

    public static void main(String[] args) throws Exception {
        String TOKENIZER_URI = "D:\\llm\\bge-m3-onnx\\tokenizer.json";
        String MODEL_URI = "D:\\llm\\bge-m3-onnx\\model.onnx";
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(TOKENIZER_URI), Map.of());
        try (OrtEnvironment environment = OrtEnvironment.getEnvironment();
                OrtSession session = environment.createSession(MODEL_URI);) {
            String[] sentences = new String[] { "I like you", "我喜欢你", "我讨厌你" };
            emb(environment, session, tokenizer, sentences);
            long l = System.currentTimeMillis();
            for (int i = 0; i < 100; i++) {
                float[][] embeddings = emb(environment, session, tokenizer, sentences);
            }
            System.out.println("used time:" + (System.currentTimeMillis() - l));
            float[][] embeddings = emb(environment, session, tokenizer, sentences);
            double similaryity = cosineSimilarity(embeddings[0], embeddings[1]);
            System.out.println(similaryity);
            double similaryity2 = cosineSimilarity(embeddings[0], embeddings[2]);
            System.out.println(similaryity2);
            double similaryity3 = cosineSimilarity(embeddings[1], embeddings[2]);
            System.out.println(similaryity3);
        }

    }

    public static float[][] emb(OrtEnvironment environment, OrtSession session, HuggingFaceTokenizer tokenizer,
            String[] sentences) throws OrtException {
        Encoding[] encodings = tokenizer.batchEncode(sentences);
        long[][] input_ids0 = new long[encodings.length][];
        long[][] attention_mask0 = new long[encodings.length][];
        float[][] embeddings = new float[0][0];
        for (int i = 0; i < encodings.length; i++) {
            input_ids0[i] = encodings[i].getIds();
            attention_mask0[i] = encodings[i].getAttentionMask();
        }
        try (OnnxTensor inputIds = OnnxTensor.createTensor(environment, input_ids0);
                OnnxTensor attentionMask = OnnxTensor.createTensor(environment, attention_mask0);) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIds);
            inputs.put("attention_mask", attentionMask);
            try (OrtSession.Result results = session.run(inputs)) {
                embeddings = (float[][]) results.get("sentence_embedding").get().getValue();
            }
            inputs.clear();

        }
        return embeddings;

    }

    public static double cosineSimilarity(float[] vectorA, float[] vectorB) {
        float dotProduct = 0.0f;
        float normA = 0.0f;
        float normB = 0.0f;
        for (int i = 0; i < vectorA.length; i++) {
            float v1 = vectorA[i];
            float v2 = vectorB[i];
            dotProduct += v1 * v2;
            normA += Math.pow(v1, 2);
            normB += Math.pow(v2, 2);
        }
        if (normA == 0 && normB == 0) {
            return 0.0f;
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

}