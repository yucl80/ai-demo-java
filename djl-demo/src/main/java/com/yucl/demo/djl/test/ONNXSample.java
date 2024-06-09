package com.yucl.demo.djl.test;

import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class ONNXSample {

    public static NDArray meanPooling(NDArray tokenEmbeddings, NDArray attentionMask) {

        NDArray attentionMaskExpanded = attentionMask.expandDims(-1)
                .broadcast(tokenEmbeddings.getShape())
                .toType(DataType.FLOAT32, false);

        // Multiply token embeddings with expanded attention mask
        NDArray weightedEmbeddings = tokenEmbeddings.mul(attentionMaskExpanded);

        // Sum along the appropriate axis
        NDArray sumEmbeddings = weightedEmbeddings.sum(new int[] { 1 });

        // Clamp the attention mask sum to avoid division by zero
        NDArray sumMask = attentionMaskExpanded.sum(new int[] { 1 }).clip(1e-9f, Float.MAX_VALUE);

        // Divide sum embeddings by sum mask
        return sumEmbeddings.div(sumMask);
    }

    public static void main(String[] args) throws Exception {
        String TOKENIZER_URI = "file:/D:\\llm\\gpt2_pt/tokenizer.json";
        String MODEL_URI = "file:/D:\\llm\\gpt2_onnx\\gpt2.onnx";

        String[] sentences = new String[] { "Hello world" };

        // https://docs.djl.ai/extensions/tokenizers/index.html
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(TOKENIZER_URI), Map.of());
        Encoding[] encodings = tokenizer.batchEncode(sentences);

        long[][] input_ids0 = new long[encodings.length][];
        long[][] attention_mask0 = new long[encodings.length][];
        long[][] token_type_ids0 = new long[encodings.length][];

        for (int i = 0; i < encodings.length; i++) {
            input_ids0[i] = encodings[i].getIds();
            attention_mask0[i] = encodings[i].getAttentionMask();
            token_type_ids0[i] = encodings[i].getTypeIds();
        }

        // https://onnxruntime.ai/docs/get-started/with-java.html
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.setMemoryPatternOptimization(false);
        // sessionOptions.addDirectML(0);

        OrtSession session = environment.createSession(MODEL_URI, sessionOptions);

        OnnxTensor inputIds = OnnxTensor.createTensor(environment, input_ids0);
        OnnxTensor attentionMask = OnnxTensor.createTensor(environment, attention_mask0);
        OnnxTensor tokenTypeIds = OnnxTensor.createTensor(environment, token_type_ids0);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputIds);
        inputs.put("attention_mask", attentionMask);
        // inputs.put("token_type_ids", tokenTypeIds);

        try (OrtSession.Result results = session.run(inputs)) {

            OnnxValue lastHiddenState = results.get(0);

            float[][][] tokenEmbeddings = (float[][][]) lastHiddenState.getValue();

            System.out.println(tokenEmbeddings[0][0][0]);
            System.out.println(tokenEmbeddings[0][1][0]);
            System.out.println(tokenEmbeddings[0][2][0]);
            System.out.println(tokenEmbeddings[0][3][0]);

            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray ndTokenEmbeddings = create(tokenEmbeddings, manager);
                NDArray ndAttentionMask = manager.create(attention_mask0);
                System.out.println(ndTokenEmbeddings);

                var embedding = meanPooling(ndTokenEmbeddings, ndAttentionMask);
                System.out.println(embedding);
            }

        }
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

}