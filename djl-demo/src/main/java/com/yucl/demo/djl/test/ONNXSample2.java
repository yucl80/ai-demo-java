package com.yucl.demo.djl.test;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtUtil;

public class ONNXSample2 {

    public static void main(String[] args) throws Exception {
        String TOKENIZER_URI = "file:/D:\\llm\\llama_quantize/tokenizer.json";
        String MODEL_URI = "file:/D:\\llm\\llama_quantize\\model_quantized.onnx";

        String[] sentences = new String[] { "Where is ShenZhen ?" };

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

        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.setMemoryPatternOptimization(true);
        OrtSession session = environment.createSession(MODEL_URI, sessionOptions);

        Set<String> onnxModelOutputs = session.getOutputNames();
        Stream<String> strs = onnxModelOutputs.stream();
        System.out.println("Model output names: " + (String) strs.collect(Collectors.joining(", ")));

        OnnxTensor inputIds = OnnxTensor.createTensor(environment, input_ids0);
        OnnxTensor attentionMask = OnnxTensor.createTensor(environment, attention_mask0);
        OnnxTensor tokenTypeIds = OnnxTensor.createTensor(environment, token_type_ids0);
        // OnnxTensor inputTensor = OnnxTensor.createTensor(env,
        // LongBuffer.wrap(inputIds), new long[]{1, inputIds.length});
        // OnnxTensor positionTensor = OnnxTensor.createTensor(env,
        // LongBuffer.wrap(positionIds), new long[]{1, positionIds.length});

        OnnxTensor position_ids = createPositionIds(environment, attentionMask);

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputIds);
        inputs.put("attention_mask", attentionMask);
        inputs.put("position_ids", position_ids);

        try (OrtSession.Result results = session.run(inputs)) {
            System.out.println(session.getNumOutputs());
            System.out.println(session.getOutputInfo());

            OnnxValue lastHiddenState = results.get(0);

            System.err.println(lastHiddenState);

            // Object tensorIn = OrtUtil.reshape(flatInput, shape)

            float[][][] predictions = (float[][][]) lastHiddenState.getValue();

            // String[] ss = tokenizer.batchDecode(data[0]);
            int[] predictedTokenIndices = getTopKIndices(predictions[0][0], 5); // Helper function to get
            // top K indices

            // Get the top predicted tokens
            List<String> topPredictedTokens = new ArrayList<>();
            for (int i = 0; i < Math.min(predictedTokenIndices.length, 5); i++) {
                long predictedTokenId = predictedTokenIndices[i];
                String predictedToken = tokenizer.decode(new long[] { predictedTokenId });

                // Clean the predicted token
                StringBuilder cleanTokenBuilder = new StringBuilder(predictedToken.length());
                for (char c : predictedToken.toCharArray()) {
                    if (Character.isLetterOrDigit(c) || Character.isWhitespace(c)) {
                        cleanTokenBuilder.append(c);
                    }
                }
                String cleanedToken = cleanTokenBuilder.toString().trim();

                topPredictedTokens.add(cleanedToken);
            }

            System.out.println(topPredictedTokens);

        }
    }

    private static OnnxTensor createPositionIds(OrtEnvironment environment, OnnxTensor attentionMask)
            throws OrtException {
        long[][] maskData = (long[][]) attentionMask.getValue();
        long[][] positionIds = new long[maskData.length][maskData[0].length];
        for (int i = 0; i < maskData.length; i++) {
            for (int j = 0; j < maskData[i].length; j++) {
                positionIds[i][j] = j; // Simple cumulative sum for position ids
            }
        }
        return OnnxTensor.createTensor(environment, positionIds);
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

    private static int[] getTopKIndices(float[] array, int k) {
        int[] indices = new int[k];
        for (int i = 0; i < k; i++) {
            int maxIndex = -1;
            float maxValue = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < array.length; j++) {
                if (array[j] > maxValue) {
                    maxValue = array[j];
                    maxIndex = j;
                }
            }
            indices[i] = maxIndex;
            array[maxIndex] = Float.NEGATIVE_INFINITY;
        }
        return indices;
    }

}