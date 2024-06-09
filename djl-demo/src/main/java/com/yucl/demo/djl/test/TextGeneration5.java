package com.yucl.demo.djl.test;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.onnxruntime.zoo.nlp.textgeneration.OrtGptTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.onnxruntime.OrtUtil;
import ai.djl.modality.nlp.translator.SimpleText2TextTranslator;

public class TextGeneration5 {

    public static void main(String[] args) throws ModelException, TranslateException, IOException {
        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optEngine("OnnxRuntime")
                .optModelName("llama")
                .optModelPath(Path.of("D:\\llm\\llama_quantize\\model_quantized.onnx"))
                .optTranslator(new GPT2Translator())
                // .optTranslatorFactory(new OrtGptTranslatorFactory())
                .build();

        String TOKENIZER_URI = "file:/D:\\llm\\llama_quantize\\tokenizer.json";
        // 加载模型
        NDManager manager = NDManager.newBaseManager();
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(TOKENIZER_URI), Map.of());

        String sentences = "how to learn ai";
        Encoding encodings = tokenizer.encode(sentences);
        long[] input_ids = encodings.getIds();
        long[] attention_mask = encodings.getAttentionMask();
        long[] position_ids = new long[input_ids.length];
        for (int i = 0; i < input_ids.length; i++) {
            position_ids[i] = i;
        }

        // 这里假设您已经处理并序列化了这些输入数据
        // 假设 inputData 包含 input_ids, attention_mask, position_ids
        NDArray inputIds = manager.create(LongBuffer.wrap(input_ids), new Shape(1, input_ids.length), DataType.INT64);
        inputIds.setName("input_ids");
        NDArray attentionMask = manager.create(LongBuffer.wrap(attention_mask), new Shape(1, input_ids.length),
                DataType.INT64);
        attentionMask.setName("attention_mask");
        NDArray positionIds = manager.create(LongBuffer.wrap(position_ids), new Shape(1, input_ids.length),
                DataType.INT64);
        positionIds.setName("position_ids");

        try (ZooModel<NDList, NDList> model = criteria.loadModel();) {
            // 创建预测器
            try (Predictor<NDList, NDList> predictor = model.newPredictor()) {
                // 输入文本

                // 生成文本
                Map<String, NDArray> inputData = new HashMap<>();
                NDList ndList = new NDList(inputIds, attentionMask, positionIds);
                NDList output = predictor.predict(ndList);
                // long[] tokenIds = output.totoLongArray();
                // List<Integer> tokenIdsList =
                // Arrays.stream(tokenIds).boxed().collect(Collectors.toList());
                // String generatedText = tokenizer.decode(tokenIdsList);
                // 打印生成的文本
                NDArray data = output.get(0);
                long[] shape = data.getShape().getShape();
                System.out.println(output.get(0).getShape());

                // 选择最大概率对应的 token ID (贪心搜索)
                // long[] outputIds = new long[logits[0].length];
                // long nextTokenId = argmax(logits[0][logits[0].length - 1]);

                float[] xxx = output.get(0).toFloatArray();
                float[][][] logits = (float[][][]) OrtUtil.reshape(xxx, shape);
                long[] genIds = new long[(int) shape[1]];
                for (int i = 0; i < shape[1]; i++) {
                    long nextTokenId = argmax(logits[0][i]);
                    genIds[i] = nextTokenId;
                }
                String outputText = tokenizer.decode(genIds);
                System.out.println("Generated text: " + outputText);

                System.out.println(output);
            }
        }
    }

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

class GPT2Translator implements Translator<NDList, NDList> {

    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) {
        // 根据需要处理输入
        return input;
    }

    @Override
    public NDList processOutput(TranslatorContext ctx, NDList list) {
        return list;
    }

    @Override
    public Batchifier getBatchifier() {
        return null; // 如果不需要批处理则返回 null
    }
}