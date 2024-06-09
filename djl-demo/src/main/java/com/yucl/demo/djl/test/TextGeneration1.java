package com.yucl.demo.djl.test;

import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.modality.nlp.generate.SearchConfig;
import ai.djl.modality.nlp.generate.TextGenerator;
import ai.djl.modality.nlp.generate.MyTextGenerator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.onnxruntime.zoo.nlp.textgeneration.OrtGptTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.DeferredTranslatorFactory;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public final class TextGeneration1 {

    private static final Logger logger = LoggerFactory.getLogger(TextGeneration1.class);

    private TextGeneration1() {
    }

    public static void main(String[] args)
            throws IOException,
            TranslateException, ModelException {
        String ret1 = generateTextWithPyTorchGreedy();
        logger.info("{}", ret1);
        // String[] ret2 = generateTextWithPyTorchContrastive();
        // logger.info("{}", ret2[0]);
        // String[] ret3 = generateTextWithPyTorchBeam();
        // logger.info("{}", ret3[0]);
        // String[] ret4 = generateTextWithOnnxRuntimeBeam();
    }

    public static String generateTextWithPyTorchGreedy()
            throws ModelNotFoundException,
            MalformedModelException,
            IOException,
            TranslateException {
        SearchConfig config = new SearchConfig();
        config.setMaxSeqLength(100);

        // You can use src/main/python/trace_gpt2.py to trace gpt2 model
        // String url =
        // "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_pt.zip";
        String url = "file:///D:/llm/gpt2_pt.zip";

        Criteria<NDList, CausalLMOutput> criteria = Criteria.builder()
                .setTypes(NDList.class, CausalLMOutput.class)
                // .optModelUrls(url)
                // .optModelPath(Paths.get("D:/llm/gpt2_pt"))
                .optModelName("llama")
                .optModelPath(Path.of("D:\\llm\\llama_quantize\\model_quantized.onnx"))
                .optEngine("OnnxRuntime")
                .optTranslator(new MyTranslator())
                // .optTranslatorFactory(new OrtGptTranslatorFactory())
                // .optTranslatorFactory(new DeferredTranslatorFactory())
                .build();
        String input = "China is";

        try (ZooModel<NDList, CausalLMOutput> model = criteria.loadModel();
                Predictor<NDList, CausalLMOutput> predictor = model.newPredictor();
                NDManager manager = model.getNDManager().newSubManager();
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer
                        .newInstance(Paths.get("D:\\llm\\llama_quantize\\tokenizer.json"))) {

            MyTextGenerator generator = new MyTextGenerator(predictor, "greedy", config);

            Encoding encoding = tokenizer.encode(input);
            long[] inputIds = encoding.getIds();
            NDArray inputIdArray = manager.create(inputIds).expandDims(0);

            NDArray output = generator.generate(inputIdArray);
            long[] outputIds = output.toLongArray();

            return tokenizer.decode(outputIds);
        }
    }

    public static NDArray encodeWithPadding(
            NDManager manager, HuggingFaceTokenizer tokenizer, String[] inputs, long padTokenId) {
        NDArray inputIdArray = null;
        for (String input : inputs) {
            long[] inputIds = tokenizer.encode(input).getIds();
            NDArray deltaInputIdArray = manager.create(inputIds).expandDims(0);
            if (inputIdArray == null) {
                inputIdArray = deltaInputIdArray;
            } else {
                if (inputIdArray.getShape().get(1) > deltaInputIdArray.getShape().get(1)) {
                    // pad deltaInputIdArray
                    long batchSize = deltaInputIdArray.getShape().get(0);
                    long deltaSeqLength = inputIdArray.getShape().get(1) - deltaInputIdArray.getShape().get(1);
                    deltaInputIdArray = manager.full(
                            new Shape(batchSize, deltaSeqLength),
                            padTokenId,
                            DataType.INT64)
                            .concat(deltaInputIdArray, 1);
                } else if (inputIdArray.getShape().get(1) < deltaInputIdArray.getShape().get(1)) {
                    // pad inputIdArray
                    long batchSize = inputIdArray.getShape().get(0);
                    long deltaSeqLength = deltaInputIdArray.getShape().get(1) - inputIdArray.getShape().get(1);
                    inputIdArray = manager.full(
                            new Shape(batchSize, deltaSeqLength),
                            padTokenId,
                            DataType.INT64)
                            .concat(inputIdArray, 1);
                }
                inputIdArray = inputIdArray.concat(deltaInputIdArray, 0);
            }
        }
        return inputIdArray;
    }

    public static String[] decodeWithOffset(
            HuggingFaceTokenizer tokenizer, NDArray outputIds, NDArray offset) {
        long batchSize = outputIds.getShape().get(0);
        String[] outputs = new String[(int) batchSize];
        for (int i = 0; i < batchSize; i++) {
            long startIndex = offset.getLong(i);
            long[] outputId = outputIds.get("{},{}:", i, startIndex).toLongArray();
            outputs[i] = tokenizer.decode(outputId);
        }
        return outputs;
    }
}