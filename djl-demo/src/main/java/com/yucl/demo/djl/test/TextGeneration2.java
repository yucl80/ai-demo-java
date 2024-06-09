package com.yucl.demo.djl.test;

import ai.djl.ModelException;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.modality.nlp.generate.SearchConfig;
import ai.djl.modality.nlp.generate.TextGenerator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.onnxruntime.zoo.nlp.textgeneration.OrtGptTranslatorFactory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.Collectors;

public final class TextGeneration2 {
    // {
    // System.setProperty("JL_DEFAULT_ENGINE", "PyTorch");
    // }

    private static final Logger logger = LoggerFactory.getLogger(TextGeneration1.class);

    public static void main(String[] args)
            throws IOException,
            TranslateException, ModelException {
        String[] ret4 = generateTextWithOnnxRuntimeBeam();
        System.out.println(Arrays.asList(ret4).stream().collect(Collectors.joining(" ")));
    }

    public static String[] generateTextWithOnnxRuntimeBeam()
            throws ModelException, IOException, TranslateException {
        SearchConfig config = new SearchConfig();
        config.setMaxSeqLength(60);
        long padTokenId = 220;
        config.setPadTokenId(padTokenId);

        // The model is converted optimum:
        // https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-using-past-keysvalues-in-the-decoder
        /*
         * optimum-cli export onnx --model gpt2 gpt2_onnx/
         *
         * from transformers import AutoTokenizer
         * from optimum.onnxruntime import ORTModelForCausalLM
         *
         * tokenizer = AutoTokenizer.from_pretrained("./gpt2_onnx/")
         * model = ORTModelForCausalLM.from_pretrained("./gpt2_onnx/")
         * inputs = tokenizer("My name is Arthur and I live in", return_tensors="pt")
         * gen_tokens = model.generate(**inputs)
         * print(tokenizer.batch_decode(gen_tokens))
         */
        // String url =
        // "https://djl-misc.s3.amazonaws.com/test/models/gpt2/gpt2_onnx.zip";
        String url = "file:///D:/llm/gpt2_onnx.zip";

        Criteria<NDList, CausalLMOutput> criteria = Criteria.builder()
                .setTypes(NDList.class, CausalLMOutput.class)
                .optEngine("OnnxRuntime")
                .optModelName("openai-community/gpt2")
                .optModelPath(Paths.get("D:\\llm\\gpt2"))
                // .optModelUrls(url)
                .optTranslatorFactory(new OrtGptTranslatorFactory())
                .build();
        String[] inputs = { "DeepMind Company is" };

        try (ZooModel<NDList, CausalLMOutput> model = criteria.loadModel();
                Predictor<NDList, CausalLMOutput> predictor = model.newPredictor();
                NDManager manager = model.getNDManager().newSubManager();
                // HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance("gpt2"))
                HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer
                        .newInstance(Paths.get(
                                "D:\\llm\\gpt2\\tokenizer.json"))) {

            TextGenerator generator = new TextGenerator(predictor, "beam", config);
            NDArray inputIdArray = encodeWithPadding(manager, tokenizer, inputs, padTokenId);

            NDArray outputs = generator.generate(inputIdArray);
            return decodeWithOffset(
                    tokenizer, outputs, generator.getPositionOffset().repeat(0, config.getBeam()));
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