package com.yucl.demo.djl.test;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import com.google.gson.reflect.TypeToken;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.net.URL;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PythonTranslator implements NoBatchifyTranslator<String, Classifications> {

    private ZooModel<Input, Output> model;
    private Predictor<Input, Output> predictor;

    @Override
    public void prepare(TranslatorContext ctx) throws ModelException, IOException {
        if (predictor == null) {
            Criteria<Input, Output> criteria = Criteria.builder()
                    .setTypes(Input.class, Output.class)
                    .optModelPath(Paths.get("src/test/resources/resnet18"))
                    .optEngine("Python")
                    .build();
            model = criteria.loadModel();
            predictor = model.newPredictor();
        }
    }

    @Override
    public NDList processInput(TranslatorContext ctx, String url)
            throws IOException, TranslateException {
        Input input = new Input();
        try (InputStream is = new URL(url).openStream()) {
            input.add("data", Utils.toByteArray(is));
        }
        input.addProperty("Content-Type", "image/jpeg");
        // calling preprocess() function in model.py
        input.addProperty("handler", "preprocess");
        Output output = predictor.predict(input);
        if (output.getCode() != 200) {
            throw new TranslateException("Python preprocess() failed: " + output.getMessage());
        }

        return output.getDataAsNDList(ctx.getNDManager());
    }

    @Override
    public Classifications processOutput(TranslatorContext ctx, NDList list)
            throws TranslateException {
        Input input = new Input();
        input.add("data", list);
        // calling postprocess() function in processing.py
        input.addProperty("handler", "postprocess");
        Output output = predictor.predict(input);
        if (output.getCode() != 200) {
            throw new TranslateException("Python postprocess() failed: " + output.getMessage());
        }

        String json = output.getData().getAsString();
        Type type = new TypeToken<Map<String, Double>>() {
        }.getType();
        Map<String, Double> map = JsonUtils.GSON.fromJson(json, type);
        List<String> keys = new ArrayList<>(map.keySet());
        List<Double> values = new ArrayList<>(map.values());
        return new Classifications(keys, values);
    }

    public void close() {
        if (predictor != null) {
            predictor.close();
            model.close();
            predictor = null;
            model = null;
        }
    }
}
