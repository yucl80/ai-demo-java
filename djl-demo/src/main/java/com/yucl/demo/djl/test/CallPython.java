package com.yucl.demo.djl.test;

import java.io.IOException;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

public class CallPython {
    public static void main(String[] args)
            throws TranslateException, ModelNotFoundException, MalformedModelException, IOException {
        // Input preProcessing = new Input();
        // byte[] image = new byte[1024];
        // preProcessing.add("data", image);
        // preProcessing.addProperty("Content-Type", "image/jpeg");
        // // calling preprocess() function in model.py
        // preProcessing.addProperty("handler", "preprocess");
        // Output preprocessed = processingPredictor.predict(preProcessing);

        PythonTranslator translator = new PythonTranslator();
        Criteria<String, Classifications> criteria = Criteria.builder()
                .setTypes(String.class, Classifications.class)
                .optModelUrls("djl://ai.djl.pytorch/resnet")
                .optTranslator(translator)
                .build();
        String url = "https://resources.djl.ai/images/kitten.jpg";
        try (ZooModel<String, Classifications> model = criteria.loadModel();
                Predictor<String, Classifications> predictor = model.newPredictor()) {
            Classifications ret = predictor.predict(url);
            System.out.println(ret);
        }

        // unload python model
        translator.close();
    }

}
