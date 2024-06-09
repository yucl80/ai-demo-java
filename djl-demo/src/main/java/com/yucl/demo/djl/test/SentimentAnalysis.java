package com.yucl.demo.djl.test;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

/**
 * An example of inference using DistilBERT for Sentiment Analysis.
 *
 * <p>
 * See this <a
 * href=
 * "https://github.com/deepjavalibrary/djl/blob/master/examples/docs/sentiment_analysis.md">doc</a>
 * for information about this example.*
 */
public final class SentimentAnalysis {

    private static final Logger logger = LoggerFactory.getLogger(SentimentAnalysis.class);

    private SentimentAnalysis() {
    }

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        System.setProperty("java.net.useSystemProxies", "true");
        Classifications classifications = SentimentAnalysis.predict();
        logger.info(classifications.toString());
    }

    public static Classifications predict()
            throws MalformedModelException,
            ModelNotFoundException,
            IOException,
            TranslateException {
        String input = "I like DJL. DJL is the best DL framework!";
        logger.info("input Sentence: {}", input);

        Criteria<String, Classifications> criteria = Criteria.builder()
                .optApplication(Application.NLP.SENTIMENT_ANALYSIS)
                .setTypes(String.class, Classifications.class)
                .optEngine("PyTorch")

                // This model was traced on CPU and can only run on CPU
                .optDevice(Device.cpu())
                .optProgress(new ProgressBar())
                .build();

        try (ZooModel<String, Classifications> model = criteria.loadModel();
                Predictor<String, Classifications> predictor = model.newPredictor()) {
            return predictor.predict(input);
        }
    }
}