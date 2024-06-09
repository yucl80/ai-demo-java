package com.yucl.demo.djl.test;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

public class ObjectDetect {
    private static final Logger logger = LoggerFactory.getLogger(ObjectDetect.class);

    public static void main(String[] args) throws IOException, ModelException, TranslateException {
        Path imageFile = Paths
                .get("D:\\workspaces\\ai_demo\\learn-ai\\learn-ai\\src\\main\\resources\\pose_soccer.png");
        Image img = ImageFactory.getInstance().fromFile(imageFile);

        // First detect a person from the main image using PyTorch engine
        Image person = detectPersonWithPyTorchModel(img);

        // If no person is detected, we can't pass to next model, log and exit.
        if (person == null) {
            logger.warn("No person found in image.");
            return;
        }

    }

    private static Image detectPersonWithPyTorchModel(Image img)
            throws MalformedModelException,
            ModelNotFoundException,
            IOException,
            TranslateException {

        // Criteria object to load the model from model zoo
        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
                .optApplication(Application.CV.OBJECT_DETECTION)
                .setTypes(Image.class, DetectedObjects.class)
                .optProgress(new ProgressBar())
                // We specify a resnet50 model that runs using the PyTorch engine here.
                .optFilter("size", "300")
                .optFilter("backbone", "resnet50")
                .optFilter("dataset", "coco")
                .optEngine("PyTorch") // Use PyTorch engine
                .build();

        // Inference call to detect the person form the image.
        DetectedObjects detectedObjects;
        try (ZooModel<Image, DetectedObjects> ssd = criteria.loadModel();
                Predictor<Image, DetectedObjects> predictor = ssd.newPredictor()) {
            detectedObjects = predictor.predict(img);
        }

        // Get the first resulting image of the person and return it
        List<DetectedObjects.DetectedObject> items = detectedObjects.items();
        for (DetectedObjects.DetectedObject item : items) {
            System.out.println(item.getClassName());
            if ("person".equals(item.getClassName())) {
                Rectangle rect = item.getBoundingBox().getBounds();
                int width = img.getWidth();
                int height = img.getHeight();
                return img.getSubImage(
                        (int) (rect.getX() * width),
                        (int) (rect.getY() * height),
                        (int) (rect.getWidth() * width),
                        (int) (rect.getHeight() * height));
            }
        }
        return null;
    }

}
