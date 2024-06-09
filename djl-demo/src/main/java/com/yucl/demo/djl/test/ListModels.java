package com.yucl.demo.djl.test;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public final class ListModels {

    private static final Logger logger = LoggerFactory.getLogger(ListModels.class);

    private ListModels() {
    }

    public static void main(String[] args) throws IOException, ModelNotFoundException {
        System.setProperty("https.proxyHost", "127.0.0.1");
        System.setProperty("https.proxyPort", "49879");
        System.setProperty("java.net.useSystemProxies", "true");
        Map<Application, List<Artifact>> models = ModelZoo.listModels();
        models.forEach(
                (app, list) -> {
                    String appName = app.toString();
                    list.forEach(artifact -> logger.info("{} {}", appName, artifact));
                });
    }
}
