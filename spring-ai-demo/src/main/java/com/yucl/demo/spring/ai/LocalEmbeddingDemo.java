package com.yucl.demo.spring.ai;

import java.util.List;
import java.util.Map;

import org.springframework.ai.transformers.TransformersEmbeddingClient;

public class LocalEmbeddingDemo {
    public static void main(String[] args) throws Exception {
        TransformersEmbeddingClient embeddingClient = new TransformersEmbeddingClient();
        // EmbeddingClient embeddingClient = new EmbeddingClient();

        // (optional) defaults to classpath:/onnx/all-MiniLM-L6-v2/tokenizer.json
        embeddingClient.setTokenizerResource("file:/D:/llm/bge-zh/tokenizer.json");
        // embeddingClient.setTokenizerResource(
        // "file:/D:/workspaces/ai_demo/learn-ai/learn-ai/src/main/resources/tokenizer.json");

        // (optional) defaults to classpath:/onnx/all-MiniLM-L6-v2/model.onnx
        embeddingClient.setModelResource("file:/D:/llm/bge-zh/model.onnx");

        embeddingClient.setModelOutputName("token_embeddings");

        // (optional) defaults to ${java.io.tmpdir}/spring-ai-onnx-model
        // Only the http/https resources are cached by default.
        embeddingClient.setResourceCacheDirectory("/tmp/onnx-zoo");

        // (optional) Set the tokenizer padding if you see an errors like:
        // "ai.onnxruntime.OrtException: Supplied array is ragged, ..."
        embeddingClient.setTokenizerOptions(Map.of("padding", "true"));

        embeddingClient.afterPropertiesSet();

        List<List<Double>> embeddings = embeddingClient.embed(List.of("Hello world", "World is big"));

        long begin = System.currentTimeMillis();
        for (int i = 0; i < 10; i++)
            embeddingClient.embed(List.of("World is big"));
        System.out.println("used time :" + (System.currentTimeMillis() - begin));
        System.out.println(embeddings.get(0).size());
    }
}
