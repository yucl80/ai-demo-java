package com.yucl.demo.spring.ai.rag.test;

import java.time.Duration;
import java.util.List;

import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.openai.OpenAiChatClient;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.OpenAiEmbeddingClient;
import org.springframework.ai.openai.OpenAiEmbeddingOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.boot.web.client.ClientHttpRequestFactories;
import org.springframework.boot.web.client.ClientHttpRequestFactorySettings;
import org.springframework.web.client.RestClient;

public class OpenAiClient {
        static final String baseUrl = "http://192.168.32.129:8000/";
        static final String openAiToken = "NoKey";

        public static void main(String[] args) {
                RestClient.Builder restClientBuilder = RestClient.builder()
                                .requestFactory(ClientHttpRequestFactories.get(ClientHttpRequestFactorySettings.DEFAULTS
                                                .withConnectTimeout(Duration.ofSeconds(1))
                                                .withReadTimeout(Duration.ofSeconds(120))));
                OpenAiApi openAiApi = new OpenAiApi(baseUrl, openAiToken, restClientBuilder);

                var openAiChatOptions = OpenAiChatOptions.builder()
                                .withModel("chatglm3")
                                .withTemperature(0.4f)
                                .withMaxTokens(200)
                                .build();

                OpenAiChatClient client = new OpenAiChatClient(openAiApi, openAiChatOptions);
                var result = client.call("Generate the names of 5 famous pirates.");
                System.err.println(result);

                embedding(openAiApi);

        }

        private static void embedding(OpenAiApi openAiApi) {
                OpenAiEmbeddingOptions openAiEmbeddingOptions = new OpenAiEmbeddingOptions();
                openAiEmbeddingOptions.setModel("bge-large-zh-v1.5");
                // openAiEmbeddingOptions.setEncodingFormat("float");
                // openAiEmbeddingOptions.setUser("test");

                OpenAiEmbeddingClient embeddingClient = new OpenAiEmbeddingClient(openAiApi, MetadataMode.EMBED,
                                openAiEmbeddingOptions, RetryUtils.DEFAULT_RETRY_TEMPLATE);

                // embeddingClient.embed("World is big and salvation is near");
                EmbeddingResponse embeddingResponse = embeddingClient
                                .embedForResponse(List.of("Hello World", "World is big and salvation is near"));

                System.out.println(embeddingResponse);
                System.out.println(embeddingResponse.getResults().get(0).getOutput().size());
        }

}
