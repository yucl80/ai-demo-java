package com.yucl.demo.spring.ai.rag.config;

import java.time.Duration;
import java.util.List;

import org.springframework.ai.autoconfigure.openai.OpenAiChatProperties;
import org.springframework.ai.autoconfigure.openai.OpenAiConnectionProperties;
import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.document.DocumentRetriever;
import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.model.function.FunctionCallbackContext;
import org.springframework.ai.openai.OpenAiChatClient;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.web.client.ClientHttpRequestFactories;
import org.springframework.boot.web.client.ClientHttpRequestFactorySettings;
import org.springframework.boot.web.client.RestClientCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;
import org.springframework.web.client.ResponseErrorHandler;
import org.springframework.web.client.RestClient;

import com.yucl.demo.spring.ai.rag.spring.engine.QueryEngine;
import com.yucl.demo.spring.ai.rag.spring.retriever.VectorStoreRetriever;
import com.yucl.demo.spring.ai.rag.tools.FunctionLLMClient;

@Configuration
public class ApplicationConfiguration {
	@Bean
	public RestClientCustomizer restClientCustomizer() {
		return restClientBuilder -> restClientBuilder
				.requestFactory(ClientHttpRequestFactories.get(ClientHttpRequestFactorySettings.DEFAULTS
						.withConnectTimeout(Duration.ofSeconds(1))
						.withReadTimeout(Duration.ofSeconds(12000))));
	}

	@Bean
	QueryEngine carinaQueryEngine(ChatClient chatClient, VectorStore vectorStore) {
		DocumentRetriever documentRetriever = new VectorStoreRetriever(vectorStore); // searches
																						// all
																						// documents,
																						// useful
																						// only
																						// in
																						// demos
		return new QueryEngine(chatClient, documentRetriever);
	}

	@Primary
	@Bean
	public OpenAiChatClient OpenAiChatClient(OpenAiConnectionProperties commonProperties,
			OpenAiChatProperties chatProperties, RestClient.Builder restClientBuilder,
			List<FunctionCallback> toolFunctionCallbacks, FunctionCallbackContext functionCallbackContext,
			RetryTemplate retryTemplate, ResponseErrorHandler responseErrorHandler) {
		var openAiApi = openAiApi(chatProperties.getBaseUrl(),
				commonProperties.getBaseUrl(),
				chatProperties.getApiKey(), commonProperties.getApiKey(), restClientBuilder,
				responseErrorHandler);

		if (!CollectionUtils.isEmpty(toolFunctionCallbacks)) {
			chatProperties.getOptions().getFunctionCallbacks().addAll(toolFunctionCallbacks);
		}

		return new OpenAiChatClient(openAiApi, chatProperties.getOptions(), functionCallbackContext, retryTemplate);

	}

	@Bean
	public FunctionLLMClient functionLLMClient(OpenAiConnectionProperties commonProperties,
			OpenAiChatProperties chatProperties, RestClient.Builder restClientBuilder,
			List<FunctionCallback> toolFunctionCallbacks, FunctionCallbackContext functionCallbackContext,
			RetryTemplate retryTemplate, ResponseErrorHandler responseErrorHandler) {

		// var openAiApi = openAiApi(chatProperties.getBaseUrl(),
		// commonProperties.getBaseUrl(),
		// chatProperties.getApiKey(), commonProperties.getApiKey(), restClientBuilder,
		// responseErrorHandler);
		var openAiApi = openAiApi("http://192.168.32.129:7000", commonProperties.getBaseUrl(),
				chatProperties.getApiKey(), commonProperties.getApiKey(), restClientBuilder, responseErrorHandler);

		if (!CollectionUtils.isEmpty(toolFunctionCallbacks)) {
			chatProperties.getOptions().getFunctionCallbacks().addAll(toolFunctionCallbacks);
		}

		return new FunctionLLMClient(openAiApi, chatProperties.getOptions(), functionCallbackContext, retryTemplate);

	}

	private OpenAiApi openAiApi(String baseUrl, String commonBaseUrl, String apiKey, String commonApiKey,
			RestClient.Builder restClientBuilder, ResponseErrorHandler responseErrorHandler) {
		String resolvedBaseUrl = StringUtils.hasText(baseUrl) ? baseUrl : commonBaseUrl;
		String resolvedApiKey = StringUtils.hasText(apiKey) ? apiKey : commonApiKey;
		return new OpenAiApi(resolvedBaseUrl, resolvedApiKey, restClientBuilder, responseErrorHandler);
	}

}
