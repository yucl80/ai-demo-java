package com.yucl.demo.spring.ai;

import java.time.Duration;
import java.util.List;
import java.util.function.Function;

import org.springframework.ai.autoconfigure.openai.OpenAiChatProperties;
import org.springframework.ai.autoconfigure.openai.OpenAiConnectionProperties;
import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.model.function.FunctionCallbackContext;
import org.springframework.ai.model.function.FunctionCallbackWrapper;
import org.springframework.ai.openai.OpenAiChatClient;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.boot.web.client.ClientHttpRequestFactories;
import org.springframework.boot.web.client.ClientHttpRequestFactorySettings;
import org.springframework.boot.web.client.RestClientCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Description;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.CollectionUtils;
import org.springframework.util.StringUtils;
import org.springframework.web.client.ResponseErrorHandler;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestTemplate;

import com.yucl.demo.spring.ai.MockStockPriceService.StockName;
import com.yucl.demo.spring.ai.MockStockPriceService.StockPrice;

@Configuration
public class Configuations {

    @Bean
    public OpenAiChatClient oenAiChatClient(OpenAiConnectionProperties commonProperties,
            OpenAiChatProperties chatProperties, RestClient.Builder restClientBuilder,
            List<FunctionCallback> toolFunctionCallbacks, FunctionCallbackContext functionCallbackContext,
            RetryTemplate retryTemplate, ResponseErrorHandler responseErrorHandler) {

        var openAiApi = openAiApi(chatProperties.getBaseUrl(), commonProperties.getBaseUrl(),
                chatProperties.getApiKey(), commonProperties.getApiKey(), restClientBuilder, responseErrorHandler);

        if (!CollectionUtils.isEmpty(toolFunctionCallbacks)) {
            chatProperties.getOptions().getFunctionCallbacks().addAll(toolFunctionCallbacks);
        }

        return new MyChatClient(openAiApi, chatProperties.getOptions(), functionCallbackContext, retryTemplate);

    }

    private OpenAiApi openAiApi(String baseUrl, String commonBaseUrl, String apiKey, String commonApiKey,
            RestClient.Builder restClientBuilder, ResponseErrorHandler responseErrorHandler) {
        String resolvedBaseUrl = StringUtils.hasText(baseUrl) ? baseUrl : commonBaseUrl;
        String resolvedApiKey = StringUtils.hasText(apiKey) ? apiKey : commonApiKey;
        return new OpenAiApi(resolvedBaseUrl, resolvedApiKey, restClientBuilder, responseErrorHandler);
    }

    @Bean
    public RestClientCustomizer restClientCustomizer() {
        return restClientBuilder -> restClientBuilder
                .requestFactory(ClientHttpRequestFactories.get(ClientHttpRequestFactorySettings.DEFAULTS
                        .withConnectTimeout(Duration.ofSeconds(1))
                        .withReadTimeout(Duration.ofSeconds(12000))));
    }

    @Bean
    @Description("Get the current weather") // function
                                            // description
    public Function<MockWeatherService.Request, MockWeatherService.Response> currentWeather() {
        return new MockWeatherService();
    }

    @Bean("get_stock_price")
    @Description("Get the latest stock price ") // function description
    public Function<StockName, StockPrice> getStockPrice() {
        return new MockStockPriceService();
    }

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }

    @Bean
    public FunctionCallback weatherFunctionInfo() {

        return FunctionCallbackWrapper.builder(new MockWeatherService())
                .withName("WeatherInfo")
                .withDescription("Get the weather in location")
                .withResponseConverter((response) -> "" + response.temp() + response.unit())
                .build();
    }

    @Bean
    public Function<MockWeatherService2.Request, MockWeatherService2.Response> weatherFunctionTwo() {
        MockWeatherService2 weatherService = new MockWeatherService2();
        return (weatherService::apply);
    }

}
