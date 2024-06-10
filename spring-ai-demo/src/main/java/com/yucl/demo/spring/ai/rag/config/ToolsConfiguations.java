package com.yucl.demo.spring.ai.rag.config;

import java.util.function.Function;

import org.springframework.ai.model.function.FunctionCallback;
import org.springframework.ai.model.function.FunctionCallbackWrapper;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Description;
import org.springframework.web.client.RestTemplate;

import com.yucl.demo.spring.ai.rag.tools.MockStockPriceService;
import com.yucl.demo.spring.ai.rag.tools.MockWeatherService;
import com.yucl.demo.spring.ai.rag.tools.MockWeatherService2;
import com.yucl.demo.spring.ai.rag.tools.MockStockPriceService.StockName;
import com.yucl.demo.spring.ai.rag.tools.MockStockPriceService.StockPrice;

@Configuration
public class ToolsConfiguations {

    @Bean
    @Description("Get the current weather in a given location")
    public Function<MockWeatherService.Request, MockWeatherService.Response> currentWeather() {
        return new MockWeatherService();
    }

    @Bean("get_stock_price")
    @Description("Get the latest stock price ,获取股票的价格,查询股票价格") // function description
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
                .withName("get_current_weather")
                .withDescription("Get the current weather in a given location")
                .withResponseConverter((response) -> "" + response.temp() + response.unit())
                .build();
    }

    @Bean
    public Function<MockWeatherService2.Request, MockWeatherService2.Response> weatherQueryFunction() {
        MockWeatherService2 weatherService = new MockWeatherService2();
        return (weatherService::apply);
    }

}
