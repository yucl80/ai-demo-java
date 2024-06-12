package com.yucl.demo.spring.ai.rag.tools;

import java.util.function.Function;

import com.fasterxml.jackson.annotation.JsonPropertyDescription;

public class MockStockPriceService implements
        Function<com.yucl.demo.spring.ai.rag.tools.MockStockPriceService.StockName, com.yucl.demo.spring.ai.rag.tools.MockStockPriceService.StockPrice> {

    // @JsonClassDescription("the stock name")
    public record StockName(@JsonPropertyDescription("the stock name") String stockName) {
    }

    public record StockPrice(double price) {
    }

    public StockPrice apply(StockName stockName) {
        System.out.println("call MockStockPriceService");
        return new StockPrice(40d + (int) (Math.random() * 5));
    }

}