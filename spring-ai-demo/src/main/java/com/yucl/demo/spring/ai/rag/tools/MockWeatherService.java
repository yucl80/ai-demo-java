package com.yucl.demo.spring.ai.rag.tools;

import java.util.function.Function;

import com.fasterxml.jackson.annotation.JsonClassDescription;

public class MockWeatherService
        implements
        Function<com.yucl.demo.spring.ai.rag.tools.MockWeatherService.Request, com.yucl.demo.spring.ai.rag.tools.MockWeatherService.Response> {

    public enum Unit {
        C, F
    }

    public record Request(String location, Unit unit) {
    }

    public record Response(double temp, Unit unit) {
    }

    public Response apply(Request request) {
        System.out.println("call MockWeatherService");

        return new Response(35.5, Unit.C);
    }

}