package com.yucl.demo.spring.ai.rag.tools;

import java.util.function.Function;

import com.fasterxml.jackson.annotation.JsonClassDescription;
import com.fasterxml.jackson.annotation.JsonPropertyDescription;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Mock 3rd party weather service.
 *
 * @author Christian Tzolov
 */
public class MockWeatherService2 implements Function<MockWeatherService2.Request, MockWeatherService2.Response> {

    /**
     * Weather Function request.
     */
    @JsonInclude(Include.NON_NULL)
    @JsonClassDescription("Get the current weather in a given location")
    public record Request(
            @JsonProperty(required = true, value = "location") @JsonPropertyDescription("The city and state e.g. San Francisco, CA") String location,
            @JsonProperty(required = true, value = "unit") @JsonPropertyDescription("Temperature unit") Unit unit) {
    }

    /**
     * Temperature units.
     */
    public enum Unit {

        /**
         * Celsius.
         */
        C("metric"),
        /**
         * Fahrenheit.
         */
        F("imperial");

        /**
         * Human readable unit name.
         */
        public final String unitName;

        private Unit(String text) {
            this.unitName = text;
        }

    }

    /**
     * Weather Function response.
     */
    public record Response(double temp, double feels_like, double temp_min, double temp_max, int pressure, int humidity,
            Unit unit) {
    }

    @Override
    public Response apply(Request request) {

        double temperature = 0;
        if (request.location().contains("Paris")) {
            temperature = 15;
        } else if (request.location().contains("Tokyo")) {
            temperature = 10;
        } else if (request.location().contains("San Francisco")) {
            temperature = 30;
        } else {
            temperature = 40;
        }
        System.out.println("call MockWeatherService2");
        return new Response(temperature, 15, 20, 2, 53, 45, Unit.C);
    }

}