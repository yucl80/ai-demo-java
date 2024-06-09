package com.yucl.demo;

import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.openai.OpenAiChatOptions;

/**
 * Unit test for simple App.
 */
public class AppTest {
    /**
     * Rigorous Test :-)
     */
    @Test
    public void shouldAnswerWithTrue() {
        assertTrue(true);
    }

    private void prompt() {
        String SYSTEM_PROMPT_FOR_CHAT_MODEL = """
                You are an expert in composing functions. You are given a question and a set of possible functions.
                Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
                If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
                also point it out. You should only return the function call in tool_calls sections.
                """;

        String USER_PROMPT_FOR_CHAT_MODEL = """
                    Questions:{user_prompt}\nHere is a list of functions in JSON format that you can invoke:\n{functions}.
                    Should you decide to return the function call(s),Put it in the format of [func1(params_name=params_value, params_name2=params_value2...), func2(params)]\n
                    NO other text MUST be included.
                """;

        SystemMessage systemMessage = new SystemMessage(SYSTEM_PROMPT_FOR_CHAT_MODEL);

        // PromptTemplate promptTemplate = new
        // PromptTemplate(USER_PROMPT_FOR_CHAT_MODEL);

        // Message userMessage = promptTemplate.createMessage(
        // Map.of("user_prompt", "What's the weather like in SanFrancisco? ",
        // "functions", "currentWeather"));

        OpenAiChatOptions chatOptions = OpenAiChatOptions.builder().withFunction("currentWeather").build();

        UserMessage userMessage = new UserMessage("What's the weather like in SanFrancisco? ");

    }
}
