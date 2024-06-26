package com.yucl.demo.spring.ai.rag.test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.openai.OpenAiChatClient;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion.Choice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionFinishReason;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.yucl.demo.spring.ai.rag.tools.FunctionLLMClient;

@RestController
public class FunctionCallController {
    @Autowired
    OpenAiChatClient openAiChatClient;

    @Autowired
    @Qualifier("functionLLMClient")
    FunctionLLMClient functionLLMClient;

    @GetMapping("/autoCall")
    public String autoFunctionCall(
            @RequestParam(value = "question", defaultValue = "What's the weather like in San Francisco? ") String question) {
        SystemMessage systemMessage = new SystemMessage("You are a helpful assistant");

        UserMessage userMessage = new UserMessage(question);

        OpenAiChatOptions chatOptions = OpenAiChatOptions.builder().withToolChoice("auto").withModel("chatglm3")
                .withFunctions(Set.of("get_stock_price", "weatherQueryFunction")).build();

        ChatResponse response = openAiChatClient.call(new Prompt(List.of(userMessage), chatOptions));
        return response.getResult().getOutput().getContent();
    }

    @GetMapping("/manualCall")
    public String manualFunctionCall(
            @RequestParam(value = "question", defaultValue = "中国石化的股票价格是多少? ") String question) {

        SystemMessage systemMessage = new SystemMessage("You are a helpful assistant");

        UserMessage userMessage = new UserMessage(question);

        OpenAiChatOptions funcOptions = OpenAiChatOptions.builder().withToolChoice("auto").withModel("functionary")
                .withFunctions(Set.of("get_stock_price", "weatherQueryFunction")).build();

        ResponseEntity<ChatCompletion> chatCompletionResponse = functionLLMClient
                .callWithFunciton(new Prompt(List.of(systemMessage, userMessage), funcOptions));

        System.out.println(chatCompletionResponse);
        List<FunctionCallResult> toolCallResultList = callFunction(chatCompletionResponse);
        toolCallResultList.forEach(result -> {
            System.out.println(result);
        });

        List<Message> msgList = new ArrayList<>();
        msgList.add(systemMessage);
        msgList.add(userMessage);

        toolCallResultList.forEach(toolCall -> {
            String msg = "call tool: " + toolCall.functionName() + "(" + toolCall.functionArguments()
                    + "), function call result:" + toolCall.result();

            System.out.println(msg);

            AssistantMessage functionMessage = new AssistantMessage(msg, Map.of("function_name",
                    toolCall.functionName(), "result", toolCall.result(), "arguments", toolCall.functionArguments()));

            msgList.add(functionMessage);
        });

        OpenAiChatOptions chartOptions = OpenAiChatOptions.builder().withToolChoice("auto").withModel("qwen")
                .withFunctions(Set.of("get_stock_price", "weatherQueryFunction")).build();

        var result = openAiChatClient.call(new Prompt(msgList, chartOptions));
        return result.getResult().getOutput().getContent();

    }

    public List<FunctionCallResult> callFunction(ResponseEntity<ChatCompletion> chatCompletionResponse) {
        List<FunctionCallResult> list = new ArrayList<>();
        ChatCompletion body = chatCompletionResponse.getBody();
        if (body != null) {
            for (Choice choice : body.choices()) {
                if (choice.finishReason().equals(ChatCompletionFinishReason.TOOL_CALLS)
                        || choice.finishReason().equals(ChatCompletionFinishReason.FUNCTION_CALL)) {
                    for (ToolCall toolCall : choice.message().toolCalls()) {
                        var functionName = toolCall.function().name();
                        String functionArguments = toolCall.function().arguments();
                        if (!functionLLMClient.getFunctionCallbackRegister().containsKey(functionName)) {
                            throw new IllegalStateException(
                                    "No function callback found for function name: " + functionName);
                        }
                        Object functionCallResult = functionLLMClient.getFunctionCallbackRegister().get(functionName)
                                .call(functionArguments);
                        list.add(
                                new FunctionCallResult(functionName, functionCallResult, toolCall.id(),
                                        functionArguments));
                    }
                }

            }
        }
        return list;

    }

}

record FunctionCallResult(String functionName, Object result, String id, String functionArguments) {
}
