package com.yucl.demo.spring.ai.rag;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.openai.OpenAiChatClient;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion.Choice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionFinishReason;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import org.springframework.http.ResponseEntity;

import com.yucl.demo.spring.ai.rag.tools.FunctionLLMClient;

import reactor.core.publisher.Flux;

@SpringBootApplication
public class DemoApplication implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args);
	}

	@Autowired
	EmbeddingClient embeddingClient;

	@Autowired
	OpenAiChatClient openAiChatClient;

	@Autowired
	@Qualifier("functionLLMClient")
	FunctionLLMClient functionLLMClient;

	@Override
	public void run(String... args) throws Exception {
		manualFunctionCall();
		// autoFunctionCall();
		System.exit(0);

	}

	public void streamChat() {
		OpenAiChatOptions chatOptions = OpenAiChatOptions.builder().build();
		Flux<ChatResponse> responseFlux = openAiChatClient.stream(new Prompt(new UserMessage("Hello"), chatOptions));
		responseFlux.collectList().block().stream().map(chatResponse -> {
			return chatResponse.getResults().get(0).getOutput().getContent();
		}).forEach(content -> {
			System.out.print(content);
		});

	}

	public void manualFunctionCall() {
		SystemMessage systemMessage = new SystemMessage("You are a helpful assistant");

		UserMessage userMessage = new UserMessage("中国石化的股票价格是多少? ");

		OpenAiChatOptions funcOptions = OpenAiChatOptions.builder().withToolChoice("auto").withModel("functionary")
				.withFunctions(Set.of("get_stock_price", "weatherQueryFunction")).build();

		ResponseEntity<ChatCompletion> chatCompletionResponse = functionLLMClient
				.callWithFunciton(new Prompt(List.of(systemMessage, userMessage), funcOptions));

		System.out.println(chatCompletionResponse);
		List<FunctionCallResult> toolCallResultList = doCallTolls(chatCompletionResponse);
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

		var result = openAiChatClient.call(new Prompt(msgList));
		System.err.println(result);

	}

	public void autoFunctionCall() {
		SystemMessage systemMessage = new SystemMessage(
				"You are a helpful assistant");
		// , You don't know everything about the current, please use tool or function to
		// answer the question
		UserMessage userMessage = new UserMessage(
				"What's the weather like in San Francisco?");

		OpenAiChatOptions chatOptions = OpenAiChatOptions.builder().withToolChoice("auto").withModel("chatglm3")
				.withFunctions(Set.of("get_stock_price", "weatherQueryFunction")).build();

		ChatResponse response = openAiChatClient.call(new Prompt(List.of(userMessage), chatOptions));
		System.out.println(response);
	}

	public List<FunctionCallResult> doCallTolls(
			ResponseEntity<ChatCompletion> chatCompletionResponse) {
		List<FunctionCallResult> list = new ArrayList<>();
		for (Choice choice : chatCompletionResponse.getBody().choices()) {
			if (choice.finishReason().equals(ChatCompletionFinishReason.TOOL_CALLS)
					|| choice.finishReason().equals(ChatCompletionFinishReason.FUNCTION_CALL)) {
				for (ToolCall toolCall : choice.message().toolCalls()) {
					var functionName = toolCall.function().name();
					String functionArguments = toolCall.function().arguments();
					if (!functionLLMClient.getFunctionCallbackRegister().containsKey(functionName)) {
						throw new IllegalStateException(
								"No function callback found for function name: " + functionName);
					}
					// String functionResponse =
					Object functionCallResult = functionLLMClient.getFunctionCallbackRegister().get(functionName)
							.call(functionArguments);
					list.add(
							new FunctionCallResult(functionName, functionCallResult, toolCall.id(), functionArguments));
				}
			}

		}
		return list;

	}

}

record FunctionCallResult(String functionName, Object result, String id, String functionArguments) {
}
