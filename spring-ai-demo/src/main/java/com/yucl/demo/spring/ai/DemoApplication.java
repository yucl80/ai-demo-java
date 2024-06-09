package com.yucl.demo.spring.ai;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;

import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.openai.OpenAiChatClient;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion.Choice;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionFinishReason;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage.ToolCall;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.http.ResponseEntity;

import reactor.core.publisher.Flux;

@SpringBootApplication
public class DemoApplication implements CommandLineRunner {

	public static void main(String[] args) {
		SpringApplication.run(DemoApplication.class, args);
	}

	@Autowired
	EmbeddingClient embeddingClient;

	@Autowired
	OpenAiChatClient chatClient;

	@Override
	public void run(String... args) throws Exception {
		manualFunctionCall();
		autoFunctionCall();
		System.exit(0);

	}

	public void streamChat() {
		OpenAiChatOptions chatOptions = OpenAiChatOptions.builder().build();
		Flux<ChatResponse> responseFlux = chatClient.stream(new Prompt(new UserMessage("Hello"), chatOptions));
		responseFlux.collectList().block().stream().map(chatResponse -> {
			return chatResponse.getResults().get(0).getOutput().getContent();
		}).forEach(content -> {
			System.out.print(content);
		});

	}

	public void manualFunctionCall() {
		SystemMessage systemMessage = new SystemMessage(
				"You are a helpful assistant");
		// , You don't know everything about the current, please use tool or function to
		// answer the question

		UserMessage userMessage = new UserMessage(
				"中国平安的股票价格是多少? 今天深圳的天气怎么样?");

		OpenAiChatOptions chatOptions = OpenAiChatOptions.builder().withToolChoice("auto").withModel("functionary")
				.withFunctions(Set.of("get_stock_price", "WeatherInfo", "weatherFunctionTwo")).build();

		MyChatClient myChatClient = (MyChatClient) chatClient;
		ResponseEntity<ChatCompletion> chatCompletionResponse = myChatClient
				.callWithFunciton(new Prompt(List.of(systemMessage, userMessage), chatOptions));

		System.out.println(chatCompletionResponse);
		List<Object> toolCallResultList = doCallTolls(chatCompletionResponse);
		toolCallResultList.forEach(result -> {
			System.out.println(result);
		});

	}

	public void autoFunctionCall() {
		SystemMessage systemMessage = new SystemMessage(
				"You are a helpful assistant");
		// , You don't know everything about the current, please use tool or function to
		// answer the question
		UserMessage userMessage = new UserMessage(
				"今天深圳的天气是多少摄氏度?");

		OpenAiChatOptions chatOptions = OpenAiChatOptions.builder().withToolChoice("auto").withModel("chatglm3")
				.withFunctions(Set.of("get_stock_price", "WeatherInfo", "weatherFunctionTwo")).build();

		ChatResponse response = chatClient.call(new Prompt(List.of(systemMessage, userMessage), chatOptions));
		System.out.println(response);
	}

	public List<Object> doCallTolls(
			ResponseEntity<ChatCompletion> chatCompletionResponse) {
		List<Object> list = new ArrayList<>();
		for (Choice choice : chatCompletionResponse.getBody().choices()) {
			if (choice.finishReason().equals(ChatCompletionFinishReason.TOOL_CALLS)
					|| choice.finishReason().equals(ChatCompletionFinishReason.FUNCTION_CALL)) {
				for (ToolCall toolCall : choice.message().toolCalls()) {
					var functionName = toolCall.function().name();
					String functionArguments = toolCall.function().arguments();
					if (!chatClient.getFunctionCallbackRegister().containsKey(functionName)) {
						throw new IllegalStateException(
								"No function callback found for function name: " + functionName);
					}
					// String functionResponse =
					Object functionCallResult = chatClient.getFunctionCallbackRegister().get(functionName)
							.call(functionArguments);
					list.add(functionCallResult);
				}
			}

		}
		return list;

	}

}
