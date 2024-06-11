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
		// manualFunctionCall();
		// autoFunctionCall();
		// System.exit(0);

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

}
