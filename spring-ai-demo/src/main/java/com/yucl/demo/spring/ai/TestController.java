package com.yucl.demo.spring.ai;

import java.util.List;
import java.util.Map;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.Generation;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.openai.OpenAiChatClient;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.OpenAiEmbeddingOptions;
import org.springframework.ai.parser.BeanOutputParser;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController {

        private final EmbeddingClient embeddingClient;

        private final OpenAiChatClient chatClient;

        @Autowired
        public TestController(EmbeddingClient embeddingClient, OpenAiChatClient chatClient) {
                this.embeddingClient = embeddingClient;
                this.chatClient = chatClient;
        }

        @GetMapping("/ai/embedding")
        public Map embed(@RequestParam(value = "message", defaultValue = "Tell me a joke") String message) {
                EmbeddingResponse embeddingResponse = this.embeddingClient.embedForResponse(List.of(message));
                return Map.of("embedding", embeddingResponse);
        }

        @GetMapping("/ai/embedding2")
        public Map embed2(
                        @RequestParam(value = "message", defaultValue = "Hello World, World is big and salvation is near") String message) {
                EmbeddingResponse embeddingResponse = embeddingClient.call(
                                new EmbeddingRequest(List.of(message.split(",")),
                                                OpenAiEmbeddingOptions.builder()
                                                                .withModel("bge-m3")
                                                                .build()));
                return Map.of("embedding", embeddingResponse);
        }

        @GetMapping("/ai/chat")
        public String chat(@RequestParam(value = "message", defaultValue = "Tell me a joke") String message) {
                return chatClient.call(message);
        }

        @GetMapping("/ai/callFunc")
        public Map callFunc(@RequestParam(value = "message", defaultValue = "Tell me a joke") String message) {
                SystemMessage systemMessage = new SystemMessage("You are a helpful assistant");
                UserMessage userMessage = new UserMessage(
                                "What's the weather like in San Francisco, Tokyo, and Paris?");
                ChatResponse response = chatClient.call(new Prompt(List.of(systemMessage,
                                userMessage),
                                OpenAiChatOptions.builder().withFunction("CurrentWeather").build()));
                return Map.of("response", response);
        }

        @GetMapping("/ai/output")
        public ActorsFilms generate(@RequestParam(value = "actor", defaultValue = "Jeff Bridges") String actor) {
                var outputParser = new BeanOutputParser<>(ActorsFilms.class);

                String userMessage = """
                                Generate the filmography for the actor {actor}.
                                {format}
                                """;

                PromptTemplate promptTemplate = new PromptTemplate(userMessage,
                                Map.of("actor", actor, "format", outputParser.getFormat()));
                Prompt prompt = promptTemplate.create();
                Generation generation = chatClient.call(prompt).getResult();

                System.out.println(generation.getOutput().getContent());

                ActorsFilms actorsFilms = outputParser.parse(generation.getOutput().getContent());
                return actorsFilms;
        }

}
