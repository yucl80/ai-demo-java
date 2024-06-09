package com.yucl.demo.spring.ai;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.model.function.FunctionCallbackContext;
import org.springframework.ai.openai.OpenAiChatClient;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.openai.api.OpenAiApi;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletion;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionMessage;
import org.springframework.ai.openai.api.OpenAiApi.ChatCompletionRequest;

import org.springframework.http.ResponseEntity;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.CollectionUtils;

public class MyChatClient extends OpenAiChatClient {
    private static final Logger logger = LoggerFactory.getLogger(MyChatClient.class);

    private OpenAiChatOptions defaultOptions;

    public MyChatClient(OpenAiApi openAiApi, OpenAiChatOptions options, FunctionCallbackContext functionCallbackContext,
            RetryTemplate retryTemplate) {
        // TODO Auto-generated constructor stub
        super(openAiApi, options, functionCallbackContext, retryTemplate);
        this.defaultOptions = options;

    }

    public ResponseEntity<ChatCompletion> callWithFunciton(Prompt prompt) {

        ChatCompletionRequest request = createRequest(prompt, false);

        return this.retryTemplate.execute(ctx -> this.callWithFunctionSupport(request)

        );
    }

    protected ResponseEntity<ChatCompletion> callWithFunctionSupport(OpenAiApi.ChatCompletionRequest request) {
        return this.doChatCompletion(request);
    }

    /**
     * Accessible for testing.
     */
    ChatCompletionRequest createRequest(Prompt prompt, boolean stream) {

        Set<String> functionsForThisRequest = new HashSet<>();

        List<ChatCompletionMessage> chatCompletionMessages = prompt.getInstructions()
                .stream()
                .map(m -> new ChatCompletionMessage(m.getContent(),
                        ChatCompletionMessage.Role.valueOf(m.getMessageType().name())))
                .toList();

        ChatCompletionRequest request = new ChatCompletionRequest(chatCompletionMessages, stream);

        if (prompt.getOptions() != null) {
            if (prompt.getOptions() instanceof ChatOptions runtimeOptions) {
                OpenAiChatOptions updatedRuntimeOptions = ModelOptionsUtils.copyToTarget(runtimeOptions,
                        ChatOptions.class, OpenAiChatOptions.class);

                Set<String> promptEnabledFunctions = this.handleFunctionCallbackConfigurations(updatedRuntimeOptions,
                        IS_RUNTIME_CALL);
                functionsForThisRequest.addAll(promptEnabledFunctions);

                request = ModelOptionsUtils.merge(updatedRuntimeOptions, request, ChatCompletionRequest.class);
            } else {
                throw new IllegalArgumentException("Prompt options are not of type ChatOptions: "
                        + prompt.getOptions().getClass().getSimpleName());
            }
        }

        if (this.defaultOptions != null) {

            Set<String> defaultEnabledFunctions = this.handleFunctionCallbackConfigurations(this.defaultOptions,
                    !IS_RUNTIME_CALL);

            functionsForThisRequest.addAll(defaultEnabledFunctions);

            request = ModelOptionsUtils.merge(request, this.defaultOptions, ChatCompletionRequest.class);
        }

        // Add the enabled functions definitions to the request's tools parameter.
        if (!CollectionUtils.isEmpty(functionsForThisRequest)) {

            request = ModelOptionsUtils.merge(
                    OpenAiChatOptions.builder().withTools(this.getFunctionTools(functionsForThisRequest)).build(),
                    request, ChatCompletionRequest.class);
        }

        return request;
    }

    private List<OpenAiApi.FunctionTool> getFunctionTools(Set<String> functionNames) {
        return this.resolveFunctionCallbacks(functionNames).stream().map(functionCallback -> {
            var function = new OpenAiApi.FunctionTool.Function(functionCallback.getDescription(),
                    functionCallback.getName(), functionCallback.getInputTypeSchema());
            return new OpenAiApi.FunctionTool(function);
        }).toList();
    }

}
