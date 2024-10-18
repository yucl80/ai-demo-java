package com.yucl.demo.spring.ai.graph.nebula;

import java.util.Collections;

import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.client.RestTemplate;

public class LLMClient {
    private final RestTemplate restTemplate;

    private static String baseUrl = "http://127.0.0.1:8000/v1/";

    private static String CHAT_ENDPOINT= "chat/completions";

    public LLMClient(RestTemplate restTemplate){
        this.restTemplate = restTemplate;
    }

    public String invoke(String prompt){
         HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setAccept(Collections.singletonList(MediaType.APPLICATION_JSON));
        //  headers.setBearerAuth(apiKey);

        // 构建请求体
        String requestBody = "{\"model\": \"glm-4\", \"messages\": [{\"role\": \"user\", \"content\": \"" + prompt + "\"}]}";

        HttpEntity<String> entity = new HttpEntity<>(requestBody, headers);

        String url = baseUrl +CHAT_ENDPOINT;

        ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, entity, String.class);

        return response.getBody();

    }


       

}
