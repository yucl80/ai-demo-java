package com.yucl.demo.spring.ai.graph.nebula;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

@Configuration
public class AppConfig {
    @Bean
    public RestTemplate restTemplate() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setReadTimeout(5000); // 设置读取超时时间为5000毫秒
        factory.setConnectTimeout(5000); // 设置连接超时时间为5000毫秒
        return new RestTemplate(factory);

    }

}
