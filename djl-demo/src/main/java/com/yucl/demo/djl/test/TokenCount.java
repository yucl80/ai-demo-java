package com.yucl.demo.djl.test;

import java.nio.file.Paths;
import java.util.Map;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;


public class TokenCount {
    
    public static void main(String[] args) throws Exception {
        String TOKENIZER_URI = "D:\\llm\\Qwen1.5-110B-Chat-AWQ\\tokenizer.json";       
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(Paths.get(TOKENIZER_URI), Map.of("modelMaxLength", "20000","maxLength","20000"));
        Encoding encodings = tokenizer.encode("this is a cat string 你好人们");
        System.out.println(encodings.getTokens().length); 
        System.out.println(String.join(",", encodings.getTokens()));
       

    }
}
