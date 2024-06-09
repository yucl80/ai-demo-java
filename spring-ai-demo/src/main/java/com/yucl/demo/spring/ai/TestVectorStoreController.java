package com.yucl.demo.spring.ai;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingClient;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;

public class TestVectorStoreController {
        private EmbeddingClient embeddingClient;

        private VectorStore vectorStore;

        @GetMapping("/ai/chroma/init")
        public String chromaInit() {
                List<Document> documents = List.of(
                                new Document(
                                                "Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!! Spring AI rocks!!",
                                                Map.of("meta1", "meta1")),
                                new Document("The World is Big and Salvation Lurks Around the Corner"),
                                new Document("You walk forward facing the past and you turn back toward the future.",
                                                Map.of("meta2", "meta2")));
                vectorStore.add(documents);
                return "ok";
        }

        @GetMapping("/ai/chroma/add")
        public String chromaAdd(@RequestParam(value = "content") String content,
                        @RequestParam(value = "metaKey") String metaKey,
                        @RequestParam(value = "metaValue") String metaValue) {
                List<Document> documents = List.of(new Document(content, Map.of(metaKey, metaValue)));
                vectorStore.add(documents);
                return "ok";
        }

        @GetMapping("/ai/chroma/search")
        public Map chromaSearch(@RequestParam(value = "query", defaultValue = "Spring") String query,
                        @RequestParam(value = "top_k", defaultValue = "3") int top_k,
                        @RequestParam(value = "similarity", defaultValue = "0.7") double similarity) {
                FilterExpressionBuilder b = new FilterExpressionBuilder();
                List<Document> result = vectorStore.similaritySearch(SearchRequest.defaults()
                                .withQuery(query)
                                .withTopK(top_k)
                                .withSimilarityThreshold(similarity)
                // .withFilterExpression(b.and(
                // b.in("john", "jill"),
                // b.eq("article_type", "blog")).build())
                );
                return Map.of("response",
                                result.stream().map(doc -> Map.of("content", doc.getContent(), "metadata",
                                                doc.getMetadata()))
                                                .collect(Collectors.toList()));

        }

}
