package com.yucl.demo.spring.ai.rag.qa;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.yucl.demo.spring.ai.rag.spring.engine.EngineResponse;
import com.yucl.demo.spring.ai.rag.spring.engine.QueryEngine;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collector;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/qa/engine")
public class EngineController {

	private final QueryEngine queryEngine;

	public EngineController(QueryEngine queryEngine) {
		this.queryEngine = queryEngine;
	}

	@GetMapping
	public Map query(
			@RequestParam(value = "question", defaultValue = "What is the purpose of Carina?") String question) {
		EngineResponse engineResponse = this.queryEngine.call(question);
		Map<String, Object> response = new HashMap<>();
		response.put("question", question);
		String content = engineResponse.getChatResponse().getResult().getOutput().getContent();
		if (content != null) {
			content = content.replaceAll("\\\n", "");
		}
		Set<String> ref = engineResponse.getDocuments().stream()
				.map(doc -> doc.getMetadata().toString())
				.collect(Collectors.toSet());
		response.put("answer", content);
		response.put("ref", ref);
		return response;
	}

}
