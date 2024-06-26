package com.yucl.demo.spring.ai.rag.data;

import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/data")
public class DataController {

	private final DataLoadingService dataLoadingService;

	@Autowired
	public DataController(DataLoadingService dataLoadingService) {
		this.dataLoadingService = dataLoadingService;

	}

	@GetMapping("/load")
	public ResponseEntity<String> load() {
		try {
			this.dataLoadingService.load();
			return ResponseEntity.ok("Data loaded successfully!");
		} catch (Exception e) {
			return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
					.body("An error occurred while loading data: " + e.getMessage());
		}
	}

	@GetMapping("/loadfile")
	public ResponseEntity<String> loadFile(@RequestParam(value = "file") String file) {
		try {
			this.dataLoadingService.load(file);
			return ResponseEntity.ok("Data loaded successfully!");
		} catch (Exception e) {
			return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
					.body("An error occurred while loading data: " + e.getMessage());
		}
	}

	@ExceptionHandler(Exception.class)
	public ResponseEntity<String> handleException(Exception e) {
		return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
				.body("An error occurred in the controller: " + e.getMessage());
	}

}
