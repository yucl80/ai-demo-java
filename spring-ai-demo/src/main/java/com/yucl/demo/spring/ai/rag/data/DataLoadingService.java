package com.yucl.demo.spring.ai.rag.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.reader.ExtractedTextFormatter;
import org.springframework.ai.reader.pdf.PagePdfDocumentReader;
import org.springframework.ai.reader.pdf.ParagraphPdfDocumentReader;
import org.springframework.ai.reader.pdf.config.PdfDocumentReaderConfig;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;
import org.springframework.util.Assert;

@Service
public class DataLoadingService {

	private static final Logger logger = LoggerFactory.getLogger(DataLoadingService.class);

	@Value("classpath:/data/2023年中国证券业调查报告.pdf")
	private Resource pdfResource;

	private final VectorStore vectorStore;

	@Autowired
	private ResourceLoader resourceLoader;

	@Autowired
	public DataLoadingService(VectorStore vectorStore) {
		Assert.notNull(vectorStore, "VectorStore must not be null.");
		this.vectorStore = vectorStore;
	}

	public void load(String fileName) {
		String uri = "classpath:/data/" + fileName;
		if (fileName.endsWith("pdf")) {
			PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(resourceLoader.getResource(uri),
					PdfDocumentReaderConfig.builder()
							.withPageExtractedTextFormatter(ExtractedTextFormatter.builder()
									.withNumberOfBottomTextLinesToDelete(3)
									.withNumberOfTopPagesToSkipBeforeDelete(1)
									.build())
							.withPagesPerDocument(1).withReversedParagraphPosition(true)
							.build());

			var tokenTextSplitter = new TokenTextSplitter();

			logger.info(
					"Parsing document, splitting, creating embeddings and storing in vector store...  this will take a while.");
			this.vectorStore.accept(tokenTextSplitter.apply(pdfReader.get()));
			logger.info("Done parsing document, splitting, creating embeddings and storing in vector store");
		} else {
			TikaDocumentReader tikaDocumentReader = new TikaDocumentReader(resourceLoader.getResource(uri));

			var tokenTextSplitter = new TokenTextSplitter();

			logger.info(
					"Parsing document, splitting, creating embeddings and storing in vector store...  this will take a while.");
			this.vectorStore.accept(tokenTextSplitter.apply(tikaDocumentReader.get()));
			logger.info("Done parsing document, splitting, creating embeddings and storing in vector store");
		}

	}

	public void load() {
		PagePdfDocumentReader pdfReader = new PagePdfDocumentReader(this.pdfResource,
				PdfDocumentReaderConfig.builder()
						.withPageExtractedTextFormatter(ExtractedTextFormatter.builder()
								.withNumberOfBottomTextLinesToDelete(3)
								.withNumberOfTopPagesToSkipBeforeDelete(1)
								.build())
						.withPagesPerDocument(1).withReversedParagraphPosition(true)
						.build());

		var tokenTextSplitter = new TokenTextSplitter();

		logger.info(
				"Parsing document, splitting, creating embeddings and storing in vector store...  this will take a while.");
		this.vectorStore.accept(tokenTextSplitter.apply(pdfReader.get()));
		logger.info("Done parsing document, splitting, creating embeddings and storing in vector store");

	}

}
