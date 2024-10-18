import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class LargeTextSummarizer {
    private static final int MAX_CONCURRENT_TASKS = 10;
    private static final int CHUNK_SIZE = 1000; // 每个块的大小，假设为1000字符
    private static final int MAX_CHUNK_SIZE = 8000; // 最大块的大小，假设为8k字符

    public static void main(String[] args) throws Exception {
        // 假设你有一个超大文本
        String largeText = "你的超大文本内容...";

        // 将文本分割为多个小块
        List<String> chunks = splitTextIntoChunks(largeText, CHUNK_SIZE);

        // 创建一个线程池，限制最大并发数为10
        ExecutorService executor = Executors.newFixedThreadPool(MAX_CONCURRENT_TASKS);

        // 初始摘要
        List<String> initialSummaries = new ArrayList<>();
        List<Future<String>> futures = new ArrayList<>();
        for (String chunk : chunks) {
            futures.add(executor.submit(() -> summarizeChunk(chunk)));
        }
        for (Future<String> future : futures) {
            initialSummaries.add(future.get());
        }

        // 递归归并摘要结果
        String finalSummary = mergeAndSummarize(initialSummaries, executor);

        // 输出最终的摘要
        System.out.println(finalSummary);

        // 关闭线程池
        executor.shutdown();
    }

    private static List<String> splitTextIntoChunks(String text, int chunkSize) {
        List<String> chunks = new ArrayList<>();
        for (int i = 0; i < text.length(); i += chunkSize) {
            chunks.add(text.substring(i, Math.min(text.length(), i + chunkSize)));
        }
        return chunks;
    }

    private static String summarizeChunk(String chunk) {
        // 这里你需要调用LLM对chunk进行摘要
        // 假设summary是调用LLM后返回的摘要
        String summary = "摘要结果"; // 这里需要替换为实际的LLM调用
        return summary;
    }

    private static String mergeAndSummarize(List<String> summaries, ExecutorService executor) throws ExecutionException, InterruptedException {
        while (summaries.size() > 1) {
            List<Future<String>> futures = new ArrayList<>();
            List<String> mergedSummaries = new ArrayList<>();
            StringBuilder currentChunk = new StringBuilder();

            for (String summary : summaries) {
                if (currentChunk.length() + summary.length() <= MAX_CHUNK_SIZE) {
                    currentChunk.append(summary).append("\n");
                } else {
                    mergedSummaries.add(currentChunk.toString());
                    currentChunk.setLength(0);
                    currentChunk.append(summary).append("\n");
                }
            }

            if (currentChunk.length() > 0) {
                mergedSummaries.add(currentChunk.toString());
            }

            for (String mergedSummary : mergedSummaries) {
                futures.add(executor.submit(() -> summarizeChunk(mergedSummary)));
            }

            summaries.clear();
            for (Future<String> future : futures) {
                summaries.add(future.get());
            }
        }
        return summaries.get(0);
    }
}
