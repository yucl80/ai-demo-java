package com.yucl.demo;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

public class MergeRecursively {

    public List<String> group(List<String> textList) {
        List<String> groupList = new ArrayList<>();
        return groupList;
    }

    private ExecutorService executor = Executors.newFixedThreadPool(12);

    private String callAI(String text) {
        return "";
    }

    private void callAIService(Text text, BlockingQueue<Text> textQueue) {
        String aiText = callAI(text.text);
        try {
            textQueue.put(new Text(aiText, text.level + 1));
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    /**
     * 从队列中获取文本数据并组织生成新的文本数据
     * 
     * @param textQueue 文本数据队列
     * @throws InterruptedException 当线程被中断时抛出此异常
     */
    private void generate(BlockingQueue<Text> textQueue) throws InterruptedException {
        // 存储提交给线程池的任务
        List<MergeFuture> futureList = new ArrayList<>();
        // 定义文本最大长度
        int MAX_LEN = 1000;
        // 初始化当前层级
        int level = 0;
        // 构建文本字符串
        StringBuilder textBuilder = new StringBuilder();
        // 上一个文本数据，用于判断是否需要提交当前构建的文本
        Text lastText = null;
        while (true) {
            // 从队列中获取文本数据，超时时间为1秒
            Text text = textQueue.poll(1, TimeUnit.SECONDS);
            if (text != null) {
                // 当文本长度超过最大长度或文本层级发生变化时，提交当前构建的文本
                if (textBuilder.length() + text.text.length() >= MAX_LEN
                        || (lastText != null && text.level != lastText.level)) {
                    String str = textBuilder.toString();
                    // 提交任务给线程池，并存储任务的未来对象和层级
                    Future<?> future = executor.submit(() -> callAIService(new Text(str, level), textQueue));
                    futureList.add(new MergeFuture(future, text.level));
                    // 重置文本构建器
                    textBuilder = new StringBuilder();
                } else {
                    // 否则，继续追加文本数据
                    textBuilder.append(text.text);
                }
                // 更新上一个文本数据
                lastText = text;
            }
            // 判断当前层级的所有任务是否已完成
            boolean finished = futureList.stream().filter(t -> t.level == text.level - 1)
                    .filter(t -> !t.future.isDone()).count() == 0;
            if (finished) {
                // 如果文本构建器非空，提交当前构建的文本
                if (!textBuilder.isEmpty()) {
                    String str = textBuilder.toString();
                    Future<?> future = executor.submit(() -> callAIService(new Text(str, level), textQueue));
                    futureList.add(new MergeFuture(future, text.level));
                    // 重置文本构建器
                    textBuilder = new StringBuilder();
                }
            }
            // 如果当前层级的所有任务已完成，且只有一个任务，则退出循环
            if (futureList.stream().filter(t -> t.level == text.level - 1).count() == 1) {
                break;
            }
        }
    }

    private void putToQueue(String text, int level, BlockingQueue<Text> textQueue) {
        textQueue.add(new Text(text, level));

    }

    public void handle(List<String> textList) {
        BlockingQueue<Text> textQueue = new LinkedBlockingQueue<>();
        for (String text : textList) {
            putToQueue(text, 0, textQueue);
        }

        Future future1 = executor.submit(() -> {
            try {
                generate(textQueue);
            } catch (InterruptedException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        });

        try {
            future1.get();
        } catch (InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (ExecutionException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }

    }

    static class Text {
        private String text;
        private int level;

        public Text(String text, int level) {
            this.text = text;
            this.level = level;
        }

        public String getText() {
            return text;
        }

        public int getLevel() {
            return level;
        }

    }

    static class MergeFuture {
        private Future future;
        private int level;

        public Future getFutrue() {
            return future;
        }

        public int getLevel() {
            return level;
        }

        public MergeFuture(Future future, int level) {
            this.future = future;
            this.level = level;
        }

    }

}