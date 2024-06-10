package org.springframework.ai.chat.messages;

import java.util.Map;

public class ToolMessage extends AbstractMessage {

    public ToolMessage(String content) {
        super(MessageType.TOOL, content);
    }

    public ToolMessage(String content, Map<String, Object> properties) {
        super(MessageType.TOOL, content, properties);
    }

    @Override
    public String toString() {
        return "FunctionMessage{" + "content='" + getContent() + '\'' + ", properties=" + properties + ", messageType="
                + messageType + '}';
    }

}