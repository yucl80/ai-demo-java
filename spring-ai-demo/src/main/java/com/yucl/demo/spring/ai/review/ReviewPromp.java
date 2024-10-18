package com.yucl.demo.spring.ai.review;

public class ReviewPromp {
    public static String promp = "请审查以下代码变更，关注点包括代码质量、可读性及潜在的bug。请在50字以内描述可能的问题和改进建议： "
            + "\n\n"
            + "旧代码:\n"
            + "```python\n"
            + "{old_code}\n"
            + "```\n\n"
            + "新代码:\n"
            + "```python\n"
            + "{new_code}\n"
            + "```";
}
