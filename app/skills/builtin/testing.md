---
name: testing
description: 修复后运行测试并汇总验证结果
triggers:
  - test
  - verify
  - 测试
  - 验证
---

# Testing Skill

- 使用请求中的 test_command 优先验证。
- 记录 returncode、stdout 和 stderr。
- returncode 为 0 才视为验证通过。
- 测试失败时不要掩盖失败，应把失败信息留在结构化报告中。

