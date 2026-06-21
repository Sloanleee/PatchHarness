---
name: bug_fix
description: Bug 修复时采用最小修改并保留可验证证据
triggers:
  - fix
  - bug
  - 修复
---

# Bug Fix Skill

- 先读取目标文件，再执行最小替换。
- 修改前确认请求允许编辑。
- 修改后查看 diff，并把 changed_files 写入报告。
- 如果无法明确定位 old/new 替换，不要猜测性大改。

