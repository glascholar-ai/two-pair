# 代码规范

Python 代码必须遵守：

1. **Type hints**：所有函数签名（参数与返回值）必须带类型标注。
2. **长度限制**：单个函数不超过 200 行；单个文件不超过 1000 行。
3. **静态检查零告警**：每次改动 Python 代码后运行 `npx pyright`（配置见
   `pyrightconfig.json`，basic 模式，与 VS Code Pylance 一致），必须保持
   **0 errors / 0 warnings** 才算完成。收窄 pandas/Optional 类型时优先用
   显式判空与 `cast`，禁止用 `# type: ignore` 敷衍（确有第三方库标注缺陷
   时须附一行原因注释）。
4. 其余遵循 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)。
