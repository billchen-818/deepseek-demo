# 运行方式

```shell
export DEEPSEEK_API_KEY=sk-xxxxxxxx
cargo run
```

## Examples

```shell
# 无上下文记忆
cargo run --example llm_no_memory

# 流式输出 + 多轮上下文
cargo run --example stream_context

# Agent 记忆：短期记忆 + 上下文裁剪 + 长期记忆
cargo run --example memory_agent
```
