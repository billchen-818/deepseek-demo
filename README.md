# 运行方式

```shell
export DEEPSEEK_API_KEY=sk-xxxxxxxx
cargo run
```

`cargo run` 会启动一个记忆 + 多轮命令行 Agent。每次启动都会创建一个新的
`agent_sessions/session_<timestamp>/memory.json`，用于保存本次会话的长期记忆。

## Examples

```shell
# 无上下文记忆
cargo run --example llm_no_memory

# 流式输出 + 多轮上下文
cargo run --example stream_context

# Agent 记忆：短期记忆 + 上下文裁剪 + 长期记忆
cargo run --example memory_agent
```
