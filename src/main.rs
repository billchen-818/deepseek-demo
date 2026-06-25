use anyhow::{Context, Result};
use deepseek_demo::{ChatRequest, Message};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/chat/completions";
const MODEL: &str = "deepseek-v4-pro";
const MAX_SHORT_HISTORY_MESSAGES: usize = 10;
const MAX_CONTEXT_CHARS: usize = 8_000;

#[derive(Debug, Serialize, Deserialize, Default)]
struct MemoryConfig {
    long_term_memories: Vec<String>,
}

#[derive(Debug)]
struct AgentState {
    session_dir: PathBuf,
    memory_file: PathBuf,
    config: MemoryConfig,
    summary: Option<String>,
    short_history: Vec<Message>,
}

impl AgentState {
    fn new() -> Result<Self> {
        let session_dir = create_session_dir()?;
        let memory_file = session_dir.join("memory.json");
        let config = MemoryConfig {
            long_term_memories: vec![
                "用户正在学习 AI Agent 开发。".into(),
                "用户使用 Rust 和 DeepSeek API 做学习 Demo。".into(),
            ],
        };

        write_memory_file(&memory_file, &config)?;

        Ok(Self {
            session_dir,
            memory_file,
            config,
            summary: None,
            short_history: Vec::new(),
        })
    }

    fn remember(&mut self, memory: String) -> Result<()> {
        self.config.long_term_memories.push(memory);
        write_memory_file(&self.memory_file, &self.config)
    }

    fn add_user_message(&mut self, content: String) {
        self.short_history.push(text_message("user", content));
    }

    fn add_assistant_message(&mut self, content: String) {
        self.short_history.push(text_message("assistant", content));
        self.trim_short_history();
    }

    fn clear_short_memory(&mut self) {
        self.summary = None;
        self.short_history.clear();
    }

    fn build_messages(&self, current_input: &str) -> Vec<Message> {
        let mut messages = vec![text_message(
            "system",
            "你是一个命令行 AI Agent，正在帮助用户学习 AI Agent 开发。\
             请优先用中文回答，回答要清晰、具体、适合初学者。\
             你会收到长期记忆、历史摘要和最近几轮短期对话。\
             长期记忆是跨本次会话保存的重要信息，短期记忆是当前会话的最近上下文。",
        )];

        let relevant_memories = self.search_long_term_memory(current_input, 5);
        if !relevant_memories.is_empty() {
            messages.push(text_message(
                "system",
                format!("相关长期记忆：\n{}", relevant_memories.join("\n")),
            ));
        }

        if let Some(summary) = &self.summary {
            messages.push(text_message("system", format!("历史摘要：\n{summary}")));
        }

        messages.extend(self.short_history.clone());
        trim_by_chars(&mut messages, MAX_CONTEXT_CHARS);
        messages
    }

    fn search_long_term_memory(&self, query: &str, limit: usize) -> Vec<String> {
        let query_terms = keywords(query);
        if query_terms.is_empty() {
            return self
                .config
                .long_term_memories
                .iter()
                .take(limit)
                .map(|memory| format!("- {memory}"))
                .collect();
        }

        let mut scored: Vec<(usize, &String)> = self
            .config
            .long_term_memories
            .iter()
            .map(|memory| {
                let memory_terms = keywords(memory);
                let score = query_terms
                    .iter()
                    .filter(|term| memory_terms.iter().any(|item| item.contains(*term)))
                    .count();
                (score, memory)
            })
            .filter(|(score, _)| *score > 0)
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored
            .into_iter()
            .take(limit)
            .map(|(_, memory)| format!("- {memory}"))
            .collect()
    }

    fn trim_short_history(&mut self) {
        if self.short_history.len() <= MAX_SHORT_HISTORY_MESSAGES {
            return;
        }

        let overflow_count = self.short_history.len() - MAX_SHORT_HISTORY_MESSAGES;
        let old_messages: Vec<Message> = self.short_history.drain(..overflow_count).collect();
        let new_summary = summarize_messages(&old_messages);

        self.summary = Some(match &self.summary {
            Some(summary) => format!("{summary}\n{new_summary}"),
            None => new_summary,
        });
    }
}

async fn chat_stream(api_key: &str, messages: Vec<Message>) -> Result<String> {
    let client = reqwest::Client::new();
    let body = ChatRequest {
        model: MODEL.to_string(),
        messages,
        tools: None,
        stream: Some(true),
        temperature: Some(0.7),
        max_tokens: Some(2048),
    };

    let resp = client
        .post(DEEPSEEK_API_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await?;
        anyhow::bail!("API 返回错误 {status}: {text}");
    }

    let mut stream = resp.bytes_stream();
    let mut pending = String::new();
    let mut assistant_reply = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        pending.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(line_end) = pending.find('\n') {
            let line: String = pending.drain(..=line_end).collect();
            if handle_stream_line(line.trim(), &mut assistant_reply)? {
                println!();
                return Ok(assistant_reply);
            }
        }
    }

    if !pending.trim().is_empty() {
        handle_stream_line(pending.trim(), &mut assistant_reply)?;
    }

    println!();
    Ok(assistant_reply)
}

fn handle_stream_line(line: &str, assistant_reply: &mut String) -> Result<bool> {
    if line.is_empty() || !line.starts_with("data: ") {
        return Ok(false);
    }

    let data = &line["data: ".len()..];
    if data == "[DONE]" {
        return Ok(true);
    }

    let json: serde_json::Value = serde_json::from_str(data)?;
    if let Some(content) = json["choices"][0]["delta"]["content"].as_str() {
        print!("{content}");
        io::stdout().flush()?;
        assistant_reply.push_str(content);
    }

    Ok(false)
}

fn create_session_dir() -> Result<PathBuf> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("系统时间早于 Unix epoch")?
        .as_secs();
    let session_dir = PathBuf::from("agent_sessions").join(format!("session_{timestamp}"));

    fs::create_dir_all(&session_dir)
        .with_context(|| format!("创建会话目录失败: {}", session_dir.display()))?;

    Ok(session_dir)
}

fn write_memory_file(path: &Path, config: &MemoryConfig) -> Result<()> {
    let json = serde_json::to_string_pretty(config)?;
    fs::write(path, json).with_context(|| format!("写入记忆配置失败: {}", path.display()))
}

fn text_message(role: &str, content: impl Into<String>) -> Message {
    Message {
        role: role.into(),
        content: Some(content.into()),
        tool_calls: None,
        tool_call_id: None,
    }
}

fn summarize_messages(messages: &[Message]) -> String {
    let mut lines = Vec::new();
    for message in messages {
        let Some(content) = &message.content else {
            continue;
        };

        let compact = content.replace('\n', " ");
        let short = if compact.chars().count() > 120 {
            compact.chars().take(120).collect::<String>() + "..."
        } else {
            compact
        };
        lines.push(format!("{}: {}", message.role, short));
    }

    format!("较早对话摘要：{}", lines.join(" | "))
}

fn trim_by_chars(messages: &mut Vec<Message>, max_chars: usize) {
    while total_chars(messages) > max_chars && messages.len() > 1 {
        let removable = messages
            .iter()
            .position(|message| message.role != "system")
            .unwrap_or(1);
        messages.remove(removable);
    }
}

fn total_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .filter_map(|message| message.content.as_ref())
        .map(|content| content.chars().count())
        .sum()
}

fn keywords(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(str::trim)
        .filter(|item| item.chars().count() >= 2)
        .map(|item| item.to_lowercase())
        .collect()
}

fn print_help() {
    println!("\n命令：");
    println!("  /remember 内容  写入长期记忆 memory.json");
    println!("  /memory         查看长期记忆、短期记忆和摘要");
    println!("  /clear          清空当前会话的短期记忆和摘要");
    println!("  /session        查看本次会话目录");
    println!("  /help           查看命令");
    println!("  exit / quit     退出");
}

fn print_memory(agent: &AgentState) {
    println!("\n=== 会话目录 ===");
    println!("{}", agent.session_dir.display());

    println!("\n=== 长期记忆 memory.json ===");
    for memory in &agent.config.long_term_memories {
        println!("- {memory}");
    }

    println!("\n=== 历史摘要 summary ===");
    println!("{}", agent.summary.as_deref().unwrap_or("(暂无)"));

    println!("\n=== 短期记忆 short_history ===");
    if agent.short_history.is_empty() {
        println!("(暂无)");
        return;
    }

    for message in &agent.short_history {
        println!(
            "{}: {}",
            message.role,
            message.content.as_deref().unwrap_or_default()
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置环境变量 DEEPSEEK_API_KEY");
    let mut agent = AgentState::new()?;

    println!("记忆 + 多轮命令行 Agent Demo");
    println!("本次会话目录：{}", agent.session_dir.display());
    println!("长期记忆文件：{}", agent.memory_file.display());
    print_help();

    loop {
        print!("\n你: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if matches!(input, "exit" | "quit") {
            println!("已退出。长期记忆保存在：{}", agent.memory_file.display());
            break;
        }

        if input == "/help" {
            print_help();
            continue;
        }

        if input == "/session" {
            println!("会话目录：{}", agent.session_dir.display());
            println!("长期记忆文件：{}", agent.memory_file.display());
            continue;
        }

        if input == "/memory" {
            print_memory(&agent);
            continue;
        }

        if input == "/clear" {
            agent.clear_short_memory();
            println!("已清空当前会话的短期记忆和摘要，长期记忆仍保留。");
            continue;
        }

        if let Some(memory) = input.strip_prefix("/remember ") {
            let memory = memory.trim();
            if memory.is_empty() {
                println!("请在 /remember 后面输入要保存的内容。");
            } else {
                agent.remember(memory.to_string())?;
                println!("已写入长期记忆：{}", agent.memory_file.display());
            }
            continue;
        }

        agent.add_user_message(input.to_string());
        let messages = agent.build_messages(input);

        print!("AI: ");
        io::stdout().flush()?;

        match chat_stream(&api_key, messages).await {
            Ok(reply) => agent.add_assistant_message(reply),
            Err(err) => {
                agent.short_history.pop();
                println!("\n请求失败：{err:#}");
            }
        }
    }

    Ok(())
}
