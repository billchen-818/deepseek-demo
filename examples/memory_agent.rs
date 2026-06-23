use anyhow::Result;
use futures_util::StreamExt;
use std::io::{self, Write};

use deepseek_demo::{ChatRequest, Message};

const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/chat/completions";
const MODEL: &str = "deepseek-v4-pro";
const MAX_SHORT_HISTORY_MESSAGES: usize = 8;
const MAX_CONTEXT_CHARS: usize = 6_000;

#[derive(Debug, Default)]
struct MemoryStore {
    summary: Option<String>,
    short_history: Vec<Message>,
    long_term_memories: Vec<String>,
}

impl MemoryStore {
    fn remember(&mut self, memory: String) {
        self.long_term_memories.push(memory);
    }

    fn add_user_message(&mut self, content: String) {
        self.short_history.push(text_message("user", content));
    }

    fn add_assistant_message(&mut self, content: String) {
        self.short_history.push(text_message("assistant", content));
        self.trim_short_history();
    }

    fn build_messages(&self, current_input: &str) -> Vec<Message> {
        let mut messages = vec![text_message(
            "system",
            "你是一个耐心、简洁的 AI Agent 学习助手。请优先用中文回答。\
             你会收到三类上下文：长期记忆、历史摘要、最近几轮短期记忆。\
             回答时优先使用和当前问题相关的信息。",
        )];

        let relevant_memories = self.search_long_term_memory(current_input, 3);
        if !relevant_memories.is_empty() {
            messages.push(text_message(
                "system",
                format!("长期记忆：\n{}", relevant_memories.join("\n")),
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
        let mut scored: Vec<(usize, &String)> = self
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

fn print_memory(memory: &MemoryStore) {
    println!("\n=== 短期记忆 short_history ===");
    for message in &memory.short_history {
        println!(
            "{}: {}",
            message.role,
            message.content.as_deref().unwrap_or_default()
        );
    }

    println!("\n=== 历史摘要 summary ===");
    println!("{}", memory.summary.as_deref().unwrap_or("(暂无)"));

    println!("\n=== 长期记忆 long_term_memories ===");
    for item in &memory.long_term_memories {
        println!("- {item}");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置环境变量 DEEPSEEK_API_KEY");
    let mut memory = MemoryStore::default();

    memory.remember("用户正在用 Rust 学习 AI Agent 开发。".into());
    memory.remember("当前项目已经实现过 POST 请求、函数调用、流式输出和多轮上下文。".into());

    println!("Agent 记忆 Demo：短期记忆 + 上下文裁剪 + 长期记忆");
    println!("命令：/remember 内容  写入长期记忆");
    println!("命令：/memory         查看当前记忆");
    println!("命令：/clear          清空短期记忆和摘要");
    println!("命令：exit / quit     退出");

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
            break;
        }

        if input == "/memory" {
            print_memory(&memory);
            continue;
        }

        if input == "/clear" {
            memory.short_history.clear();
            memory.summary = None;
            println!("已清空短期记忆和历史摘要，长期记忆保留。");
            continue;
        }

        if let Some(item) = input.strip_prefix("/remember ") {
            memory.remember(item.trim().to_string());
            println!("已写入长期记忆。");
            continue;
        }

        memory.add_user_message(input.to_string());
        let messages = memory.build_messages(input);

        print!("AI: ");
        io::stdout().flush()?;

        let assistant_reply = chat_stream(&api_key, messages).await?;
        memory.add_assistant_message(assistant_reply);
    }

    Ok(())
}
