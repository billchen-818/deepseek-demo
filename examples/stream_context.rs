use anyhow::Result;
use futures_util::StreamExt;
use std::io::{self, Write};

use deepseek_demo::{ChatRequest, Message};

const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/chat/completions";
const MODEL: &str = "deepseek-v4-pro";
const MAX_HISTORY_TURNS: usize = 8;

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

fn trim_history(messages: &mut Vec<Message>) {
    let max_messages = 1 + MAX_HISTORY_TURNS * 2;
    if messages.len() <= max_messages {
        return;
    }

    let system_message = messages.remove(0);
    let keep_from = messages.len().saturating_sub(MAX_HISTORY_TURNS * 2);
    let mut recent_messages = messages.split_off(keep_from);

    messages.clear();
    messages.push(system_message);
    messages.append(&mut recent_messages);
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置环境变量 DEEPSEEK_API_KEY");

    let mut messages = vec![Message {
        role: "system".into(),
        content: Some(
            "你是一个耐心、简洁的 AI 编程学习助手。请优先用中文回答，并记住本轮对话中的上下文。"
                .into(),
        ),
        tool_calls: None,
        tool_call_id: None,
    }];

    println!("流式输出 + 多轮上下文 Demo");
    println!("输入 exit / quit 结束；输入 clear 清空对话历史。");

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

        if input == "clear" {
            messages.truncate(1);
            println!("已清空对话历史。");
            continue;
        }

        messages.push(Message {
            role: "user".into(),
            content: Some(input.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        print!("AI: ");
        io::stdout().flush()?;

        let assistant_reply = chat_stream(&api_key, messages.clone()).await?;

        messages.push(Message {
            role: "assistant".into(),
            content: Some(assistant_reply),
            tool_calls: None,
            tool_call_id: None,
        });
        trim_history(&mut messages);
    }

    Ok(())
}
