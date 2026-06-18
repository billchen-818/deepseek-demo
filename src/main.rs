use anyhow::Result;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/chat/completions";

// ── 请求相关数据结构 ──

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

// ── 响应相关数据结构 ──

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
}

// ── 非流式调用 ──

pub async fn chat(api_key: &str, messages: Vec<Message>) -> Result<String> {
    let client = reqwest::Client::new();

    let body = ChatRequest {
        model: "deepseek-v4-pro".to_string(),
        messages,
        stream: Some(false),
        temperature: Some(0.7),
        max_tokens: Some(4096),
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

    let data: ChatResponse = resp.json().await?;
    data.choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow::anyhow!("API 返回了空的 choices"))
}

// ── 流式调用 ──

pub async fn chat_stream(api_key: &str, messages: Vec<Message>) -> Result<String> {
    let client = reqwest::Client::new();

    let body = ChatRequest {
        model: "deepseek-chat".to_string(),
        messages,
        stream: Some(true),
        temperature: Some(0.7),
        max_tokens: Some(4096),
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
    let mut full_content = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || !line.starts_with("data: ") {
                continue;
            }

            let json_str = &line[6..]; // "data: ".len()

            if json_str == "[DONE]" {
                println!();
                break;
            }

            if let Ok(chunk_data) = serde_json::from_str::<serde_json::Value>(json_str) {
                if let Some(content) = chunk_data["choices"][0]["delta"]["content"].as_str() {
                    print!("{}", content);
                    full_content.push_str(content);
                }
            }
        }
    }

    Ok(full_content)
}

// ── 多轮对话 ──

pub struct Conversation {
    api_key: String,
    history: Vec<Message>,
}

impl Conversation {
    pub fn new(api_key: String, system_prompt: Option<&str>) -> Self {
        let mut history = Vec::new();
        if let Some(prompt) = system_prompt {
            history.push(Message {
                role: "system".into(),
                content: prompt.to_string(),
            });
        }
        Self { api_key, history }
    }

    pub async fn send(&mut self, user_input: &str) -> Result<String> {
        self.history.push(Message {
            role: "user".into(),
            content: user_input.to_string(),
        });

        let reply = chat(&self.api_key, self.history.clone()).await?;

        self.history.push(Message {
            role: "assistant".into(),
            content: reply.clone(),
        });

        Ok(reply)
    }
}

// ── 主函数 ──

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置环境变量 DEEPSEEK_API_KEY");

    println!("=== 非流式调用 ===");
    let messages = vec![
        Message {
            role: "system".into(),
            content: "你是一个 Rust 专家，回答简洁，代码示例不超过 15 行。".into(),
        },
        Message {
            role: "user".into(),
            content: "Rust 中 ? 操作符的原理是什么？".into(),
        },
    ];
    let reply = chat(&api_key, messages).await?;
    println!("回复：\n{}", reply);

    println!("\n=== 流式调用 ===");
    let messages = vec![
        Message {
            role: "system".into(),
            content: "你是一个 Rust 专家，回答简洁。".into(),
        },
        Message {
            role: "user".into(),
            content: "用三句话解释 Rust 的所有权系统。".into(),
        },
    ];
    let reply = chat_stream(&api_key, messages).await?;
    println!("\n完整回复长度：{} 字符", reply.len());

    Ok(())
}
