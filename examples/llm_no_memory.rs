use anyhow::Result;

use deepseek_demo::{ChatRequest, ChatResponse, Message};

const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/chat/completions";
const MODEL: &str = "deepseek-v4-pro";

async fn chat_once(api_key: &str, user_content: &str) -> Result<String> {
    let client = reqwest::Client::new();

    let messages = vec![
        Message {
            role: "system".into(),
            content: Some("你是一个简洁的中文 AI 学习助手。".into()),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".into(),
            content: Some(user_content.to_string()),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let body = ChatRequest {
        model: MODEL.to_string(),
        messages,
        tools: None,
        stream: Some(false),
        temperature: Some(0.7),
        max_tokens: Some(1024),
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
    Ok(data.choices[0]
        .message
        .content
        .clone()
        .unwrap_or_else(|| "(模型没有返回文本内容)".into()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置环境变量 DEEPSEEK_API_KEY");

    let first_question = "我正在学习 Rust，我学到了所有权。";
    let second_question = "我学到哪里了？";

    println!("无上下文记忆 Demo");
    println!("这个例子会发起两次完全独立的请求。");
    println!("第二次请求不会携带第一次的 user/assistant 消息，所以模型不知道你上一轮说过什么。\n");

    println!("=== 第一次请求 ===");
    println!("你: {first_question}");
    let first_reply = chat_once(&api_key, first_question).await?;
    println!("AI: {first_reply}\n");

    println!("=== 第二次请求（没有带上第一次上下文）===");
    println!("你: {second_question}");
    let second_reply = chat_once(&api_key, second_question).await?;
    println!("AI: {second_reply}\n");

    println!(
        "观察点：如果想让第二次知道你学到了所有权，就必须把第一次对话保存到 messages 里一起发送。"
    );

    Ok(())
}
