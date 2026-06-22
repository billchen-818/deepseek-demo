use anyhow::Result;

use deepseek_demo::{ChatRequest, ChatResponse, FunctionDef, Message, Tool};

fn create_weather_tool() -> Tool {
    Tool {
        tool_type: "function".to_string(),
        function: FunctionDef {
            name: "get_weather".to_string(),
            description: "查询指定城市的实时天气信息".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、深圳"
                    }
                },
                "required": ["city"],
                "additionalProperties": false
            }),
        },
    }
}

fn execute_weather_tool(arguments: &str) -> Result<String> {
    // 解析模型生成的参数
    let args: serde_json::Value = serde_json::from_str(arguments)?;
    let city = args["city"].as_str().unwrap_or("未知城市");

    // 模拟返回（实际项目里这里调天气 API）
    let result = serde_json::json!({
        "city": city,
        "temperature": 25.6,
        "weather": "晴",
        "humidity": "45%",
        "wind": "东北风 3 级"
    });

    Ok(result.to_string())
}

const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/chat/completions";

/// 发送请求到 DeepSeek API
pub async fn chat_with_tools(
    api_key: &str,
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
) -> Result<ChatResponse> {
    let client = reqwest::Client::new();

    let body = ChatRequest {
        model: "deepseek-v4-pro".to_string(),
        messages,
        tools,
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
    Ok(data)
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置环境变量 DEEPSEEK_API_KEY");

    // Step 1: 准备工具定义
    let tools = vec![create_weather_tool()];

    // Step 2: 构造初始消息
    let mut messages: Vec<Message> = vec![
        Message {
            role: "system".into(),
            content: Some(
                "你是一个实用的天气助手。当用户询问天气时，使用 get_weather 工具查询。".into(),
            ),
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".into(),
            content: Some("北京今天天气怎么样？".into()),
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    // Step 3: 第一轮——发给模型，看它想干什么
    println!("=== 第一轮请求（用户问题 + 工具定义）===");
    let resp1 = chat_with_tools(&api_key, messages.clone(), Some(tools.clone())).await?;
    let choice1 = &resp1.choices[0];

    println!("finish_reason: {:?}", choice1.finish_reason);

    // Step 4: 判断 finish_reason，决定下一步
    if choice1.finish_reason.as_deref() == Some("tool_calls") {
        let tool_calls = choice1.message.tool_calls.as_ref().unwrap();

        // 把模型的 tool_calls 消息加入历史
        messages.push(choice1.message.clone());

        println!("\n模型想调用 {} 个工具：", tool_calls.len());
        for tc in tool_calls {
            println!("  → {} ({})", tc.function.name, tc.function.arguments);
        }

        // Step 5: 执行每个工具，把结果加回消息历史
        for tc in tool_calls {
            let tool_result = match tc.function.name.as_str() {
                "get_weather" => execute_weather_tool(&tc.function.arguments)?,
                _ => serde_json::json!({"error": "未知工具"}).to_string(),
            };

            println!("\n工具执行结果：{}", tool_result);

            messages.push(Message {
                role: "tool".into(),
                content: Some(tool_result),
                tool_calls: None,
                tool_call_id: Some(tc.id.clone()),
            });
        }

        // Step 6: 第二轮——把结果发回去，让模型生成最终回复
        println!("\n=== 第二轮请求（工具结果）===");
        let resp2 = chat_with_tools(&api_key, messages.clone(), Some(tools.clone())).await?;
        let choice2 = &resp2.choices[0];

        if let Some(content) = &choice2.message.content {
            println!("\n模型最终回复：\n{}", content);
        }
    } else {
        // 模型没调工具，直接回复了
        if let Some(content) = &choice1.message.content {
            println!("模型直接回复：\n{}", content);
        }
    }

    Ok(())
}
