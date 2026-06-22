use anyhow::Result;
use futures_util::StreamExt;
use std::collections::BTreeMap;

use deepseek_demo::{
    ChatRequest, ChatResponse, Choice, FunctionCall, FunctionDef, Message, Tool, ToolCall,
};

const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/chat/completions";

// ═══════════════════════════════════════════
//  API 调用（非流式 + 流式）
// ═══════════════════════════════════════════

pub async fn chat(
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

    Ok(resp.json().await?)
}

pub async fn chat_stream(
    api_key: &str,
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
) -> Result<ChatResponse> {
    let client = reqwest::Client::new();

    let body = ChatRequest {
        model: "deepseek-v4-pro".to_string(),
        messages,
        tools,
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
    let mut final_content = String::new();
    let mut finish_reason: Option<String> = None;
    let mut tool_calls_map: BTreeMap<usize, ToolCall> = BTreeMap::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || !line.starts_with("data: ") {
                continue;
            }

            let json_str = &line[6..];

            if json_str == "[DONE]" {
                break;
            }

            if let Ok(data) = serde_json::from_str::<serde_json::Value>(json_str) {
                let delta = &data["choices"][0]["delta"];

                if let Some(content) = delta["content"].as_str() {
                    print!("{}", content);
                    final_content.push_str(content);
                }

                if let Some(tc_array) = delta["tool_calls"].as_array() {
                    for tc in tc_array {
                        let idx = tc["index"].as_u64().unwrap_or(0) as usize;
                        let entry = tool_calls_map.entry(idx).or_insert_with(|| ToolCall {
                            id: String::new(),
                            call_type: "function".to_string(),
                            function: FunctionCall {
                                name: String::new(),
                                arguments: String::new(),
                            },
                        });

                        if let Some(id) = tc["id"].as_str() {
                            entry.id = id.to_string();
                        }
                        if let Some(name) = tc["function"]["name"].as_str() {
                            entry.function.name.push_str(name);
                        }
                        if let Some(args) = tc["function"]["arguments"].as_str() {
                            entry.function.arguments.push_str(args);
                        }
                    }
                }

                if let Some(fr) = data["choices"][0]["finish_reason"].as_str() {
                    finish_reason = Some(fr.to_string());
                }
            }
        }
    }

    println!();

    let message = if tool_calls_map.is_empty() {
        Message {
            role: "assistant".into(),
            content: Some(final_content),
            tool_calls: None,
            tool_call_id: None,
        }
    } else {
        Message {
            role: "assistant".into(),
            content: None,
            tool_calls: Some(tool_calls_map.into_values().collect()),
            tool_call_id: None,
        }
    };

    Ok(ChatResponse {
        choices: vec![Choice {
            message,
            finish_reason,
        }],
    })
}

// ═══════════════════════════════════════════
//  工具定义与执行
// ═══════════════════════════════════════════

fn create_tools() -> Vec<Tool> {
    vec![
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
                            "description": "城市名称"
                        }
                    },
                    "required": ["city"],
                    "additionalProperties": false
                }),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "calculate".to_string(),
                description: "执行数学计算，支持加减乘除、幂运算等".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，例如：'3^8 + sqrt(256)'"
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": false
                }),
            },
        },
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_current_time".to_string(),
                description: "获取当前日期和时间".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "时区，默认 Asia/Shanghai"
                        }
                    },
                    "additionalProperties": false
                }),
            },
        },
    ]
}

fn execute_tool(name: &str, arguments: &str) -> Result<String> {
    match name {
        "get_weather" => {
            let args: serde_json::Value = serde_json::from_str(arguments)?;
            let city = args["city"].as_str().unwrap_or("未知");
            let data = match city {
                "北京" => serde_json::json!({"city":"北京","temperature":22,"weather":"多云"}),
                "上海" => serde_json::json!({"city":"上海","temperature":28,"weather":"小雨"}),
                "深圳" => serde_json::json!({"city":"深圳","temperature":32,"weather":"晴"}),
                _ => serde_json::json!({"city":city,"temperature":20,"weather":"未知"}),
            };
            Ok(data.to_string())
        }
        "calculate" => {
            let args: serde_json::Value = serde_json::from_str(arguments)?;
            let expr = args["expression"].as_str().unwrap_or("0");
            // 模拟计算（演示用）
            let result = format!("计算结果: eval({})", expr);
            Ok(serde_json::json!({"expression": expr, "result": result}).to_string())
        }
        "get_current_time" => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs();
            let ts = now + 8 * 3600; // UTC+8
            let h = (ts % 86400) / 3600;
            let m = (ts % 3600) / 60;
            let s = ts % 60;
            Ok(serde_json::json!({
                "datetime": format!("2026-06-22 {:02}:{:02}:{:02}", h, m, s),
                "timezone": "Asia/Shanghai"
            })
            .to_string())
        }
        _ => Ok(serde_json::json!({"error": "未知工具"}).to_string()),
    }
}

// ═══════════════════════════════════════════
//  交互循环
// ═══════════════════════════════════════════

async fn run(
    api_key: &str,
    messages: &mut Vec<Message>,
    tools: &[Tool],
    use_stream: bool,
    max_rounds: usize,
) -> Result<String> {
    for _ in 0..max_rounds {
        let resp = if use_stream {
            chat_stream(api_key, messages.clone(), Some(tools.to_vec())).await?
        } else {
            chat(api_key, messages.clone(), Some(tools.to_vec())).await?
        };

        let choice = &resp.choices[0];

        match choice.finish_reason.as_deref() {
            Some("tool_calls") => {
                let tool_calls = choice.message.tool_calls.as_ref().unwrap();
                messages.push(choice.message.clone());

                for tc in tool_calls {
                    println!("🔧 调用 {} → {}", tc.function.name, tc.function.arguments);
                    let result = execute_tool(&tc.function.name, &tc.function.arguments)?;
                    println!("   结果: {}", result);

                    messages.push(Message {
                        role: "tool".into(),
                        content: Some(result),
                        tool_calls: None,
                        tool_call_id: Some(tc.id.clone()),
                    });
                }
                continue;
            }
            _ => {
                return Ok(choice.message.content.clone().unwrap_or_default());
            }
        }
    }
    anyhow::bail!("达到最大轮次")
}

// ═══════════════════════════════════════════
//  main
// ═══════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置环境变量 DEEPSEEK_API_KEY");

    let tools = create_tools();
    let system = Message {
        role: "system".into(),
        content: Some(
            "你是一个全能的 AI 助手，可以查天气、算数学、查时间。需要工具时主动调用。".into(),
        ),
        tool_calls: None,
        tool_call_id: None,
    };

    let queries = vec![
        "3的8次方加上256的平方根是多少？",
        "深圳现在天气怎么样？适合出去玩吗？",
        "Rust 的生命周期是什么？",
    ];

    for q in queries {
        println!("\n════════════════════════════════");
        println!("🧑 用户：{}", q);
        let mut messages = vec![
            system.clone(),
            Message {
                role: "user".into(),
                content: Some(q.to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
        ];
        let reply = run(&api_key, &mut messages, &tools, false, 5).await?;
        println!("🤖 助手：{}\n", reply);
    }

    Ok(())
}
