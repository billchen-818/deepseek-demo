use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    /// 普通文本内容，tool role 时可以省略
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// 模型返回的工具调用列表（assistant role）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// 工具执行结果对应的调用 ID（tool role）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// 模型返回的工具调用
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ToolCall {
    /// 本次调用的唯一 ID，后续回传结果时要对上
    pub id: String,
    /// 固定值 "function"
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FunctionCall {
    /// 要调用的函数名
    pub name: String,
    /// 模型生成的参数，是一个 JSON 字符串，需要自己解析
    pub arguments: String,
}

/// 一个工具 = 一个函数定义
#[derive(Debug, Serialize, Clone)]
pub struct Tool {
    /// 固定值 "function"
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDef,
}

#[derive(Debug, Serialize, Clone)]
pub struct FunctionDef {
    /// 函数名，模型用这个名字来"调用"
    pub name: String,
    /// 函数描述，告诉模型什么时候该用这个工具
    pub description: String,
    /// 参数定义，JSON Schema 格式
    pub parameters: serde_json::Value,
}

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
    /// 关键：把工具列表传给 API
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    pub message: Message,
    /// finish_reason 可以判断模型是想"说话"还是"调工具"
    #[serde(default)]
    pub finish_reason: Option<String>,
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

/// 创建一组实用工具
fn create_tools() -> Vec<Tool> {
    vec![
        // 工具 1：天气查询
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_weather".to_string(),
                description: "查询指定城市的实时天气。当用户询问天气、气温、是否下雨等问题时使用此工具。".to_string(),
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
        // 工具 2：计算器
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "calculate".to_string(),
                description: "执行数学计算。当用户需要进行算术运算、求值等时使用此工具。支持加减乘除、幂运算、三角函数等。".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，例如：'2 + 3 * 4'、'sqrt(16)'、'sin(30 * pi / 180)'"
                        }
                    },
                    "required": ["expression"],
                    "additionalProperties": false
                }),
            },
        },
        // 工具 3：获取当前时间
        Tool {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: "get_current_time".to_string(),
                description: "获取当前的日期和时间。当用户询问'现在几点'、'今天几号'、'今天星期几'等问题时使用。".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "时区，例如：Asia/Shanghai。默认为 Asia/Shanghai"
                        }
                    },
                    "additionalProperties": false
                }),
            },
        },
    ]
}

/// 工具调度中心：根据函数名派发执行
fn execute_tool(name: &str, arguments: &str) -> Result<String> {
    match name {
        "get_weather" => {
            let args: serde_json::Value = serde_json::from_str(arguments)?;
            let city = args["city"].as_str().unwrap_or("未知城市");

            // 模拟天气数据（实际项目替换为真实 API 调用）
            let weather_data = match city {
                "北京" => {
                    serde_json::json!({"city":"北京","temperature":22.0,"weather":"多云","humidity":"55%"})
                }
                "上海" => {
                    serde_json::json!({"city":"上海","temperature":28.0,"weather":"小雨","humidity":"80%"})
                }
                "深圳" => {
                    serde_json::json!({"city":"深圳","temperature":32.0,"weather":"晴","humidity":"70%"})
                }
                _ => {
                    serde_json::json!({"city":city,"temperature":20.0,"weather":"未知","humidity":"50%"})
                }
            };
            Ok(weather_data.to_string())
        }

        "calculate" => {
            let args: serde_json::Value = serde_json::from_str(arguments)?;
            let expr = args["expression"].as_str().unwrap_or("0");

            // 注意：生产环境不要用系统 shell 执行用户输入，这里仅是示例
            // 更好的做法是用 meval 或其他表达式求值库
            let output = if cfg!(target_os = "windows") {
                Command::new("cmd").args(["/C", "echo", expr]).output()
            } else {
                Command::new("bc")
                    .arg(format!("scale=4; {}", expr))
                    .output()
            };

            match output {
                Ok(o) if o.status.success() => {
                    let result = String::from_utf8_lossy(&o.stdout).trim().to_string();
                    let result = if result.is_empty() {
                        format!("无法计算表达式: {}", expr)
                    } else {
                        result
                    };
                    Ok(serde_json::json!({"expression": expr, "result": result}).to_string())
                }
                _ => {
                    // 降级：手动模拟几个常见表达式
                    let result = simple_eval(expr);
                    Ok(serde_json::json!({"expression": expr, "result": result}).to_string())
                }
            }
        }

        "get_current_time" => {
            // Rust 标准库没有直接获取时区时间的方法
            // 这里用一个简化的实现
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let secs = now.as_secs();

            // 转成 UTC+8 时间
            let total_secs = secs + 8 * 3600;
            let days = total_secs / 86400;
            let time_of_day = total_secs % 86400;
            let hours = time_of_day / 3600;
            let minutes = (time_of_day % 3600) / 60;
            let seconds = time_of_day % 60;

            // 计算年月日（简化版）
            let (year, month, day) = civil_from_days(days as i64);

            // If `days` is Unix days (0 = 1970-01-01, a Thursday):
            // (0 + 4) % 7 = 4 -> "四" (Thursday)
            let weekday_idx = ((days % 7 + 7) % 7 + 4) % 7;
            let weekday = ["日", "一", "二", "三", "四", "五", "六"][weekday_idx as usize];

            Ok(serde_json::json!({
                "datetime": format!("{}-{:02}-{:02} {:02}:{:02}:{:02}", year, month, day, hours, minutes, seconds),
                "timezone": "Asia/Shanghai",
                "day_of_week": weekday
            }).to_string())
        }

        _ => Ok(serde_json::json!({"error": format!("未知工具: {}", name)}).to_string()),
    }
}

/// 简化版表达式求值
fn simple_eval(expr: &str) -> f64 {
    // 这里仅做演示，实际项目推荐用 meval crate
    let expr = expr.replace("pi", "3.141592653589793");
    let expr = expr.replace("e", "2.718281828459045");
    let expr = expr.trim();

    // 尝试简单的 a + b 模式
    let parts: Vec<&str> = expr.split('+').collect();
    if parts.len() == 2 {
        let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
        let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
        return a + b;
    }
    let parts: Vec<&str> = expr.split('*').collect();
    if parts.len() == 2 {
        let a: f64 = parts[0].trim().parse().unwrap_or(0.0);
        let b: f64 = parts[1].trim().parse().unwrap_or(0.0);
        return a * b;
    }
    expr.parse().unwrap_or(0.0)
}

// 简化的日期转换（儒略日 → 公历）
fn civil_from_days(jd: i64) -> (i64, u32, u32) {
    // 1970-01-01 的儒略日约为 2440588
    let z = jd + 2440588;
    let f = z + 1401 + (((4 * z + 274277) / 146097) * 3) / 4 - 38;
    let e = 4 * f + 3;
    let g = (e % 1461) / 4;
    let h = 5 * g + 2;
    let day = ((h % 153) / 5 + 1) as u32;
    let month = ((h / 153 + 2) % 12 + 1) as u32;
    let year = (e / 1461) - 4716 + ((14 - month as i64) / 12);
    (year, month, day)
}

// 完整的 Tool Use 交互循环
/// 支持多轮工具调用（模型一次可能调多个工具）
pub async fn run_conversation(
    api_key: &str,
    messages: &mut Vec<Message>,
    tools: &[Tool],
    max_rounds: usize,
) -> Result<String> {
    for _round in 0..max_rounds {
        let resp = chat_with_tools(api_key, messages.clone(), Some(tools.to_vec())).await?;
        let choice = &resp.choices[0];

        match choice.finish_reason.as_deref() {
            Some("tool_calls") => {
                let tool_calls = choice.message.tool_calls.as_ref().unwrap();

                // 把 assistant 的 tool_calls 消息加入历史
                messages.push(choice.message.clone());

                // 执行每个工具
                for tc in tool_calls {
                    println!(
                        "🔧 调用工具: {} → {}",
                        tc.function.name, tc.function.arguments
                    );
                    let result = execute_tool(&tc.function.name, &tc.function.arguments)?;
                    println!("   结果: {}", result);

                    messages.push(Message {
                        role: "tool".into(),
                        content: Some(result),
                        tool_calls: None,
                        tool_call_id: Some(tc.id.clone()),
                    });
                }

                // 继续循环，让模型处理工具结果
                continue;
            }

            Some("stop") | None => {
                return Ok(choice
                    .message
                    .content
                    .clone()
                    .unwrap_or_else(|| "（模型无响应）".to_string()));
            }

            Some(reason) => {
                anyhow::bail!("意外的 finish_reason: {}", reason);
            }
        }
    }

    anyhow::bail!("达到最大轮次限制 ({})，对话未结束", max_rounds)
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置环境变量 DEEPSEEK_API_KEY");

    let tools = create_tools();

    // 测试 1：纯计算问题
    {
        let mut messages = vec![Message {
            role: "system".into(),
            content: Some("你是一个全能的 AI 助手，可以查天气、算数学、查时间。遇到需要工具的问题请主动调用。".into()),
            tool_calls: None,
            tool_call_id: None,
        }];

        println!("═══════════════════════════════════════");
        println!("用户：3的8次方加上 256 的平方根是多少？");
        messages.push(Message {
            role: "user".into(),
            content: Some("3的8次方加上 256 的平方根是多少？".into()),
            tool_calls: None,
            tool_call_id: None,
        });

        let reply = run_conversation(&api_key, &mut messages, &tools, 5).await?;
        println!("\n模型最终回复：\n{}\n", reply);
    }

    // 测试 2：需要同时用天气 + 时间
    {
        let mut messages = vec![Message {
            role: "system".into(),
            content: Some("你是一个全能的 AI 助手。".into()),
            tool_calls: None,
            tool_call_id: None,
        }];

        println!("═══════════════════════════════════════");
        println!("用户：深圳现在几点了？天气怎么样？适合跑步吗？");
        messages.push(Message {
            role: "user".into(),
            content: Some("深圳现在几点了？天气怎么样？适合跑步吗？".into()),
            tool_calls: None,
            tool_call_id: None,
        });

        let reply = run_conversation(&api_key, &mut messages, &tools, 5).await?;
        println!("\n模型最终回复：\n{}\n", reply);
    }

    // 测试 3：不需要工具的问题
    {
        let mut messages = vec![Message {
            role: "system".into(),
            content: Some("你是一个全能的 AI 助手。".into()),
            tool_calls: None,
            tool_call_id: None,
        }];

        println!("═══════════════════════════════════════");
        println!("用户：什么是 Rust 的所有权机制？");
        messages.push(Message {
            role: "user".into(),
            content: Some("什么是 Rust 的所有权机制？".into()),
            tool_calls: None,
            tool_call_id: None,
        });

        let reply = run_conversation(&api_key, &mut messages, &tools, 5).await?;
        println!("\n模型最终回复：\n{}\n", reply);
    }

    Ok(())
}
