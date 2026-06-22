use serde::{Deserialize, Serialize};

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
