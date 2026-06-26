use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

const API_URL: &str = "https://api.deepseek.com/chat/completions";

#[derive(Debug, Clone)]
struct Document {
    id: String,
    title: String,
    content: String,
}

#[derive(Debug, Clone)]
struct Chunk {
    id: String,
    #[allow(unused)]
    doc_id: String,
    title: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    temperature: f64,
    max_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

fn msg(role: &str, content: impl Into<String>) -> Message {
    Message {
        role: role.into(),
        content: content.into(),
    }
}

fn load_documents(dir: &str) -> Result<Vec<Document>> {
    let mut docs = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }

        let content = fs::read_to_string(&path)?;
        let title = extract_title(&content).unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("untitled")
                .to_string()
        });

        let id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        docs.push(Document { id, title, content });
    }

    Ok(docs)
}

fn extract_title(content: &str) -> Option<String> {
    content
        .lines()
        .find_map(|line| line.strip_prefix("# ").map(|s| s.trim().to_string()))
}

fn chunk_documents(docs: &[Document]) -> Vec<Chunk> {
    let mut chunks = Vec::new();

    for doc in docs {
        let paragraphs: Vec<String> = doc
            .content
            .split("\n\n")
            .map(|p| p.trim().replace('\n', " "))
            .filter(|p| !p.is_empty())
            .collect();

        for (index, paragraph) in paragraphs.into_iter().enumerate() {
            let chunk_id = format!("{}#{}", doc.id, index);

            chunks.push(Chunk {
                id: chunk_id,
                doc_id: doc.id.clone(),
                title: doc.title.clone(),
                content: paragraph,
            });
        }
    }

    chunks
}

fn retrieve(question: &str, chunks: &[Chunk], top_k: usize) -> Vec<Chunk> {
    let query_terms = extract_terms(question);
    let mut scored = Vec::new();

    for chunk in chunks {
        let text = format!("{} {}", chunk.title, chunk.content);
        let score = score_text(&query_terms, &text);

        if score > 0 {
            scored.push((score, chunk.clone()));
        }
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored
        .into_iter()
        .take(top_k)
        .map(|(_, chunk)| chunk)
        .collect()
}

fn extract_terms(text: &str) -> HashSet<char> {
    let stop_chars: HashSet<char> =
        "的了么吗呢什么什么时候可以能不能用户企业个人一个以及和或在后内每天多少"
            .chars()
            .collect();

    text.chars()
        .filter(|c| c.is_alphanumeric() || ('\u{4e00}'..='\u{9fff}').contains(c))
        .filter(|c| !stop_chars.contains(c))
        .collect()
}

fn score_text(query_terms: &HashSet<char>, text: &str) -> usize {
    text.chars().filter(|c| query_terms.contains(c)).count()
}

fn build_context(chunks: &[Chunk]) -> String {
    chunks
        .iter()
        .map(|chunk| {
            format!(
                "[{}]\n文档：{}\n内容：{}\n",
                chunk.id, chunk.title, chunk.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn build_messages(question: &str, chunks: &[Chunk]) -> Vec<Message> {
    let context = build_context(chunks);

    vec![
        msg(
            "system",
            r#"你是一个严谨的知识库问答助手。
你只能基于 <context> 中的资料回答。
如果资料不足以回答，必须回答："资料不足，无法判断。"
不要编造政策、数字、日期、链接、文件名或来源。
回答中每个关键结论都要引用来源，格式为 [来源: chunk-id]。"#,
        ),
        msg(
            "user",
            format!(
                "<context>\n{}\n</context>\n\n用户问题：{}",
                context, question
            ),
        ),
    ]
}

async fn chat(api_key: &str, messages: Vec<Message>) -> Result<String> {
    let client = reqwest::Client::new();

    let body = ChatRequest {
        model: "deepseek-v4-pro".to_string(),
        messages,
        stream: false,
        temperature: 0.1,
        max_tokens: 1024,
    };

    let resp = client
        .post(API_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await?;

    if !resp.status().is_success() {
        anyhow::bail!("API 返回错误 {}: {}", resp.status(), resp.text().await?);
    }

    let data: ChatResponse = resp.json().await?;
    data.choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow::anyhow!("API 返回了空的 choices"))
}

fn extract_sources(answer: &str) -> HashSet<String> {
    let mut sources = HashSet::new();
    let marker = "[来源: ";
    let mut rest = answer;

    while let Some(start) = rest.find(marker) {
        let after_marker = &rest[start + marker.len()..];
        let Some(end) = after_marker.find(']') else {
            break;
        };

        let source = after_marker[..end].trim().to_string();
        sources.insert(source);
        rest = &after_marker[end + 1..];
    }

    sources
}

fn validate_sources(answer: &str, chunks: &[Chunk]) -> Result<()> {
    let allowed: HashSet<&str> = chunks.iter().map(|chunk| chunk.id.as_str()).collect();
    let used = extract_sources(answer);

    if answer.contains("资料不足") {
        return Ok(());
    }

    if used.is_empty() {
        anyhow::bail!("模型回答没有引用来源，拒绝展示");
    }

    for source in used {
        if !allowed.contains(source.as_str()) {
            anyhow::bail!("模型引用了未提供的来源：{source}");
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("DEEPSEEK_API_KEY").expect("请设置环境变量 DEEPSEEK_API_KEY");

    let docs_dir = "fixtures";
    if !Path::new(docs_dir).exists() {
        anyhow::bail!("请先创建 fixtures 目录，并放入 Markdown 文档");
    }

    let question = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Pro 会员每天能用多少次 AI 问答？".to_string());

    // 1. 加载文档
    let docs = load_documents(docs_dir)?;
    println!("加载文档：{} 篇", docs.len());

    // 2. 文档切片
    let chunks = chunk_documents(&docs);
    println!("生成片段：{} 个", chunks.len());

    // 3. 检索相关片段
    let hits = retrieve(&question, &chunks, 3);
    println!("命中片段：");
    for chunk in &hits {
        println!("  - {} ({})", chunk.id, chunk.title);
    }

    if hits.is_empty() {
        println!("资料不足，无法判断。");
        return Ok(());
    }

    // 4. 构造 prompt
    let messages = build_messages(&question, &hits);

    // 5. 调模型
    let answer = chat(&api_key, messages).await?;

    // 6. 校验来源
    validate_sources(&answer, &hits)?;

    println!("\n问题：{question}");
    println!("回答：{answer}");

    Ok(())
}
