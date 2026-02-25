use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub key: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResult {
    pub text: String,
    pub index: usize,
    pub metadata: Option<ChunkMetadata>,
}

#[derive(Debug, Clone)]
pub enum ChunkingStrategy {
    Fixed { chunk_size: usize, overlap: usize },
    Recursive { chunk_size: usize, overlap: usize },
    Semantic { overlap: usize },
    Markdown { preserve_headers: bool, overlap: usize },
    Json { overlap: usize },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        Self::Recursive { chunk_size: 512, overlap: 50 }
    }
}

pub fn chunk(text: &str, strategy: ChunkingStrategy) -> Vec<ChunkResult> {
    match strategy {
        ChunkingStrategy::Fixed { chunk_size, overlap } => chunk_fixed(text, chunk_size, overlap),
        ChunkingStrategy::Recursive { chunk_size, overlap } => {
            chunk_recursive(text, chunk_size, overlap)
        }
        ChunkingStrategy::Semantic { overlap } => chunk_semantic(text, overlap),
        ChunkingStrategy::Markdown { preserve_headers, overlap } => {
            chunk_markdown(text, preserve_headers, overlap)
        }
        ChunkingStrategy::Json { overlap } => chunk_json(text, overlap),
    }
}

fn chunk_fixed(text: &str, chunk_size: usize, overlap: usize) -> Vec<ChunkResult> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    if chunk_size == 0 {
        return vec![];
    }

    let step = chunk_size.saturating_sub(overlap);
    if step == 0 {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    let mut index = 0;

    while start < text.len() {
        let mut end = (start + chunk_size).min(text.len());

        if end < text.len() {
            if let Some(snap) = text[..end].rfind(' ') {
                if snap > start {
                    end = snap;
                }
            }
        }

        let chunk_text = text[start..end].trim().to_string();
        if !chunk_text.is_empty() {
            chunks.push(ChunkResult { text: chunk_text, index, metadata: None });
            index += 1;
        }

        start += step;
    }

    chunks
}

fn chunk_recursive(text: &str, chunk_size: usize, overlap: usize) -> Vec<ChunkResult> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let separators = ["\n\n\n", "\n\n", "\n", ".!?", ",;: ", " "];
    let mut chunks = Vec::new();
    let mut index = 0;

    fn split_recursive(
        text: &str,
        separators: &[&str],
        chunk_size: usize,
        overlap: usize,
        chunks: &mut Vec<ChunkResult>,
        index: &mut usize,
    ) {
        if text.len() <= chunk_size {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                chunks.push(ChunkResult {
                    text: trimmed.to_string(),
                    index: *index,
                    metadata: None,
                });
                *index += 1;
            }
            return;
        }

        let mut split_done = false;
        for (_i, sep) in separators.iter().enumerate() {
            if *sep == " " {
                continue;
            }

            if text.contains(sep) {
                let parts: Vec<&str> = text.split(sep).filter(|s| !s.trim().is_empty()).collect();

                if parts.len() > 1 {
                    let mut current = String::new();
                    for part in parts {
                        let test = if current.is_empty() {
                            part.to_string()
                        } else {
                            format!("{}{}{}", current, sep, part)
                        };

                        if test.len() > chunk_size {
                            if !current.is_empty() {
                                let trimmed = current.trim().to_string();
                                if !trimmed.is_empty() {
                                    chunks.push(ChunkResult {
                                        text: trimmed,
                                        index: *index,
                                        metadata: None,
                                    });
                                    *index += 1;
                                }
                            }
                            current = part.to_string();
                        } else {
                            current = test;
                        }
                    }

                    if !current.is_empty() {
                        let trimmed = current.trim().to_string();
                        if !trimmed.is_empty() {
                            chunks.push(ChunkResult {
                                text: trimmed,
                                index: *index,
                                metadata: None,
                            });
                            *index += 1;
                        }
                    }

                    split_done = true;
                    break;
                }
            }
        }

        if !split_done {
            chunk_fixed(text, chunk_size, overlap);
            let fixed_chunks = chunk_fixed(text, chunk_size, overlap);
            for chunk in fixed_chunks {
                chunks.push(ChunkResult { text: chunk.text, index: *index, metadata: None });
                *index += 1;
            }
        }
    }

    split_recursive(text, &separators, chunk_size, overlap, &mut chunks, &mut index);

    apply_overlap(chunks, overlap)
}

fn chunk_semantic(text: &str, overlap: usize) -> Vec<ChunkResult> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let paragraphs: Vec<&str> = text.split("\n\n").collect();
    let mut chunks = Vec::new();
    let mut index = 0;
    let mut current_chunk = String::new();

    for paragraph in paragraphs {
        let trimmed = paragraph.trim();
        if trimmed.is_empty() {
            continue;
        }

        if current_chunk.is_empty() {
            current_chunk = trimmed.to_string();
        } else if current_chunk.len() + trimmed.len() + 2 > 2048 {
            if !current_chunk.is_empty() {
                chunks.push(ChunkResult { text: current_chunk.clone(), index, metadata: None });
                index += 1;
            }
            current_chunk = trimmed.to_string();
        } else {
            current_chunk.push_str("\n\n");
            current_chunk.push_str(trimmed);
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(ChunkResult { text: current_chunk, index, metadata: None });
    }

    apply_overlap(chunks, overlap)
}

fn chunk_markdown(text: &str, preserve_headers: bool, overlap: usize) -> Vec<ChunkResult> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let header_regex = Regex::new(r"(?m)^(#{1,6}\s+.+)$").unwrap();
    let list_regex = Regex::new(r"(?m)^[\s]*[-*+]\s+").unwrap();
    let code_block_regex = Regex::new(r"(?s)```[\s\S]*?```").unwrap();

    let mut chunks = Vec::new();
    let mut index = 0;
    let mut current_section = String::new();
    let mut last_header = String::new();

    let processed =
        code_block_regex.replace_all(text, |_caps: &regex::Captures| " [CODE_BLOCK] ".to_string());

    for line in processed.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if header_regex.is_match(trimmed) {
            if !current_section.is_empty() {
                chunks.push(ChunkResult {
                    text: current_section.trim().to_string(),
                    index,
                    metadata: None,
                });
                index += 1;
            }

            if preserve_headers {
                last_header = trimmed.to_string();
                current_section = trimmed.to_string();
            } else {
                last_header = trimmed.to_string();
                current_section = String::new();
            }
        } else if list_regex.is_match(line) {
            if !last_header.is_empty() && !current_section.is_empty() {
                current_section.push_str("\n");
            }
            current_section.push_str(trimmed);
        } else if !trimmed.starts_with("```") {
            current_section.push_str(" ");
            current_section.push_str(trimmed);
        }

        if current_section.len() > 2048 {
            chunks.push(ChunkResult {
                text: current_section.trim().to_string(),
                index,
                metadata: None,
            });
            index += 1;

            if preserve_headers && !last_header.is_empty() {
                current_section = last_header.clone();
            } else {
                current_section = String::new();
            }
        }
    }

    if !current_section.is_empty() {
        chunks.push(ChunkResult {
            text: current_section.trim().to_string(),
            index,
            metadata: None,
        });
    }

    apply_overlap(chunks, overlap)
}

fn chunk_json(text: &str, _overlap: usize) -> Vec<ChunkResult> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let json: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(_) => {
            return vec![ChunkResult { text: text.to_string(), index: 0, metadata: None }];
        }
    };

    let mut chunks = Vec::new();
    let mut index = 0;

    fn flatten_json(
        value: &serde_json::Value,
        prefix: &str,
        chunks: &mut Vec<ChunkResult>,
        index: &mut usize,
    ) {
        match value {
            serde_json::Value::Object(map) => {
                for (key, val) in map {
                    let new_key =
                        if prefix.is_empty() { key.clone() } else { format!("{}.{}", prefix, key) };
                    flatten_json(val, &new_key, chunks, index);
                }
            }
            serde_json::Value::Array(arr) => {
                for (i, val) in arr.iter().enumerate() {
                    let new_key = format!("{}[{}]", prefix, i);
                    flatten_json(val, &new_key, chunks, index);
                }
            }
            _ => {
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Number(n) => n.to_string(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Null => "null".to_string(),
                    _ => value.to_string(),
                };

                let text_clone = value_str.clone();
                let key_clone = prefix.to_string();

                chunks.push(ChunkResult {
                    text: value_str,
                    index: *index,
                    metadata: Some(ChunkMetadata { key: Some(key_clone), value: Some(text_clone) }),
                });
                *index += 1;
            }
        }
    }

    flatten_json(&json, "", &mut chunks, &mut index);

    if chunks.is_empty() {
        chunks.push(ChunkResult { text: text.to_string(), index: 0, metadata: None });
    }

    chunks
}

fn apply_overlap(chunks: Vec<ChunkResult>, overlap: usize) -> Vec<ChunkResult> {
    if overlap == 0 || chunks.len() < 2 {
        return chunks;
    }

    let mut result = Vec::new();

    for (i, chunk) in chunks.into_iter().enumerate() {
        if i > 0 {
            let prev: &ChunkResult = result.last().unwrap();
            let prev_words: Vec<&str> = prev.text.split_whitespace().collect();
            let overlap_words: Vec<&str> =
                prev_words.iter().rev().take(overlap.min(prev_words.len())).copied().collect();

            let mut combined = String::new();
            for word in overlap_words.iter().rev() {
                if !combined.is_empty() {
                    combined.push(' ');
                }
                combined.push_str(word);
            }
            if !combined.is_empty() {
                combined.push_str(" ");
            }
            combined.push_str(&chunk.text);

            result.push(ChunkResult {
                text: combined,
                index: chunk.index,
                metadata: chunk.metadata,
            });
        } else {
            result.push(chunk);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_chunk_basic() {
        let text = "Hello world this is a test string for chunking.";
        let chunks = chunk_fixed(text, 10, 0);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_fixed_chunk_empty() {
        let chunks = chunk_fixed("", 10, 0);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_recursive_simple() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = chunk_recursive(text, 50, 0);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_semantic_paragraphs() {
        let text = "Para one.\n\nPara two.\n\nPara three.";
        let chunks = chunk_semantic(text, 0);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_headers() {
        let text = "# Header\n\nContent here.\n\n## Subheader\n\nMore content.";
        let chunks = chunk_markdown(text, true, 0);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_json_flatten() {
        let text = r#"{"user": {"name": "John", "age": 30}}"#;
        let chunks = chunk_json(text, 0);
        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|c| {
            c.metadata
                .as_ref()
                .map(|m| m.key.as_ref().map(|k| k.contains("user.name")).unwrap_or(false))
                .unwrap_or(false)
        }));
    }

    #[test]
    fn test_json_invalid() {
        let text = "not valid json";
        let chunks = chunk_json(text, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "not valid json");
    }
}
