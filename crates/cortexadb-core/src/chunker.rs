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
    if text.is_empty() || chunk_size == 0 {
        return vec![];
    }

    // overlap >= chunk_size → step would be 0 → infinite loop guard
    if overlap >= chunk_size {
        return vec![];
    }

    // Tokenize by whitespace upfront so we never cut a word in half.
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return vec![];
    }

    let mut chunks: Vec<ChunkResult> = Vec::new();
    // `word_pos` = index into `words` where the current chunk window starts.
    let mut word_pos: usize = 0;

    while word_pos < words.len() {
        // Greedily pack whole words until we exceed chunk_size.
        let mut chunk_words: Vec<&str> = Vec::new();
        let mut char_count: usize = 0;
        let mut w = word_pos;

        while w < words.len() {
            let word = words[w];
            // First word has no leading space; all others need one.
            let add_cost = if chunk_words.is_empty() { word.len() } else { word.len() + 1 };

            if char_count + add_cost <= chunk_size || chunk_words.is_empty() {
                // Always include at least one word even if it alone exceeds chunk_size,
                // so long words are never silently dropped.
                chunk_words.push(word);
                char_count += add_cost;
                w += 1;
            } else {
                break;
            }
        }

        let chunk_text = chunk_words.join(" ");
        if !chunk_text.is_empty() {
            chunks.push(ChunkResult { text: chunk_text, index: chunks.len(), metadata: None });
        }

        // No unseen words remain, so a trailing overlap-only chunk would be redundant.
        if w >= words.len() {
            break;
        }

        // Compute how many words from the tail of this chunk should be repeated
        // at the start of the next chunk (overlap, measured in chars).
        let overlap_words = {
            let mut acc = 0usize;
            let mut count = 0usize;
            for word in chunk_words.iter().rev() {
                let cost = if count == 0 { word.len() } else { word.len() + 1 };
                if acc + cost <= overlap {
                    acc += cost;
                    count += 1;
                } else {
                    break;
                }
            }
            count
        };

        let advance = chunk_words.len().saturating_sub(overlap_words);
        // Safety: always advance by at least 1 to prevent infinite loop.
        word_pos += advance.max(1);
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
        for sep in separators.iter() {
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
            // Fall back to fixed chunking when no separator works.
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
                current_section.push('\n');
            }
            current_section.push_str(trimmed);
        } else if !trimmed.starts_with("```") {
            current_section.push(' ');
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
                combined.push(' ');
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

    // -------------------------------------------------------------------------
    // chunk_fixed
    // -------------------------------------------------------------------------

    #[test]
    fn test_fixed_empty_input() {
        assert!(chunk_fixed("", 10, 0).is_empty());
    }

    #[test]
    fn test_fixed_whitespace_only_input() {
        assert!(chunk_fixed("   \n\t  ", 10, 0).is_empty());
    }

    #[test]
    fn test_fixed_zero_chunk_size_returns_empty() {
        assert!(chunk_fixed("hello world", 0, 0).is_empty());
    }

    #[test]
    fn test_fixed_overlap_ge_chunk_size_returns_empty() {
        // step = chunk_size - overlap = 5 - 10 = saturating 0 → empty
        assert!(chunk_fixed("some text here", 5, 10).is_empty());
    }

    #[test]
    fn test_fixed_text_shorter_than_chunk_size_gives_one_chunk() {
        let chunks = chunk_fixed("Short", 100, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short");
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn test_fixed_short_text_with_overlap_still_one_chunk() {
        let chunks = chunk_fixed("Short sentence.", 512, 50);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short sentence.");
    }

    #[test]
    fn test_fixed_no_overlap_indices_are_sequential() {
        let text = "word1 word2 word3 word4 word5 word6";
        let chunks = chunk_fixed(text, 10, 0);
        assert!(!chunks.is_empty());
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i, "chunk index must be sequential");
        }
    }

    #[test]
    fn test_fixed_no_whole_word_is_cut() {
        // Every token in output must appear verbatim in the original word list.
        let text = "Hello world foo bar";
        let chunks = chunk_fixed(text, 8, 0);
        let original_words: Vec<&str> = text.split_whitespace().collect();
        for c in &chunks {
            for word in c.text.split_whitespace() {
                assert!(
                    original_words.contains(&word),
                    "unexpected token '{}' — word-boundary snap failed",
                    word
                );
            }
        }
    }

    #[test]
    fn test_fixed_all_content_is_present() {
        // All words from the original text should appear in some chunk.
        let text = "alpha beta gamma delta epsilon zeta";
        let chunks = chunk_fixed(text, 12, 0);
        let combined: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        for word in text.split_whitespace() {
            assert!(combined.contains(word), "word '{}' missing from chunked output", word);
        }
    }

    #[test]
    fn test_fixed_with_overlap_adjacent_chunks_share_words() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_fixed(text, 15, 5);
        if chunks.len() >= 2 {
            let prev_words: Vec<&str> = chunks[0].text.split_whitespace().collect();
            let next_words: Vec<&str> = chunks[1].text.split_whitespace().collect();
            let shared = prev_words.iter().any(|w| next_words.contains(w));
            assert!(shared, "adjacent chunks should share words when overlap > 0");
        }
    }

    #[test]
    fn test_fixed_long_word_stays_intact() {
        // A word too long to fit in chunk_size must not be split.
        let text = "x superlongwordthatexceedschunksize y";
        let chunks = chunk_fixed(text, 5, 0);
        let combined: String = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        assert!(combined.contains("superlongwordthatexceedschunksize"));
    }

    // -------------------------------------------------------------------------
    // chunk_recursive
    // -------------------------------------------------------------------------

    #[test]
    fn test_recursive_empty_input() {
        assert!(chunk_recursive("", 512, 0).is_empty());
    }

    #[test]
    fn test_recursive_short_text_gives_one_chunk() {
        let text = "Just one short sentence.";
        let chunks = chunk_recursive(text, 512, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
        assert_eq!(chunks[0].index, 0);
    }

    #[test]
    fn test_recursive_splits_paragraphs_on_double_newline() {
        // Three paragraphs, chunk_size=25 forces a split.
        let text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.";
        let chunks = chunk_recursive(text, 25, 0);
        assert!(chunks.len() >= 2, "expected ≥ 2 chunks, got {}", chunks.len());
        for c in &chunks {
            assert!(!c.text.trim().is_empty(), "all chunks must be non-empty");
        }
    }

    #[test]
    fn test_recursive_indices_sequential() {
        let text = "a b\n\nc d\n\ne f\n\ng h";
        let chunks = chunk_recursive(text, 5, 0);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i);
        }
    }

    #[test]
    fn test_recursive_overlap_prepends_prev_words() {
        // With overlap=2, the second chunk should start with the last 2 words of the first.
        let text = "alpha beta gamma delta epsilon zeta eta theta";
        let chunks = chunk_recursive(text, 20, 2);
        if chunks.len() >= 2 {
            let first_words: Vec<&str> = chunks[0].text.split_whitespace().collect();
            let tail: Vec<&str> = first_words.iter().rev().take(2).copied().rev().collect();
            let second_text = &chunks[1].text;
            for w in &tail {
                assert!(
                    second_text.contains(w),
                    "expected overlap word '{}' in second chunk, got: '{}'",
                    w,
                    second_text
                );
            }
        }
    }

    /// Regression: chunk_recursive previously called chunk_fixed twice and discarded
    /// the first result (dead code on line 174). The output must be deterministic.
    #[test]
    fn test_recursive_is_deterministic_regression() {
        let text = "x".repeat(300); // forces fixed fallback
        let a = chunk_recursive(&text, 50, 0);
        let b = chunk_recursive(&text, 50, 0);
        assert_eq!(a.len(), b.len(), "chunk_recursive must be deterministic");
        for (ca, cb) in a.iter().zip(b.iter()) {
            assert_eq!(ca.text, cb.text);
        }
    }

    // -------------------------------------------------------------------------
    // chunk_semantic
    // -------------------------------------------------------------------------

    #[test]
    fn test_semantic_empty_input() {
        assert!(chunk_semantic("", 0).is_empty());
    }

    #[test]
    fn test_semantic_single_paragraph_gives_one_chunk() {
        let text = "Only one paragraph here.";
        let chunks = chunk_semantic(text, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, text);
    }

    #[test]
    fn test_semantic_all_text_preserved() {
        let text = "Para one.\n\nPara two.\n\nPara three.";
        let chunks = chunk_semantic(text, 0);
        assert!(!chunks.is_empty());
        let combined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        assert!(combined.contains("Para one"));
        assert!(combined.contains("Para two"));
        assert!(combined.contains("Para three"));
    }

    #[test]
    fn test_semantic_large_paragraphs_split_at_2048_boundary() {
        // Two paragraphs whose combined length > 2048 must be split.
        let big_a = "A".repeat(1200);
        let big_b = "B".repeat(1200);
        let text = format!("{}\n\n{}", big_a, big_b);
        let chunks = chunk_semantic(&text, 0);
        assert!(chunks.len() >= 2, "paragraphs totalling >2048 chars should split");
        let combined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("");
        assert!(combined.contains(&big_a));
        assert!(combined.contains(&big_b));
    }

    #[test]
    fn test_semantic_whitespace_only_paragraphs_are_skipped() {
        let text = "Real content.\n\n   \n\n  \n\nMore content.";
        let chunks = chunk_semantic(text, 0);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(!c.text.trim().is_empty());
        }
    }

    #[test]
    fn test_semantic_indices_sequential() {
        let big_a = "A".repeat(1300);
        let big_b = "B".repeat(1300);
        let big_c = "C".repeat(1300);
        let text = format!("{}\n\n{}\n\n{}", big_a, big_b, big_c);
        let chunks = chunk_semantic(&text, 0);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i);
        }
    }

    // -------------------------------------------------------------------------
    // chunk_markdown
    // -------------------------------------------------------------------------

    #[test]
    fn test_markdown_empty_input() {
        assert!(chunk_markdown("", true, 0).is_empty());
    }

    #[test]
    fn test_markdown_splits_on_headers() {
        let text = "# Section A\n\nContent A.\n\n## Section B\n\nContent B.";
        let chunks = chunk_markdown(text, true, 0);
        assert!(chunks.len() >= 2, "each header section should produce a chunk");
    }

    #[test]
    fn test_markdown_preserve_headers_includes_header_text() {
        let text = "# My Title\n\nSome body text.";
        let chunks = chunk_markdown(text, true, 0);
        assert!(!chunks.is_empty());
        let combined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        assert!(combined.contains("My Title"), "header must appear in output");
        assert!(combined.contains("Some body text"));
    }

    #[test]
    fn test_markdown_body_text_present_regardless_of_preserve() {
        let text = "# Header\n\nBody content here.";
        for preserve in [true, false] {
            let chunks = chunk_markdown(text, preserve, 0);
            assert!(!chunks.is_empty());
            let combined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
            assert!(
                combined.contains("Body content"),
                "body text must be present (preserve={})",
                preserve
            );
        }
    }

    #[test]
    fn test_markdown_code_blocks_replaced_not_raw() {
        let text = "Intro.\n\n```rust\nfn main() {}\n```\n\nConclusion.";
        let chunks = chunk_markdown(text, false, 0);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(!c.text.contains("```"), "raw code fences must be replaced, got: '{}'", c.text);
        }
    }

    #[test]
    fn test_markdown_indices_sequential() {
        let text = "# A\n\ntext a.\n\n## B\n\ntext b.\n\n### C\n\ntext c.";
        let chunks = chunk_markdown(text, true, 0);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i);
        }
    }

    // -------------------------------------------------------------------------
    // chunk_json
    // -------------------------------------------------------------------------

    #[test]
    fn test_json_empty_input() {
        assert!(chunk_json("", 0).is_empty());
    }

    #[test]
    fn test_json_invalid_falls_back_to_raw_chunk() {
        let text = "not valid json";
        let chunks = chunk_json(text, 0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "not valid json");
        assert!(chunks[0].metadata.is_none());
    }

    #[test]
    fn test_json_flat_object_leaf_count() {
        // {"name": "Alice", "age": 30} → 2 leaf values
        let text = r#"{"name": "Alice", "age": 30}"#;
        let chunks = chunk_json(text, 0);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_json_nested_uses_dotted_key_notation() {
        let text = r#"{"user": {"name": "John", "age": 30}}"#;
        let chunks = chunk_json(text, 0);
        assert!(!chunks.is_empty());
        let has_dotted = chunks.iter().any(|c| {
            c.metadata
                .as_ref()
                .and_then(|m| m.key.as_deref())
                .map(|k| k == "user.name")
                .unwrap_or(false)
        });
        assert!(has_dotted, "nested keys must use dot notation 'user.name'");
    }

    #[test]
    fn test_json_array_uses_bracket_key_notation() {
        let text = r#"{"items": ["alpha", "beta"]}"#;
        let chunks = chunk_json(text, 0);
        assert_eq!(chunks.len(), 2);
        let keys: Vec<&str> = chunks
            .iter()
            .filter_map(|c| c.metadata.as_ref())
            .filter_map(|m| m.key.as_deref())
            .collect();
        assert!(keys.contains(&"items[0]"), "array must use bracket notation, got: {:?}", keys);
        assert!(keys.contains(&"items[1]"));
    }

    #[test]
    fn test_json_leaf_values_preserved() {
        let text = r#"{"lang": "Rust", "count": 42, "flag": true}"#;
        let chunks = chunk_json(text, 0);
        let values: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert!(values.contains(&"Rust"));
        assert!(values.contains(&"42"));
        assert!(values.contains(&"true"));
    }

    #[test]
    fn test_json_metadata_key_and_value_fields_populated() {
        let text = r#"{"x": "hello"}"#;
        let chunks = chunk_json(text, 0);
        assert_eq!(chunks.len(), 1);
        let meta = chunks[0].metadata.as_ref().expect("metadata must be present for JSON chunks");
        assert_eq!(meta.key.as_deref(), Some("x"));
        assert_eq!(meta.value.as_deref(), Some("hello"));
    }

    // -------------------------------------------------------------------------
    // apply_overlap
    // -------------------------------------------------------------------------

    #[test]
    fn test_apply_overlap_zero_leaves_chunks_unchanged() {
        let input = vec![
            ChunkResult { text: "hello".to_string(), index: 0, metadata: None },
            ChunkResult { text: "world".to_string(), index: 1, metadata: None },
        ];
        let out = apply_overlap(input, 0);
        assert_eq!(out[0].text, "hello");
        assert_eq!(out[1].text, "world");
    }

    #[test]
    fn test_apply_overlap_one_word_prepended() {
        let input = vec![
            ChunkResult { text: "alpha beta gamma".to_string(), index: 0, metadata: None },
            ChunkResult { text: "delta".to_string(), index: 1, metadata: None },
        ];
        // overlap=1 → last word of chunk[0] ("gamma") prepended to chunk[1]
        let out = apply_overlap(input, 1);
        assert_eq!(out[0].text, "alpha beta gamma");
        assert!(out[1].text.starts_with("gamma "), "got: '{}'", out[1].text);
        assert!(out[1].text.contains("delta"));
    }

    #[test]
    fn test_apply_overlap_two_words_prepended() {
        let input = vec![
            ChunkResult { text: "The quick brown fox".to_string(), index: 0, metadata: None },
            ChunkResult { text: "jumps over".to_string(), index: 1, metadata: None },
        ];
        let out = apply_overlap(input, 2);
        assert_eq!(out[0].text, "The quick brown fox");
        assert!(out[1].text.starts_with("brown fox "), "got: '{}'", out[1].text);
        assert!(out[1].text.contains("jumps over"));
    }

    #[test]
    fn test_apply_overlap_single_chunk_unchanged() {
        let input = vec![ChunkResult { text: "only one".to_string(), index: 0, metadata: None }];
        let out = apply_overlap(input, 5);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "only one");
    }

    #[test]
    fn test_apply_overlap_exceeds_prev_len_takes_all_prev_words() {
        // overlap=5 but prev chunk only has 1 word → the one word is still prepended
        let input = vec![
            ChunkResult { text: "one".to_string(), index: 0, metadata: None },
            ChunkResult { text: "two three".to_string(), index: 1, metadata: None },
        ];
        let out = apply_overlap(input, 5);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "one");
        // "one" is the last (and only) word from chunk[0], so it is prepended to chunk[1]
        assert!(out[1].text.starts_with("one "), "got '{}'", out[1].text);
        assert!(out[1].text.contains("two three"));
    }

    #[test]
    fn test_chunk_recursive_strategy() {
        // chunk_size=30: "Paragraph one." fits, "Paragraph two. Sentence A. Sentence B." =42 chars
        // → 42>30 forces sentence-level split. "Paragraph three." fits.
        let text = "Paragraph one.\n\nParagraph two. Sentence A. Sentence B.\n\nParagraph three.";
        let strategy = ChunkingStrategy::Recursive { chunk_size: 30, overlap: 0 };
        let chunks = chunk(text, strategy);
        // At least 3 chunks (para1, para2-part, para3); content matters more than count.
        assert!(chunks.len() >= 3, "expected >=3 chunks, got {}", chunks.len());
        let combined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        assert!(combined.contains("Paragraph one"));
        assert!(combined.contains("Paragraph three"));
    }

    #[test]
    fn test_chunk_semantic_strategy() {
        // Three tiny paragraphs (<2048 total) — semantic may merge them or keep separate.
        // What must hold: all text is present in the output.
        let text = "First para.\n\nSecond para.\n\nThird para.";
        let strategy = ChunkingStrategy::Semantic { overlap: 0 };
        let chunks = chunk(text, strategy);
        assert!(!chunks.is_empty());
        let combined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        assert!(combined.contains("First para"));
        assert!(combined.contains("Second para"));
        assert!(combined.contains("Third para"));
    }

    #[test]
    fn test_chunk_markdown_strategy() {
        let text = "# Title\nContent.\n\n## Subtitle\nMore content.";
        let strategy = ChunkingStrategy::Markdown { preserve_headers: true, overlap: 0 };
        let chunks = chunk(text, strategy);
        assert!(!chunks.is_empty());
        let combined = chunks.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        assert!(combined.contains("Title"), "header text must be in output");
        assert!(combined.contains("Content"));
        assert!(combined.contains("Subtitle"));
        assert!(combined.contains("More content"));
    }

    #[test]
    fn test_chunk_json_strategy() {
        let text = r#"{"data": {"id": 123, "name": "Test"}}"#;
        let strategy = ChunkingStrategy::Json { overlap: 0 };
        let chunks = chunk(text, strategy);
        assert_eq!(chunks.len(), 2);
        assert!(chunks.iter().any(|c| c
            .metadata
            .as_ref()
            .map(|m| m.key.as_ref().unwrap() == "data.id" && m.value.as_ref().unwrap() == "123")
            .unwrap_or(false)));
        assert!(chunks.iter().any(|c| c
            .metadata
            .as_ref()
            .map(|m| m.key.as_ref().unwrap() == "data.name" && m.value.as_ref().unwrap() == "Test")
            .unwrap_or(false)));
    }
}
