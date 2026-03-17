use std::sync::{OnceLock, RwLock};

#[derive(Debug, Clone)]
pub struct IntentPolicy {
    pub semantic_anchor_text: String,
    pub recency_anchor_text: String,
    pub graph_anchor_text: String,
    pub graph_hops_2_threshold: f32,
    pub graph_hops_3_threshold: f32,
    pub importance_pct: u8,
}

impl Default for IntentPolicy {
    fn default() -> Self {
        Self {
            semantic_anchor_text: "semantic meaning, similar content, concept match".to_string(),
            recency_anchor_text: "recent events, latest updates, time, timeline, schedule"
                .to_string(),
            graph_anchor_text: "relationships, connections, linked people, social graph"
                .to_string(),
            graph_hops_2_threshold: 0.55,
            graph_hops_3_threshold: 0.80,
            importance_pct: 20,
        }
    }
}

fn policy_cell() -> &'static RwLock<IntentPolicy> {
    static CELL: OnceLock<RwLock<IntentPolicy>> = OnceLock::new();
    CELL.get_or_init(|| RwLock::new(IntentPolicy::default()))
}

pub fn set_intent_policy(policy: IntentPolicy) {
    if let Ok(mut guard) = policy_cell().write() {
        *guard = policy;
        crate::query::executor::clear_intent_anchor_cache();
    } else {
        log::warn!("intent policy write lock poisoned");
    }
}

pub fn get_intent_policy() -> IntentPolicy {
    if let Ok(guard) = policy_cell().read() {
        guard.clone()
    } else {
        log::warn!("intent policy read lock poisoned, falling back to default");
        IntentPolicy::default()
    }
}
