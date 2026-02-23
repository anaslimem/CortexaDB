use std::net::SocketAddr;
use std::path::PathBuf;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use mnemos::engine::SyncPolicy;
use mnemos::index::vector::VectorBackendMode;
use mnemos::query::{IntentPolicy, set_intent_policy};
use mnemos::service::grpc::{
    AllowAllAuthProvider, ApiKeyAuthProvider, AuthProvider, MnemosGrpcService, MnemosServiceServer,
    QuotaPolicy, RbacPolicy,
};
use mnemos::store::CheckpointPolicy;
use mnemos::store::MnemosStore;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = std::env::var("MNEMOS_GRPC_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:50051".to_string())
        .parse()?;
    let status_addr: SocketAddr = std::env::var("MNEMOS_STATUS_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:50052".to_string())
        .parse()?;
    let data_dir = std::env::var("MNEMOS_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("mnemos_grpc"));
    let vector_dim: usize = std::env::var("MNEMOS_VECTOR_DIM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let sync_policy = parse_sync_policy_from_env();
    let checkpoint_policy = parse_checkpoint_policy_from_env();
    let intent_policy = parse_intent_policy_from_env();
    let auth_provider = parse_auth_provider_from_env();
    let rbac_policy = parse_rbac_policy_from_env();
    let quota_policy = parse_quota_policy_from_env();
    set_intent_policy(intent_policy.clone());

    std::fs::create_dir_all(&data_dir)?;
    let wal = data_dir.join("mnemos.wal");
    let seg = data_dir.join("segments");

    let store = if wal.exists() {
        MnemosStore::recover_with_policies(&wal, &seg, vector_dim, sync_policy, checkpoint_policy)?
    } else {
        MnemosStore::new_with_policies(&wal, &seg, vector_dim, sync_policy, checkpoint_policy)?
    };
    store.set_vector_backend_mode(parse_vector_backend_mode_from_env());

    let service = MnemosGrpcService::new_with_config(store, auth_provider, rbac_policy, quota_policy.clone());

    println!("Mnemos gRPC listening on {}", addr);
    println!("Mnemos status HTTP listening on {}", status_addr);
    println!("Data dir: {}", data_dir.display());
    println!("Vector dimension: {}", vector_dim);
    println!("Sync policy: {:?}", sync_policy);
    println!("Checkpoint policy: {:?}", checkpoint_policy);
    println!("Intent policy: {:?}", intent_policy);
    println!("Quota policy: {:?}", quota_policy);
    println!(
        "Auth mode: {}",
        std::env::var("MNEMOS_AUTH_MODE").unwrap_or_else(|_| "none".to_string())
    );

    let metrics_service = service.clone();
    tokio::spawn(async move {
        if let Err(e) = run_status_server(status_addr, metrics_service).await {
            eprintln!("status server error: {e}");
        }
    });

    log_tls_env_state();
    Server::builder()
        .add_service(MnemosServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}

fn parse_sync_policy_from_env() -> SyncPolicy {
    let mode = std::env::var("MNEMOS_SYNC_POLICY").unwrap_or_else(|_| "strict".to_string());
    match mode.to_ascii_lowercase().as_str() {
        "batch" => SyncPolicy::Batch {
            max_ops: std::env::var("MNEMOS_SYNC_BATCH_MAX_OPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64),
            max_delay_ms: std::env::var("MNEMOS_SYNC_BATCH_MAX_DELAY_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(25),
        },
        "async" => SyncPolicy::Async {
            interval_ms: std::env::var("MNEMOS_SYNC_ASYNC_INTERVAL_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(25),
        },
        _ => SyncPolicy::Strict,
    }
}

fn parse_checkpoint_policy_from_env() -> CheckpointPolicy {
    let enabled = std::env::var("MNEMOS_CHECKPOINT_ENABLED")
        .ok()
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false);
    if !enabled {
        return CheckpointPolicy::Disabled;
    }

    CheckpointPolicy::Periodic {
        every_ops: std::env::var("MNEMOS_CHECKPOINT_EVERY_OPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000),
        every_ms: std::env::var("MNEMOS_CHECKPOINT_EVERY_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30_000),
    }
}

fn parse_intent_policy_from_env() -> IntentPolicy {
    let mut policy = IntentPolicy::default();

    if let Ok(v) = std::env::var("MNEMOS_INTENT_ANCHOR_SEMANTIC") {
        let trimmed = v.trim();
        if !trimmed.is_empty() {
            policy.semantic_anchor_text = trimmed.to_string();
        }
    }
    if let Ok(v) = std::env::var("MNEMOS_INTENT_ANCHOR_RECENCY") {
        let trimmed = v.trim();
        if !trimmed.is_empty() {
            policy.recency_anchor_text = trimmed.to_string();
        }
    }
    if let Ok(v) = std::env::var("MNEMOS_INTENT_ANCHOR_GRAPH") {
        let trimmed = v.trim();
        if !trimmed.is_empty() {
            policy.graph_anchor_text = trimmed.to_string();
        }
    }

    if let Ok(v) = std::env::var("MNEMOS_INTENT_GRAPH_HOPS_2_THRESHOLD") {
        if let Ok(parsed) = v.parse::<f32>() {
            policy.graph_hops_2_threshold = parsed.clamp(0.0, 1.0);
        }
    }
    if let Ok(v) = std::env::var("MNEMOS_INTENT_GRAPH_HOPS_3_THRESHOLD") {
        if let Ok(parsed) = v.parse::<f32>() {
            policy.graph_hops_3_threshold = parsed.clamp(0.0, 1.0);
        }
    }
    if policy.graph_hops_2_threshold > policy.graph_hops_3_threshold {
        std::mem::swap(
            &mut policy.graph_hops_2_threshold,
            &mut policy.graph_hops_3_threshold,
        );
    }

    if let Ok(v) = std::env::var("MNEMOS_INTENT_IMPORTANCE_PCT") {
        if let Ok(parsed) = v.parse::<u8>() {
            policy.importance_pct = parsed.min(90);
        }
    }

    policy
}

fn parse_auth_provider_from_env() -> Arc<dyn AuthProvider> {
    let mode = std::env::var("MNEMOS_AUTH_MODE").unwrap_or_else(|_| "none".to_string());
    match mode.to_ascii_lowercase().as_str() {
        "api_key" => {
            let key = std::env::var("MNEMOS_API_KEY").unwrap_or_default();
            if key.trim().is_empty() {
                eprintln!(
                    "MNEMOS_AUTH_MODE=api_key but MNEMOS_API_KEY is empty; falling back to no auth"
                );
                Arc::new(AllowAllAuthProvider)
            } else {
                Arc::new(ApiKeyAuthProvider::new(key))
            }
        }
        _ => Arc::new(AllowAllAuthProvider),
    }
}

fn parse_vector_backend_mode_from_env() -> VectorBackendMode {
    let mode = std::env::var("MNEMOS_VECTOR_BACKEND").unwrap_or_else(|_| "exact".to_string());
    match mode.to_ascii_lowercase().as_str() {
        "ann" => VectorBackendMode::Ann {
            ann_search_multiplier: std::env::var("MNEMOS_VECTOR_ANN_SEARCH_MULTIPLIER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8),
        },
        _ => VectorBackendMode::Exact,
    }
}

fn parse_rbac_policy_from_env() -> RbacPolicy {
    let admins = parse_csv(std::env::var("MNEMOS_RBAC_ADMIN_PRINCIPALS").ok());
    let mut read_allow = HashMap::<String, HashSet<String>>::new();
    let mut write_allow = HashMap::<String, HashSet<String>>::new();

    for entry in parse_csv(std::env::var("MNEMOS_RBAC_READ").ok()) {
        if let Some((ns, principals)) = parse_namespace_principals(&entry) {
            read_allow.insert(ns, principals);
        }
    }
    for entry in parse_csv(std::env::var("MNEMOS_RBAC_WRITE").ok()) {
        if let Some((ns, principals)) = parse_namespace_principals(&entry) {
            write_allow.insert(ns, principals);
        }
    }

    RbacPolicy {
        admin_principals: admins.into_iter().collect(),
        read_namespace_allow: read_allow,
        write_namespace_allow: write_allow,
    }
}

fn parse_quota_policy_from_env() -> QuotaPolicy {
    QuotaPolicy {
        max_requests_per_minute: std::env::var("MNEMOS_QUOTA_RPM")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        max_top_k: std::env::var("MNEMOS_QUOTA_MAX_TOP_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(100),
        max_graph_hops: std::env::var("MNEMOS_QUOTA_MAX_GRAPH_HOPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4),
    }
}

fn log_tls_env_state() {
    let cert = std::env::var("MNEMOS_TLS_CERT_PATH").ok();
    let key = std::env::var("MNEMOS_TLS_KEY_PATH").ok();
    if cert.as_deref().unwrap_or("").is_empty() && key.as_deref().unwrap_or("").is_empty() {
        return;
    }
    eprintln!(
        "TLS cert/key env variables detected, but this binary is built without tonic TLS feature; run behind a TLS terminator (e.g. Envoy/Caddy/Nginx) for now."
    );
}

fn parse_csv(value: Option<String>) -> Vec<String> {
    value
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_namespace_principals(entry: &str) -> Option<(String, HashSet<String>)> {
    let (ns, principals) = entry.split_once(':')?;
    let principals: HashSet<String> = principals
        .split('|')
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    if ns.trim().is_empty() {
        return None;
    }
    Some((ns.trim().to_string(), principals))
}

async fn run_status_server(
    addr: SocketAddr,
    service: MnemosGrpcService,
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(addr).await?;
    loop {
        let (mut socket, _) = listener.accept().await?;
        let service = service.clone();
        tokio::spawn(async move {
            let mut buffer = [0u8; 1024];
            let n = socket.read(&mut buffer).await.unwrap_or(0);
            let req = String::from_utf8_lossy(&buffer[..n]);
            let path = req
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("/");
            let body = if path == "/metrics" {
                service.render_prometheus_metrics().into_bytes()
            } else {
                b"Mnemos is running".to_vec()
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.write_all(&body).await;
            let _ = socket.shutdown().await;
        });
    }
}
