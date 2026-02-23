use std::net::SocketAddr;
use std::path::PathBuf;

use mnemos::engine::SyncPolicy;
use mnemos::query::{IntentPolicy, set_intent_policy};
use mnemos::service::grpc::{MnemosGrpcService, MnemosServiceServer};
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
    set_intent_policy(intent_policy.clone());

    std::fs::create_dir_all(&data_dir)?;
    let wal = data_dir.join("mnemos.wal");
    let seg = data_dir.join("segments");

    let store = if wal.exists() {
        MnemosStore::recover_with_policies(&wal, &seg, vector_dim, sync_policy, checkpoint_policy)?
    } else {
        MnemosStore::new_with_policies(&wal, &seg, vector_dim, sync_policy, checkpoint_policy)?
    };

    let service = MnemosGrpcService::new(store);

    println!("Mnemos gRPC listening on {}", addr);
    println!("Mnemos status HTTP listening on {}", status_addr);
    println!("Data dir: {}", data_dir.display());
    println!("Vector dimension: {}", vector_dim);
    println!("Sync policy: {:?}", sync_policy);
    println!("Checkpoint policy: {:?}", checkpoint_policy);
    println!("Intent policy: {:?}", intent_policy);

    tokio::spawn(async move {
        if let Err(e) = run_status_server(status_addr).await {
            eprintln!("status server error: {e}");
        }
    });

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

async fn run_status_server(addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(addr).await?;
    loop {
        let (mut socket, _) = listener.accept().await?;
        tokio::spawn(async move {
            let mut buffer = [0u8; 1024];
            let _ = socket.read(&mut buffer).await;
            let body = b"Mnemos is running";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.write_all(body).await;
            let _ = socket.shutdown().await;
        });
    }
}
