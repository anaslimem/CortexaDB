use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use tokio::sync::Mutex;
use tonic::transport::Channel;

pub mod proto {
    tonic::include_proto!("mnemos");
}

use proto::mnemos_service_client::MnemosServiceClient;

#[derive(Debug, Clone)]
struct BenchConfig {
    addr: String,
    namespace: String,
    vector_dim: usize,
    insert_ops: u64,
    query_ops: u64,
    concurrency: usize,
    top_k: u32,
}

#[derive(Debug, Clone)]
struct PhaseSummary {
    name: &'static str,
    ops: u64,
    concurrency: usize,
    elapsed_ms: u128,
    throughput_ops_s: f64,
    avg_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = parse_args();
    let endpoint = format!("http://{}", cfg.addr);
    let channel = Channel::from_shared(endpoint)?.connect().await?;

    println!("Running Mnemos gRPC benchmark with config: {:?}", cfg);

    let insert = run_insert_phase(&cfg, channel.clone()).await?;
    let query = run_query_phase(&cfg, channel).await?;

    print_markdown_summary(&cfg, &insert, &query);
    Ok(())
}

async fn run_insert_phase(
    cfg: &BenchConfig,
    channel: Channel,
) -> Result<PhaseSummary, Box<dyn std::error::Error>> {
    let counter = Arc::new(AtomicU64::new(0));
    let latencies = Arc::new(Mutex::new(Vec::<u128>::with_capacity(
        cfg.insert_ops as usize,
    )));
    let started = Instant::now();

    let mut handles = Vec::with_capacity(cfg.concurrency);
    for worker in 0..cfg.concurrency {
        let counter = Arc::clone(&counter);
        let latencies = Arc::clone(&latencies);
        let namespace = cfg.namespace.clone();
        let dim = cfg.vector_dim;
        let ops = cfg.insert_ops;
        let mut client = MnemosServiceClient::new(channel.clone());

        handles.push(tokio::spawn(async move {
            loop {
                let i = counter.fetch_add(1, Ordering::Relaxed);
                if i >= ops {
                    break;
                }
                let memory_id = 10_000_000 + i;
                let embedding = make_embedding(memory_id, dim);
                let req = proto::InsertMemoryRequest {
                    memory: Some(proto::Memory {
                        id: memory_id,
                        namespace: namespace.clone(),
                        content: format!("bench-memory-{}-w{}", i, worker).into_bytes(),
                        embedding,
                        created_at: 1_700_000_000 + i,
                        importance: 0.5,
                        metadata: vec![],
                    }),
                };

                let t0 = Instant::now();
                client.insert_memory(req).await?;
                let elapsed = t0.elapsed().as_micros();
                latencies.lock().await.push(elapsed);
            }
            Ok::<(), tonic::Status>(())
        }));
    }

    for h in handles {
        h.await??;
    }
    let elapsed = started.elapsed();
    let lats = latencies.lock().await.clone();
    Ok(summarize_phase("insert", cfg.insert_ops, cfg.concurrency, elapsed, lats))
}

async fn run_query_phase(
    cfg: &BenchConfig,
    channel: Channel,
) -> Result<PhaseSummary, Box<dyn std::error::Error>> {
    let counter = Arc::new(AtomicU64::new(0));
    let latencies = Arc::new(Mutex::new(Vec::<u128>::with_capacity(
        cfg.query_ops as usize,
    )));
    let started = Instant::now();

    let mut handles = Vec::with_capacity(cfg.concurrency);
    for _ in 0..cfg.concurrency {
        let counter = Arc::clone(&counter);
        let latencies = Arc::clone(&latencies);
        let namespace = cfg.namespace.clone();
        let dim = cfg.vector_dim;
        let ops = cfg.query_ops;
        let top_k = cfg.top_k;
        let mut client = MnemosServiceClient::new(channel.clone());

        handles.push(tokio::spawn(async move {
            loop {
                let i = counter.fetch_add(1, Ordering::Relaxed);
                if i >= ops {
                    break;
                }
                let query_vec = make_embedding(10_000_000 + (i % 1024), dim);
                let req = proto::QueryRequest {
                    query_embedding: query_vec,
                    top_k,
                    namespace: Some(namespace.clone()),
                    time_start: None,
                    time_end: None,
                    graph_hops: None,
                    candidate_multiplier: 0,
                    similarity_pct: 0,
                    importance_pct: 0,
                    recency_pct: 0,
                };
                let t0 = Instant::now();
                client.query(req).await?;
                let elapsed = t0.elapsed().as_micros();
                latencies.lock().await.push(elapsed);
            }
            Ok::<(), tonic::Status>(())
        }));
    }

    for h in handles {
        h.await??;
    }
    let elapsed = started.elapsed();
    let lats = latencies.lock().await.clone();
    Ok(summarize_phase("query", cfg.query_ops, cfg.concurrency, elapsed, lats))
}

fn summarize_phase(
    name: &'static str,
    ops: u64,
    concurrency: usize,
    elapsed: std::time::Duration,
    mut lats_us: Vec<u128>,
) -> PhaseSummary {
    lats_us.sort_unstable();
    let elapsed_ms = elapsed.as_millis();
    let throughput_ops_s = if elapsed.as_secs_f64() > 0.0 {
        ops as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    let avg_us = if lats_us.is_empty() {
        0.0
    } else {
        lats_us.iter().sum::<u128>() as f64 / lats_us.len() as f64
    };
    let p50 = percentile_us(&lats_us, 50.0);
    let p95 = percentile_us(&lats_us, 95.0);
    let p99 = percentile_us(&lats_us, 99.0);
    let max = lats_us.last().copied().unwrap_or(0) as f64;

    PhaseSummary {
        name,
        ops,
        concurrency,
        elapsed_ms,
        throughput_ops_s,
        avg_ms: avg_us / 1000.0,
        p50_ms: p50 / 1000.0,
        p95_ms: p95 / 1000.0,
        p99_ms: p99 / 1000.0,
        max_ms: max / 1000.0,
    }
}

fn percentile_us(sorted: &[u128], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0) * (sorted.len().saturating_sub(1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo] as f64;
    }
    let w = rank - lo as f64;
    (1.0 - w) * (sorted[lo] as f64) + w * (sorted[hi] as f64)
}

fn make_embedding(seed: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            ((x % 10_000) as f32) / 10_000.0
        })
        .collect()
}

fn print_markdown_summary(cfg: &BenchConfig, insert: &PhaseSummary, query: &PhaseSummary) {
    println!();
    println!("Benchmark summary");
    println!(
        "Workload: namespace={}, dim={}, insert_ops={}, query_ops={}, concurrency={}, top_k={}",
        cfg.namespace, cfg.vector_dim, cfg.insert_ops, cfg.query_ops, cfg.concurrency, cfg.top_k
    );
    println!();
    println!("| Phase | Ops | Conc | Total ms | Ops/s | Avg ms | P50 ms | P95 ms | P99 ms | Max ms |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for p in [insert, query] {
        println!(
            "| {} | {} | {} | {} | {:.2} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            p.name,
            p.ops,
            p.concurrency,
            p.elapsed_ms,
            p.throughput_ops_s,
            p.avg_ms,
            p.p50_ms,
            p.p95_ms,
            p.p99_ms,
            p.max_ms
        );
    }
}

fn parse_args() -> BenchConfig {
    BenchConfig {
        addr: arg("--addr").unwrap_or_else(|| "127.0.0.1:50051".to_string()),
        namespace: arg("--namespace").unwrap_or_else(|| "bench".to_string()),
        vector_dim: arg("--vector-dim")
            .and_then(|v| v.parse().ok())
            .unwrap_or(3),
        insert_ops: arg("--insert-ops")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000),
        query_ops: arg("--query-ops")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5000),
        concurrency: arg("--concurrency")
            .and_then(|v| v.parse().ok())
            .unwrap_or(32),
        top_k: arg("--top-k").and_then(|v| v.parse().ok()).unwrap_or(10),
    }
}

fn arg(flag: &str) -> Option<String> {
    let args = std::env::args().collect::<Vec<_>>();
    args.windows(2).find_map(|w| {
        if w[0] == flag {
            Some(w[1].clone())
        } else {
            None
        }
    })
}
