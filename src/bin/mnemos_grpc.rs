use std::net::SocketAddr;
use std::path::PathBuf;

use mnemos::service::grpc::{MnemosGrpcService, MnemosServiceServer};
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

    std::fs::create_dir_all(&data_dir)?;
    let wal = data_dir.join("mnemos.wal");
    let seg = data_dir.join("segments");

    let store = if wal.exists() {
        MnemosStore::recover(&wal, &seg, vector_dim)?
    } else {
        MnemosStore::new(&wal, &seg, vector_dim)?
    };

    let service = MnemosGrpcService::new(store);

    println!("Mnemos gRPC listening on {}", addr);
    println!("Mnemos status HTTP listening on {}", status_addr);
    println!("Data dir: {}", data_dir.display());
    println!("Vector dimension: {}", vector_dim);

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
