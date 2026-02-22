use std::net::SocketAddr;
use std::path::PathBuf;

use mnemos::service::grpc::{MnemosGrpcService, MnemosServiceServer};
use mnemos::store::MnemosStore;
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr: SocketAddr = std::env::var("MNEMOS_GRPC_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:50051".to_string())
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
    println!("Data dir: {}", data_dir.display());
    println!("Vector dimension: {}", vector_dim);

    Server::builder()
        .add_service(MnemosServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
