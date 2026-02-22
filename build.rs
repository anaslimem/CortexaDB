fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("failed to locate protoc");
    unsafe {
        std::env::set_var("PROTOC", protoc);
    }

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/mnemos.proto"], &["proto"])
        .expect("failed to compile protobuf definitions");
}
