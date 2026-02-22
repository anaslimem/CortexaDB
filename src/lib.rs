pub mod core;
pub mod engine;
pub mod index;
pub mod query;
pub mod service;
pub mod storage;
pub mod store;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
