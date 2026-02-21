pub mod engine;
pub mod core;
pub mod storage;
pub mod index;
pub mod query;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
