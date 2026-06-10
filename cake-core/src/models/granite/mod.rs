//! IBM Granite model implementation (dense variants).
pub mod block;
pub mod config;
mod history;
mod model;

pub use block::*;
pub use config::*;
pub use history::*;
pub use model::*;

crate::impl_model_for_text!(Granite);
