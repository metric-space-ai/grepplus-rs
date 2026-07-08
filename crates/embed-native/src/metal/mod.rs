//! Feature-gated Metal backend plumbing.
//!
//! This module vendors the generic ggml Metal host dispatch surface from CTOX
//! and adapts it to `greppy-embed-native`. The full Gemma graph is not wired
//! here yet; these pieces are the runtime, kargs, tensor descriptors, and op
//! dispatchers M4 needs.

pub mod errors;
pub mod ffi;
pub mod kargs;
pub mod model;
pub mod ops;
pub mod tensor;
pub mod weights;
