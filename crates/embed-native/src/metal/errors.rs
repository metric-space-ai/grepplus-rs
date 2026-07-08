use std::sync::{Mutex, OnceLock};

static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();

fn slot() -> &'static Mutex<String> {
    LAST_ERROR.get_or_init(|| Mutex::new(String::new()))
}

pub fn set_last_error(message: impl Into<String>) {
    if let Ok(mut guard) = slot().lock() {
        *guard = message.into();
    }
}

pub fn last_error() -> String {
    slot().lock().map(|guard| guard.clone()).unwrap_or_default()
}
