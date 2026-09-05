//! Stop `wasm-ld` from re-running every static constructor on every call.
//!
//! Module: `trips-web` build script (wasm32 only; a no-op on every other target)
//! Purpose: pass `--export=__wasm_call_ctors` to the wasm linker, which
//!     suppresses LLD's *command-export wrappers*. This is the single largest
//!     performance fact about the browser viewer: without it a `raw level-0`
//!     frame costs ~300 ms and with it ~20 ms.
//! Invariants:
//!     - The flag only reaches the `wasm32-*` link. `cargo::rustc-link-arg`
//!       applies to this crate's own cdylib and to nothing else in the graph,
//!       so no dependency is recompiled and the native `trips-viewer` build is
//!       untouched. (Setting `RUSTFLAGS` instead would rebuild all ~500
//!       dependency crates and would not be visible in the source tree.)
//!     - `web/trips.js` calls `__wasm_call_ctors()` exactly once after
//!       `init()`, and **refuses to start** if the export is missing. So if a
//!       future toolchain drops this flag the page says so instead of silently
//!       running 15x slower.
//! Related docs: `docs/WEB_VIEWER.md` ("Why the browser was 27x slower").
//!
//! # What LLD was doing
//!
//! `wasm32-unknown-unknown` has no libc, so `wasm-ld` synthesises
//! `__wasm_call_ctors` (the `.init_array` chain) itself, **unguarded**, and —
//! for a module that is not linked as a reactor — wraps *every* export in a
//! `<name>.command_export` wrapper that calls it on entry and
//! `__wasm_call_dtors` on exit. That is the WASI "command" ABI, where each
//! call is a fresh program run.
//!
//! Rust code normally has no static constructors, so those wrappers are free.
//! This graph does not: `cubecl-ir` pulls in `pliron`, whose dialect and
//! trait-cast registrations are thousands of `inventory::submit` calls, all of
//! them in `.init_array`. Measured on this Mac, one `__wasm_call_ctors` run is
//! ~110 microseconds.
//!
//! The wrappers are not only on the functions the page calls. `wasm-bindgen`
//! resolves `__externref_table_alloc`, `__externref_table_dealloc`,
//! `__wbindgen_malloc` and `__wbindgen_free` **by export name**, so its own
//! generated shims called the wrappers too — which means every `JsValue` that
//! `wgpu`'s WebGPU backend creates while building a bind group re-registered
//! the whole of `pliron`. At ~2,500 such allocations per frame that is the
//! entire frame.
//!
//! `--export=__wasm_call_ctors` makes LLD assume the caller will run the
//! constructors itself, so it emits none of the wrappers. `web/trips.js` is
//! that caller.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let target = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    if target.split(',').any(|f| f == "wasm") {
        println!("cargo::rustc-link-arg=--export=__wasm_call_ctors");
    }
}
