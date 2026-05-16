# burn_jepa web

The low-level wasm API is available behind the `wasm` feature.

```sh
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
```

The Bevy viewer crate owns the GitHub Pages shell under `crates/bevy_jepa/www`.
The Pages workflow builds the wasm target with `wasm-bindgen` before upload.
