# burn_jepa web

The low-level wasm API is available behind the `wasm` feature.

```sh
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
```

The Bevy example crate owns the static GitHub Pages shell under
`crates/bevy_burn_jepa/www`.
