# burn_jepa web

The low-level wasm API is available behind the `wasm` feature.

```sh
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
```

The Bevy example crate owns the static page shell under
`crates/bevy_burn_jepa/www`. GitHub Pages deployment is environment-dependent;
the repository currently keeps that workflow disabled because the account plan
does not expose Pages for this repo.
