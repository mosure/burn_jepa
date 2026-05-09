# bevy_burn_jepa

Small Bevy example crate for `burn_jepa`.

```sh
cargo run -p bevy_burn_jepa
```

The native entry point runs one Bevy schedule tick and executes the tiny sparse
V-JEPA pipeline through Burn's ndarray backend. The GitHub Pages shell in
`www/` is static by default so it can be deployed without bundling model weights.
