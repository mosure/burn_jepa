# bevy_burn_jepa

Small Bevy example crate for `burn_jepa`.

```sh
cargo run -p bevy_burn_jepa
```

The native entry point runs one Bevy schedule tick and executes the tiny sparse
V-JEPA pipeline through Burn's ndarray backend. The static page shell in `www/`
can be deployed without bundling model weights. GitHub Pages itself is
environment-dependent and is currently disabled remotely for the root repo
because the account plan does not expose Pages for it.
