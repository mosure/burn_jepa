#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    burn_jepa::cli::main()
}

#[cfg(target_arch = "wasm32")]
fn main() {}
