fn main() {
    let status = bevy_burn_jepa::run_once();
    println!(
        "burn_jepa bevy smoke: context_tokens={} target_tokens={} embedding_dim={}",
        status.context_tokens, status.target_tokens, status.embedding_dim
    );
}
