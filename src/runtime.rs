#[cfg(feature = "cuda")]
use std::env;
#[cfg(feature = "cuda")]
use std::path::Path;
#[cfg(feature = "cuda")]
use std::process::Command;

#[cfg(feature = "cuda")]
pub const CUDA_TRAIN_FORCE_ENV: &str = "BURN_JEPA_TRAIN_CUDA_FORCE";

#[cfg(feature = "cuda")]
pub fn cuda_runtime_preflight(force_env_var: &str) -> Result<(), String> {
    if env_bool(force_env_var) || env_bool("BURN_JEPA_PIPELINE_CUDA_FORCE") {
        return Ok(());
    }
    if env::var("CUDA_VISIBLE_DEVICES").ok().is_some_and(|value| {
        let value = value.trim();
        value.is_empty() || matches!(value, "-1" | "none" | "None" | "NONE")
    }) {
        return Err(format!(
            "CUDA_VISIBLE_DEVICES disables CUDA; set {force_env_var}=1 to try anyway"
        ));
    }
    let nvidia_smi = nvidia_smi_summary();
    if cfg!(target_os = "linux") && !cuda_device_nodes_visible() {
        return Err(cuda_missing_device_nodes_reason(
            nvidia_smi.as_ref(),
            force_env_var,
        ));
    }
    nvidia_smi.map(|_| ())
}

#[cfg(feature = "cuda")]
fn cuda_device_nodes_visible() -> bool {
    Path::new("/dev/nvidiactl").exists() || Path::new("/dev/nvidia0").exists()
}

#[cfg(feature = "cuda")]
fn cuda_missing_device_nodes_reason(
    nvidia_smi: Result<&String, &String>,
    force_env_var: &str,
) -> String {
    let mut reason = String::from("no /dev/nvidia* device nodes");
    match nvidia_smi {
        Ok(summary) if !summary.is_empty() => {
            reason.push_str("; nvidia-smi -L sees ");
            reason.push_str(summary);
        }
        Err(error) => {
            reason.push_str("; nvidia-smi -L probe failed: ");
            reason.push_str(error);
        }
        _ => {}
    }
    if Path::new("/proc/driver/nvidia/version").exists() {
        reason.push_str("; /proc/driver/nvidia is visible");
    }
    reason.push_str(
        &format!("; CUDA runtime cannot open a device without NVIDIA character devices; set {force_env_var}=1 to try anyway"),
    );
    reason
}

#[cfg(feature = "cuda")]
fn nvidia_smi_summary() -> Result<String, String> {
    match Command::new("nvidia-smi").arg("-L").output() {
        Ok(output) if output.status.success() && !output.stdout.is_empty() => {
            Ok(String::from_utf8_lossy(&output.stdout)
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "))
        }
        Ok(output) if output.status.success() => {
            Err("nvidia-smi -L returned no CUDA devices".into())
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("nvidia-smi -L failed: {}", stderr.trim()))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(err) => Err(format!("failed to run nvidia-smi -L: {err}")),
    }
}

#[cfg(feature = "cuda")]
fn env_bool(name: &str) -> bool {
    env::var(name).ok().is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}
