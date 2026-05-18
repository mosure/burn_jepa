use std::{env, fs, path::PathBuf};

use anyhow::{Context, Result};
use bevy_jepa::perf::{PipelinePerfConfig, run_native_perf_matrix};

fn main() -> Result<()> {
    let (config, output) = parse_args()?;
    let report = run_native_perf_matrix(config)?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory `{}`", parent.display()))?;
        }
        fs::write(&path, &json).with_context(|| format!("write `{}`", path.display()))?;
    }
    println!("{json}");
    eprintln!("{}", report.markdown());
    Ok(())
}

fn parse_args() -> Result<(PipelinePerfConfig, Option<PathBuf>)> {
    let mut config = PipelinePerfConfig::default();
    let mut output = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmups" => {
                config.warmups = parse_next(&mut args, "--warmups")?;
            }
            "--reps" => {
                config.reps = parse_next(&mut args, "--reps")?;
            }
            "--image-sizes" => {
                config.image_sizes = parse_csv(&next_arg(&mut args, "--image-sizes")?)?;
            }
            "--densities" => {
                config.densities = parse_csv(&next_arg(&mut args, "--densities")?)?;
            }
            "--output" => {
                output = Some(PathBuf::from(next_arg(&mut args, "--output")?));
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument `{other}`; pass --help for usage"),
        }
    }
    Ok((config, output))
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    next_arg(args, flag)?
        .parse()
        .map_err(|err| anyhow::anyhow!("invalid value for {flag}: {err}"))
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    args.next()
        .ok_or_else(|| anyhow::anyhow!("{flag} requires a value"))
}

fn parse_csv<T: std::str::FromStr>(value: &str) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse()
                .map_err(|err| anyhow::anyhow!("invalid CSV value `{part}`: {err}"))
        })
        .collect()
}

fn print_help() {
    println!(
        "usage: cargo run -p bevy_jepa --release --example perf_matrix -- \\
         [--warmups 4] [--reps 16] [--image-sizes 256,512] [--densities 0.1,0.25,0.5,1.0] [--output target/perf/native.json]"
    );
}
