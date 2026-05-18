use crate::{
    BurnJepaTrainConfig, JepaDataset, JepaDatasetConfig, JepaDatasetKind, JepaManifestRow,
    JepaTensorBatch, JepaTrainBackend, SparseTokenMask, TokenGridShape, TttInsertionMode,
    TttLayerPlacement, TttMemoryUpdateSource, TttTargetMode, VJepa2_1Model, VJepaConfig,
    VJepaLoadOptions, VJepaTttModel, apply_token_mask, dataset_from_config, load_jepa_tensor_batch,
    train_ttt_distillation, video_token_grid,
};
use anyhow::{Context, Result, bail, ensure};
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentConfig {
    pub name: String,
    pub output_dir: PathBuf,
    pub require_real_checkpoint: bool,
    pub require_real_dataset: bool,
    pub seeds: Vec<u64>,
    pub densities: Vec<f32>,
    pub target_density: f32,
    pub model_variants: Vec<ExperimentModelVariant>,
    pub mask_policies: Vec<ExperimentMaskPolicy>,
    pub ttt_layer_sets: Vec<ExperimentTttLayerSet>,
    pub base: BurnJepaTrainConfig,
    pub data: ExperimentDataConfig,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        let mut base = BurnJepaTrainConfig::default();
        base.model.output_dir = PathBuf::from("target/burn-jepa-experiments/trial");
        base.model.save_model = false;
        base.training.max_steps = 2;
        base.training.eval_steps = 1;
        base.training.batch_size = 1;
        base.dataset.synthetic_len = 8;
        base.loss.predictor_loss_weight = 0.25;
        Self {
            name: "ttt-open-set-ablation".to_string(),
            output_dir: PathBuf::from("target/burn-jepa-experiments"),
            require_real_checkpoint: false,
            require_real_dataset: false,
            seeds: vec![0],
            densities: vec![0.01, 0.05, 0.10, 0.25],
            target_density: 0.05,
            model_variants: vec![
                ExperimentModelVariant::Teacher3dReference,
                ExperimentModelVariant::SingleFrameNoTtt,
                ExperimentModelVariant::TttTeacherFinal,
                ExperimentModelVariant::TttSelfHidden,
            ],
            mask_policies: vec![
                ExperimentMaskPolicy::FullFrame,
                ExperimentMaskPolicy::KeepRatio,
                ExperimentMaskPolicy::RandomSparse,
                ExperimentMaskPolicy::PatchDiff,
                ExperimentMaskPolicy::AutogazeSparse,
                ExperimentMaskPolicy::PrecomputedMasks,
            ],
            ttt_layer_sets: vec![ExperimentTttLayerSet::encoder_first_last()],
            base,
            data: ExperimentDataConfig::default(),
        }
    }
}

impl ExperimentConfig {
    pub fn from_toml_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("read experiment config {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parse experiment config {}", path.display()))
    }

    pub fn to_toml_string(&self) -> Result<String> {
        toml::to_string_pretty(self).context("serialize experiment config")
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(!self.name.trim().is_empty(), "experiment.name is required");
        ensure!(
            !self.seeds.is_empty(),
            "experiment.seeds must contain at least one seed"
        );
        ensure!(
            !self.densities.is_empty(),
            "experiment.densities must contain at least one density"
        );
        ensure!(
            self.densities
                .iter()
                .all(|density| density.is_finite() && *density > 0.0 && *density < 1.0),
            "experiment densities must be finite and in (0, 1)"
        );
        ensure!(
            self.target_density.is_finite()
                && self.target_density > 0.0
                && self.target_density < 1.0,
            "experiment.target_density must be finite and in (0, 1)"
        );
        ensure!(
            !self.model_variants.is_empty(),
            "experiment.model_variants must not be empty"
        );
        ensure!(
            !self.mask_policies.is_empty(),
            "experiment.mask_policies must not be empty"
        );
        ensure!(
            !self.ttt_layer_sets.is_empty(),
            "experiment.ttt_layer_sets must not be empty"
        );
        if self.require_real_checkpoint {
            ensure!(
                self.base.model.checkpoint_dir.is_some()
                    || self.base.model.teacher_checkpoint_dir.is_some(),
                "experiment requires a real checkpoint but no model checkpoint_dir is configured"
            );
        }
        if self.require_real_dataset {
            ensure!(
                self.base.dataset.kind == JepaDatasetKind::Manifest
                    && self.base.dataset.train_manifest.is_some(),
                "experiment requires a real dataset but no train manifest is configured"
            );
        }
        Ok(())
    }

    fn planned_trials(&self) -> Vec<ExperimentTrial> {
        let mut trials = Vec::new();
        for &seed in &self.seeds {
            for &density in &self.densities {
                for model_variant in &self.model_variants {
                    for mask_policy in &self.mask_policies {
                        for ttt_layer_set_index in 0..self.ttt_layer_sets.len() {
                            trials.push(ExperimentTrial {
                                seed,
                                density,
                                model_variant: *model_variant,
                                mask_policy: *mask_policy,
                                ttt_layer_set_index,
                            });
                        }
                    }
                }
            }
        }
        trials
    }

    fn ttt_layer_set(&self, trial: ExperimentTrial) -> &ExperimentTttLayerSet {
        self.ttt_layer_sets
            .get(trial.ttt_layer_set_index)
            .unwrap_or_else(|| &self.ttt_layer_sets[0])
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentDataConfig {
    pub input: Option<PathBuf>,
    pub output_dir: PathBuf,
    pub train_manifest: PathBuf,
    pub eval_manifest: PathBuf,
    pub domain: Option<String>,
    pub domain_from_parent: bool,
    pub domain_from_clip_prefix: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub autogaze_masks: Option<ExperimentAutogazeMaskConfig>,
    pub eval_ratio: f32,
    pub window_frames: usize,
    pub window_stride: usize,
    pub extract_videos: bool,
}

impl Default for ExperimentDataConfig {
    fn default() -> Self {
        Self {
            input: None,
            output_dir: PathBuf::from("target/burn-jepa-experiments/data"),
            train_manifest: PathBuf::from("target/burn-jepa-experiments/data/train.jsonl"),
            eval_manifest: PathBuf::from("target/burn-jepa-experiments/data/eval.jsonl"),
            domain: None,
            domain_from_parent: false,
            domain_from_clip_prefix: false,
            autogaze_masks: None,
            eval_ratio: 0.2,
            window_frames: 64,
            window_stride: 64,
            extract_videos: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentAutogazeMaskConfig {
    pub checkpoint_dir: PathBuf,
    pub backend: Option<JepaTrainBackend>,
    pub streaming: bool,
    pub context_density: Option<f32>,
    pub target_density: Option<f32>,
    pub max_gaze_tokens_each_frame: Option<usize>,
    pub task_loss_requirement: Option<f32>,
    pub top_k_overfetch: f32,
    pub dilation: usize,
}

impl Default for ExperimentAutogazeMaskConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: default_autogaze_checkpoint_dir(),
            backend: None,
            streaming: false,
            context_density: None,
            target_density: None,
            max_gaze_tokens_each_frame: None,
            task_loss_requirement: None,
            top_k_overfetch: 1.0,
            dilation: 0,
        }
    }
}

fn default_autogaze_checkpoint_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("BURN_JEPA_AUTOGAZE_CHECKPOINT_DIR") {
        return PathBuf::from(path);
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a")
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentModelVariant {
    Teacher3dReference,
    SingleFrameNoTtt,
    TttTeacherFinal,
    TttSelfHidden,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentMaskPolicy {
    FullFrame,
    KeepRatio,
    RandomSparse,
    PatchDiff,
    AutogazeSparse,
    PrecomputedMasks,
    ManifestPrecomputedMasks,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentTttLayerSet {
    pub name: String,
    #[serde(default)]
    pub insertion: Option<TttInsertionMode>,
    #[serde(default)]
    pub placement: Option<TttLayerPlacement>,
    #[serde(default)]
    pub encoder_layers: Vec<usize>,
    #[serde(default)]
    pub predictor_layers: Vec<usize>,
}

impl ExperimentTttLayerSet {
    pub fn encoder_first_last() -> Self {
        Self {
            name: "encoder_first_last".to_string(),
            insertion: None,
            placement: Some(TttLayerPlacement::FirstLast),
            encoder_layers: Vec::new(),
            predictor_layers: Vec::new(),
        }
    }

    pub fn encoder_thirds() -> Self {
        Self {
            name: "encoder_thirds".to_string(),
            insertion: None,
            placement: Some(TttLayerPlacement::Thirds),
            encoder_layers: Vec::new(),
            predictor_layers: Vec::new(),
        }
    }

    fn apply_to(&self, config: &mut BurnJepaTrainConfig) {
        if let Some(insertion) = self.insertion {
            config.ttt.insertion = insertion;
        }
        if let Some(placement) = self.placement {
            config.ttt.layer_placement = placement;
        } else {
            config.ttt.layer_placement = TttLayerPlacement::Explicit;
        }
        config.ttt.layers = self.encoder_layers.clone();
        config.ttt.predictor_layers = self.predictor_layers.clone();
    }

    fn resolved_encoder_layers(&self, base: &BurnJepaTrainConfig) -> Vec<usize> {
        let mut config = base.clone();
        self.apply_to(&mut config);
        model_config(&config)
            .map(|model| config.ttt.resolved_layers(&model))
            .unwrap_or_else(|_| self.encoder_layers.clone())
    }
}

impl Default for ExperimentTttLayerSet {
    fn default() -> Self {
        Self::encoder_first_last()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ExperimentTrial {
    pub seed: u64,
    pub density: f32,
    pub model_variant: ExperimentModelVariant,
    pub mask_policy: ExperimentMaskPolicy,
    #[serde(default)]
    pub ttt_layer_set_index: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExperimentPlanReport {
    pub name: String,
    pub output_dir: PathBuf,
    pub trial_count: usize,
    pub config_hash: u64,
    pub run_manifest: PathBuf,
    pub planned_trials: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentRunReport {
    pub name: String,
    pub output_dir: PathBuf,
    pub trial_count: usize,
    pub completed_trials: usize,
    pub failed_trials: usize,
    pub elapsed_ms: u128,
    pub summary_path: PathBuf,
    pub analysis_path: PathBuf,
    pub csv_path: PathBuf,
    #[serde(default)]
    pub success_criteria: ExperimentSuccessCriteria,
    pub trials: Vec<ExperimentTrialReport>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExperimentSuccessCriteria {
    pub all_trials_completed: bool,
    pub real_checkpoint_configured: bool,
    pub real_dataset_configured: bool,
    pub full_model_matrix: bool,
    pub full_mask_matrix: bool,
    pub mask_loss_enabled: bool,
    pub density_count: usize,
    pub ttt_layer_set_count: usize,
    pub matched_ttt_trials: usize,
    pub ttt_loss_improved_trials: usize,
    pub ttt_cosine_improved_trials: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentTrialReport {
    pub trial_id: String,
    pub seed: u64,
    pub density: f32,
    pub target_density: f32,
    pub model_variant: ExperimentModelVariant,
    pub mask_policy: ExperimentMaskPolicy,
    #[serde(default)]
    pub ttt_layer_set: String,
    #[serde(default)]
    pub ttt_encoder_layers: Vec<usize>,
    #[serde(default)]
    pub ttt_predictor_layers: Vec<usize>,
    pub status: ExperimentTrialStatus,
    pub train_final_loss: Option<f64>,
    pub train_best_loss: Option<f64>,
    pub eval_loss: Option<f64>,
    pub eval_cosine: Option<f64>,
    #[serde(default)]
    pub eval_full_loss: Option<f64>,
    #[serde(default)]
    pub eval_full_cosine: Option<f64>,
    #[serde(default)]
    pub pre_train_eval_loss: Option<f64>,
    #[serde(default)]
    pub pre_train_eval_cosine: Option<f64>,
    #[serde(default)]
    pub pre_train_full_eval_loss: Option<f64>,
    #[serde(default)]
    pub pre_train_full_eval_cosine: Option<f64>,
    #[serde(default)]
    pub mask_context_tokens: Option<usize>,
    #[serde(default)]
    pub mask_target_tokens: Option<usize>,
    #[serde(default)]
    pub mask_context_density: Option<f32>,
    #[serde(default)]
    pub mask_target_density: Option<f32>,
    #[serde(default)]
    pub ttt_fast_weight_bytes_f32: Option<usize>,
    #[serde(default)]
    pub ttt_trainable_param_bytes_f32: Option<usize>,
    pub elapsed_ms: u128,
    #[serde(default)]
    pub timing: ExperimentTrialTiming,
    pub samples_per_second: Option<f64>,
    pub report_path: Option<PathBuf>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ExperimentTrialTiming {
    pub setup_ms: u128,
    pub train_ms: u128,
    pub eval_ms: u128,
    pub mask_ms: u128,
    pub teacher_forward_ms: u128,
    pub student_forward_ms: u128,
    pub loss_ms: u128,
    pub backward_ms: u128,
    pub optimizer_ms: u128,
    pub backward_optim_ms: u128,
    pub teacher_cache_hits: usize,
    pub teacher_cache_misses: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentTrialStatus {
    Completed,
    Failed,
    Skipped,
}

#[derive(Clone, Debug, Serialize)]
pub struct ExperimentDataReport {
    pub input: PathBuf,
    pub train_manifest: PathBuf,
    pub eval_manifest: PathBuf,
    pub clips: usize,
    pub domains: Vec<String>,
    pub train_rows: usize,
    pub eval_rows: usize,
    pub autogaze_mask_rows: usize,
    pub extracted_videos: usize,
}

pub fn write_experiment_plan(config: &ExperimentConfig) -> Result<ExperimentPlanReport> {
    config.validate()?;
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("create {}", config.output_dir.display()))?;
    let trials = config.planned_trials();
    let config_json = serde_json::to_string_pretty(config)?;
    let config_hash = stable_hash(&config_json);
    let run_manifest = config.output_dir.join("run-manifest.json");
    let planned_trials = config.output_dir.join("planned-trials.json");
    let manifest = serde_json::json!({
        "name": config.name,
        "created_unix_ms": unix_ms(),
        "config_hash": config_hash,
        "git_sha": git_sha().ok(),
        "checkpoint_dir": config.base.model.checkpoint_dir,
        "teacher_checkpoint_dir": config.base.model.teacher_checkpoint_dir,
        "train_manifest": config.base.dataset.train_manifest,
        "eval_manifest": config.base.dataset.eval_manifest,
        "backend": config.base.training.backend,
        "trial_count": trials.len(),
    });
    fs::write(&run_manifest, serde_json::to_string_pretty(&manifest)?)
        .with_context(|| format!("write {}", run_manifest.display()))?;
    fs::write(&planned_trials, serde_json::to_string_pretty(&trials)?)
        .with_context(|| format!("write {}", planned_trials.display()))?;
    Ok(ExperimentPlanReport {
        name: config.name.clone(),
        output_dir: config.output_dir.clone(),
        trial_count: trials.len(),
        config_hash,
        run_manifest,
        planned_trials,
    })
}

pub fn prepare_experiment_data(config: &ExperimentConfig) -> Result<ExperimentDataReport> {
    let Some(input) = config.data.input.as_ref() else {
        bail!("experiment data input is not configured");
    };
    let input = input.canonicalize().unwrap_or_else(|_| input.clone());
    fs::create_dir_all(&config.data.output_dir)
        .with_context(|| format!("create {}", config.data.output_dir.display()))?;
    let mut clips = collect_frame_clips(&input)?;
    let mut extracted_videos = 0;
    if clips.is_empty() && config.data.extract_videos {
        extracted_videos = extract_videos_to_frame_dirs(&input, &config.data.output_dir)?;
        clips = collect_frame_clips(&config.data.output_dir)?;
    }
    ensure!(
        !clips.is_empty(),
        "no frame clips found under {}; provide extracted frame directories or set extract_videos=true",
        input.display()
    );
    clips.sort_by(|left, right| left.clip_id.cmp(&right.clip_id));
    let eval_every = eval_every(config.data.eval_ratio);
    let mut train_rows = Vec::new();
    let mut eval_rows = Vec::new();
    for (clip_index, clip) in clips.iter().enumerate() {
        let domain = clip_domain(clip, &config.data);
        let rows = clip_rows(
            clip,
            config.data.window_frames,
            config.data.window_stride,
            domain,
        );
        if clip_index % eval_every == 0 {
            eval_rows.extend(rows);
        } else {
            train_rows.extend(rows);
        }
    }
    if train_rows.is_empty() && !eval_rows.is_empty() {
        train_rows.push(eval_rows[0].clone());
    }
    if eval_rows.is_empty() && !train_rows.is_empty() {
        eval_rows.push(train_rows[0].clone());
    }
    let autogaze_mask_rows =
        prepare_autogaze_masks_if_configured(config, &mut train_rows, &mut eval_rows)?;
    let domains = train_rows
        .iter()
        .chain(eval_rows.iter())
        .filter_map(|row| row.domain.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    write_jsonl(&config.data.train_manifest, &train_rows)?;
    write_jsonl(&config.data.eval_manifest, &eval_rows)?;
    Ok(ExperimentDataReport {
        input,
        train_manifest: config.data.train_manifest.clone(),
        eval_manifest: config.data.eval_manifest.clone(),
        clips: clips.len(),
        domains,
        train_rows: train_rows.len(),
        eval_rows: eval_rows.len(),
        autogaze_mask_rows,
        extracted_videos,
    })
}

pub fn run_experiment<B: crate::TttSparsePatchifyTrainingBackend>(
    config: &ExperimentConfig,
    device: &B::Device,
) -> Result<ExperimentRunReport> {
    config.validate()?;
    let start = Instant::now();
    let _plan = write_experiment_plan(config)?;
    let mut trials = Vec::new();
    for trial in config.planned_trials() {
        trials.push(run_trial::<B>(config, trial, device));
    }
    let completed_trials = trials
        .iter()
        .filter(|trial| trial.status == ExperimentTrialStatus::Completed)
        .count();
    let failed_trials = trials
        .iter()
        .filter(|trial| trial.status == ExperimentTrialStatus::Failed)
        .count();
    let success_criteria = experiment_success_criteria(config, &trials);
    let report = ExperimentRunReport {
        name: config.name.clone(),
        output_dir: config.output_dir.clone(),
        trial_count: trials.len(),
        completed_trials,
        failed_trials,
        elapsed_ms: start.elapsed().as_millis(),
        summary_path: config.output_dir.join("experiment-summary.json"),
        analysis_path: config.output_dir.join("ablation-summary.md"),
        csv_path: config.output_dir.join("trial-metrics.csv"),
        success_criteria,
        trials,
    };
    fs::write(&report.summary_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("write {}", report.summary_path.display()))?;
    write_trial_csv(&report.csv_path, &report.trials)?;
    write_analysis(&report.analysis_path, &report)?;
    Ok(report)
}

pub fn analyze_experiment(run_dir: impl AsRef<Path>) -> Result<ExperimentRunReport> {
    let run_dir = run_dir.as_ref();
    let summary_path = run_dir.join("experiment-summary.json");
    let text = fs::read_to_string(&summary_path)
        .with_context(|| format!("read {}", summary_path.display()))?;
    let mut report: ExperimentRunReport =
        serde_json::from_str(&text).with_context(|| format!("parse {}", summary_path.display()))?;
    let previous_criteria = report.success_criteria.clone();
    report.success_criteria = experiment_success_criteria_from_trials(&report.trials, &report);
    report.success_criteria.real_checkpoint_configured =
        previous_criteria.real_checkpoint_configured;
    report.success_criteria.real_dataset_configured = previous_criteria.real_dataset_configured;
    report.success_criteria.mask_loss_enabled = previous_criteria.mask_loss_enabled;
    write_trial_csv(&report.csv_path, &report.trials)?;
    write_analysis(&report.analysis_path, &report)?;
    Ok(report)
}

fn experiment_success_criteria(
    config: &ExperimentConfig,
    trials: &[ExperimentTrialReport],
) -> ExperimentSuccessCriteria {
    let mut criteria = experiment_success_criteria_from_trials(
        trials,
        &ExperimentRunReport {
            name: config.name.clone(),
            output_dir: config.output_dir.clone(),
            trial_count: trials.len(),
            completed_trials: trials
                .iter()
                .filter(|trial| trial.status == ExperimentTrialStatus::Completed)
                .count(),
            failed_trials: trials
                .iter()
                .filter(|trial| trial.status == ExperimentTrialStatus::Failed)
                .count(),
            elapsed_ms: 0,
            summary_path: PathBuf::new(),
            analysis_path: PathBuf::new(),
            csv_path: PathBuf::new(),
            success_criteria: ExperimentSuccessCriteria::default(),
            trials: Vec::new(),
        },
    );
    criteria.real_checkpoint_configured = config.base.model.checkpoint_dir.is_some()
        || config.base.model.teacher_checkpoint_dir.is_some();
    criteria.real_dataset_configured = config.base.dataset.kind == JepaDatasetKind::Manifest
        && config.base.dataset.train_manifest.is_some();
    criteria.mask_loss_enabled = !config.mask_policies.is_empty()
        && (config.base.loss.feature_loss_weight > 0.0
            || config.base.loss.predictor_loss_weight > 0.0);
    criteria.density_count = config.densities.len();
    criteria.ttt_layer_set_count = config.ttt_layer_sets.len();
    criteria
}

fn experiment_success_criteria_from_trials(
    trials: &[ExperimentTrialReport],
    report: &ExperimentRunReport,
) -> ExperimentSuccessCriteria {
    let full_model_matrix = [
        ExperimentModelVariant::Teacher3dReference,
        ExperimentModelVariant::SingleFrameNoTtt,
        ExperimentModelVariant::TttTeacherFinal,
        ExperimentModelVariant::TttSelfHidden,
    ]
    .into_iter()
    .all(|model| trials.iter().any(|trial| trial.model_variant == model));
    let full_mask_matrix = [
        ExperimentMaskPolicy::FullFrame,
        ExperimentMaskPolicy::KeepRatio,
        ExperimentMaskPolicy::RandomSparse,
        ExperimentMaskPolicy::PatchDiff,
        ExperimentMaskPolicy::AutogazeSparse,
        ExperimentMaskPolicy::PrecomputedMasks,
    ]
    .into_iter()
    .all(|policy| trials.iter().any(|trial| trial.mask_policy == policy));

    let mut matched_ttt_trials = 0;
    let mut ttt_loss_improved_trials = 0;
    let mut ttt_cosine_improved_trials = 0;
    for trial in trials.iter().filter(|trial| {
        matches!(
            trial.model_variant,
            ExperimentModelVariant::TttTeacherFinal | ExperimentModelVariant::TttSelfHidden
        ) && trial.status == ExperimentTrialStatus::Completed
    }) {
        let baseline = trials.iter().find(|candidate| {
            candidate.model_variant == ExperimentModelVariant::SingleFrameNoTtt
                && candidate.mask_policy == trial.mask_policy
                && candidate.seed == trial.seed
                && (candidate.density - trial.density).abs() <= f32::EPSILON
                && candidate.ttt_layer_set == trial.ttt_layer_set
                && candidate.status == ExperimentTrialStatus::Completed
        });
        if let Some(baseline) = baseline {
            matched_ttt_trials += 1;
            if let (Some(ttt), Some(base)) = (trial.eval_loss, baseline.eval_loss)
                && ttt < base
            {
                ttt_loss_improved_trials += 1;
            }
            if let (Some(ttt), Some(base)) = (trial.eval_cosine, baseline.eval_cosine)
                && ttt > base
            {
                ttt_cosine_improved_trials += 1;
            }
        }
    }

    ExperimentSuccessCriteria {
        all_trials_completed: report.completed_trials == report.trial_count
            && report.failed_trials == 0,
        real_checkpoint_configured: false,
        real_dataset_configured: false,
        full_model_matrix,
        full_mask_matrix,
        mask_loss_enabled: false,
        density_count: unique_density_count(trials),
        ttt_layer_set_count: unique_layer_set_count(trials),
        matched_ttt_trials,
        ttt_loss_improved_trials,
        ttt_cosine_improved_trials,
    }
}

fn run_trial<B: crate::TttSparsePatchifyTrainingBackend>(
    config: &ExperimentConfig,
    trial: ExperimentTrial,
    device: &B::Device,
) -> ExperimentTrialReport {
    let start = Instant::now();
    let trial_id = trial_id(trial);
    let result = match trial.model_variant {
        ExperimentModelVariant::Teacher3dReference => {
            evaluate_teacher_reference::<B>(config, trial, device)
        }
        ExperimentModelVariant::SingleFrameNoTtt => {
            evaluate_single_frame::<B>(config, trial, device)
        }
        ExperimentModelVariant::TttTeacherFinal | ExperimentModelVariant::TttSelfHidden => {
            train_ttt_trial::<B>(config, trial, device)
        }
    };
    match result {
        Ok(mut report) => {
            report.elapsed_ms = start.elapsed().as_millis();
            report
        }
        Err(error) => ExperimentTrialReport {
            trial_id,
            seed: trial.seed,
            density: trial.density,
            target_density: config.target_density,
            model_variant: trial.model_variant,
            mask_policy: trial.mask_policy,
            ttt_layer_set: trial_layer_set_name(config, trial),
            ttt_encoder_layers: trial_encoder_layers(config, trial),
            ttt_predictor_layers: config.ttt_layer_set(trial).predictor_layers.clone(),
            status: ExperimentTrialStatus::Failed,
            train_final_loss: None,
            train_best_loss: None,
            eval_loss: None,
            eval_cosine: None,
            eval_full_loss: None,
            eval_full_cosine: None,
            pre_train_eval_loss: None,
            pre_train_eval_cosine: None,
            pre_train_full_eval_loss: None,
            pre_train_full_eval_cosine: None,
            mask_context_tokens: None,
            mask_target_tokens: None,
            mask_context_density: None,
            mask_target_density: None,
            ttt_fast_weight_bytes_f32: None,
            ttt_trainable_param_bytes_f32: None,
            elapsed_ms: start.elapsed().as_millis(),
            timing: ExperimentTrialTiming::default(),
            samples_per_second: None,
            report_path: None,
            error: Some(format!("{error:#}")),
        },
    }
}

fn train_ttt_trial<B: crate::TttSparsePatchifyTrainingBackend>(
    config: &ExperimentConfig,
    trial: ExperimentTrial,
    device: &B::Device,
) -> Result<ExperimentTrialReport> {
    let mut train_config = trial_train_config(config, trial)?;
    train_config.ttt.target = match trial.model_variant {
        ExperimentModelVariant::TttTeacherFinal => TttTargetMode::TeacherFinal,
        ExperimentModelVariant::TttSelfHidden => TttTargetMode::SelfHidden,
        _ => unreachable!("non-TTT variant routed to train_ttt_trial"),
    };
    train_config.ttt.memory_update = match trial.model_variant {
        ExperimentModelVariant::TttTeacherFinal => TttMemoryUpdateSource::TeacherForcedDiagnostic,
        ExperimentModelVariant::TttSelfHidden => TttMemoryUpdateSource::SelfHidden,
        _ => unreachable!("non-TTT variant routed to train_ttt_trial"),
    };
    let output_dir = config.output_dir.join(trial_id(trial));
    train_config.model.output_dir = output_dir;
    let report = train_ttt_distillation::<B>(&train_config, device)?;
    Ok(ExperimentTrialReport {
        trial_id: trial_id(trial),
        seed: trial.seed,
        density: trial.density,
        target_density: config.target_density,
        model_variant: trial.model_variant,
        mask_policy: trial.mask_policy,
        ttt_layer_set: trial_layer_set_name(config, trial),
        ttt_encoder_layers: report.memory.layers.clone(),
        ttt_predictor_layers: report.memory.predictor_layers.clone(),
        status: ExperimentTrialStatus::Completed,
        train_final_loss: Some(report.final_loss),
        train_best_loss: Some(report.best_loss),
        eval_loss: report.eval_loss,
        eval_cosine: report.eval_cosine,
        eval_full_loss: report.eval_full_loss,
        eval_full_cosine: report.eval_full_cosine,
        pre_train_eval_loss: report.pre_train_eval_loss,
        pre_train_eval_cosine: report.pre_train_eval_cosine,
        pre_train_full_eval_loss: report.pre_train_full_eval_loss,
        pre_train_full_eval_cosine: report.pre_train_full_eval_cosine,
        mask_context_tokens: report.mask.as_ref().map(|mask| mask.context_tokens),
        mask_target_tokens: report.mask.as_ref().map(|mask| mask.target_tokens),
        mask_context_density: report.mask.as_ref().map(|mask| mask.context_density),
        mask_target_density: report.mask.as_ref().map(|mask| mask.target_density),
        ttt_fast_weight_bytes_f32: Some(report.memory.fast_weight_bytes_f32),
        ttt_trainable_param_bytes_f32: Some(report.memory.trainable_param_bytes_f32),
        elapsed_ms: report.elapsed_ms,
        timing: ExperimentTrialTiming {
            setup_ms: 0,
            train_ms: report.train_elapsed_ms,
            eval_ms: report.eval_elapsed_ms,
            mask_ms: report.eval_stage.mask_ms,
            teacher_forward_ms: report.eval_stage.teacher_forward_ms,
            student_forward_ms: report.eval_stage.student_forward_ms,
            loss_ms: report.eval_stage.loss_ms,
            backward_ms: report.train_stage.backward_ms,
            optimizer_ms: report.train_stage.optimizer_ms,
            backward_optim_ms: report.train_stage.backward_optim_ms,
            teacher_cache_hits: report.train_stage.teacher_cache_hits
                + report.eval_stage.teacher_cache_hits,
            teacher_cache_misses: report.train_stage.teacher_cache_misses
                + report.eval_stage.teacher_cache_misses,
        },
        samples_per_second: Some(report.samples_per_second),
        report_path: Some(report.report_path),
        error: None,
    })
}

fn evaluate_teacher_reference<B: AutodiffBackend>(
    config: &ExperimentConfig,
    trial: ExperimentTrial,
    device: &B::Device,
) -> Result<ExperimentTrialReport> {
    let eval = evaluate_teacher_or_single_frame::<B>(config, None, device)?;
    Ok(ExperimentTrialReport {
        trial_id: trial_id(trial),
        seed: trial.seed,
        density: trial.density,
        target_density: config.target_density,
        model_variant: ExperimentModelVariant::Teacher3dReference,
        mask_policy: trial.mask_policy,
        ttt_layer_set: trial_layer_set_name(config, trial),
        ttt_encoder_layers: trial_encoder_layers(config, trial),
        ttt_predictor_layers: config.ttt_layer_set(trial).predictor_layers.clone(),
        status: ExperimentTrialStatus::Completed,
        train_final_loss: None,
        train_best_loss: None,
        eval_loss: Some(0.0),
        eval_cosine: Some(eval.teacher_self_cosine),
        eval_full_loss: Some(0.0),
        eval_full_cosine: Some(eval.teacher_self_cosine),
        pre_train_eval_loss: None,
        pre_train_eval_cosine: None,
        pre_train_full_eval_loss: None,
        pre_train_full_eval_cosine: None,
        mask_context_tokens: eval.mask_context_tokens,
        mask_target_tokens: eval.mask_target_tokens,
        mask_context_density: eval.mask_context_density,
        mask_target_density: eval.mask_target_density,
        ttt_fast_weight_bytes_f32: None,
        ttt_trainable_param_bytes_f32: None,
        elapsed_ms: eval.elapsed_ms,
        timing: ExperimentTrialTiming {
            setup_ms: eval.setup_ms,
            train_ms: 0,
            eval_ms: eval.eval_ms,
            mask_ms: eval.mask_ms,
            teacher_forward_ms: eval.teacher_forward_ms,
            student_forward_ms: eval.student_forward_ms,
            loss_ms: eval.loss_ms,
            backward_ms: 0,
            optimizer_ms: 0,
            backward_optim_ms: 0,
            teacher_cache_hits: 0,
            teacher_cache_misses: 0,
        },
        samples_per_second: Some(eval.samples_per_second),
        report_path: None,
        error: None,
    })
}

fn evaluate_single_frame<B: AutodiffBackend>(
    config: &ExperimentConfig,
    trial: ExperimentTrial,
    device: &B::Device,
) -> Result<ExperimentTrialReport> {
    let eval = evaluate_teacher_or_single_frame::<B>(config, Some(trial), device)?;
    Ok(ExperimentTrialReport {
        trial_id: trial_id(trial),
        seed: trial.seed,
        density: trial.density,
        target_density: config.target_density,
        model_variant: trial.model_variant,
        mask_policy: trial.mask_policy,
        ttt_layer_set: trial_layer_set_name(config, trial),
        ttt_encoder_layers: trial_encoder_layers(config, trial),
        ttt_predictor_layers: config.ttt_layer_set(trial).predictor_layers.clone(),
        status: ExperimentTrialStatus::Completed,
        train_final_loss: None,
        train_best_loss: None,
        eval_loss: Some(eval.loss),
        eval_cosine: Some(eval.cosine),
        eval_full_loss: Some(eval.full_loss),
        eval_full_cosine: Some(eval.full_cosine),
        pre_train_eval_loss: None,
        pre_train_eval_cosine: None,
        pre_train_full_eval_loss: None,
        pre_train_full_eval_cosine: None,
        mask_context_tokens: eval.mask_context_tokens,
        mask_target_tokens: eval.mask_target_tokens,
        mask_context_density: eval.mask_context_density,
        mask_target_density: eval.mask_target_density,
        ttt_fast_weight_bytes_f32: None,
        ttt_trainable_param_bytes_f32: None,
        elapsed_ms: eval.elapsed_ms,
        timing: ExperimentTrialTiming {
            setup_ms: eval.setup_ms,
            train_ms: 0,
            eval_ms: eval.eval_ms,
            mask_ms: eval.mask_ms,
            teacher_forward_ms: eval.teacher_forward_ms,
            student_forward_ms: eval.student_forward_ms,
            loss_ms: eval.loss_ms,
            backward_ms: 0,
            optimizer_ms: 0,
            backward_optim_ms: 0,
            teacher_cache_hits: 0,
            teacher_cache_misses: 0,
        },
        samples_per_second: Some(eval.samples_per_second),
        report_path: None,
        error: None,
    })
}

struct EvalSummary {
    loss: f64,
    cosine: f64,
    full_loss: f64,
    full_cosine: f64,
    teacher_self_cosine: f64,
    mask_context_tokens: Option<usize>,
    mask_target_tokens: Option<usize>,
    mask_context_density: Option<f32>,
    mask_target_density: Option<f32>,
    elapsed_ms: u128,
    setup_ms: u128,
    eval_ms: u128,
    mask_ms: u128,
    teacher_forward_ms: u128,
    student_forward_ms: u128,
    loss_ms: u128,
    samples_per_second: f64,
}

fn evaluate_teacher_or_single_frame<B: AutodiffBackend>(
    config: &ExperimentConfig,
    trial: Option<ExperimentTrial>,
    device: &B::Device,
) -> Result<EvalSummary> {
    let start = Instant::now();
    let teacher = load_model::<B>(&config.base, device)?.no_grad();
    let mut ttt_config = config.base.ttt.clone();
    if trial.is_some() {
        ttt_config.layer_placement = TttLayerPlacement::Explicit;
        ttt_config.layers.clear();
        ttt_config.predictor_layers.clear();
    }
    let base = load_model::<B>(&config.base, device)?;
    let student = VJepaTttModel::from_model(base, ttt_config, device)?;
    let dataset = dataset_from_config(&config.base.dataset, false)?;
    let setup_ms = start.elapsed().as_millis();
    let eval_start = Instant::now();
    let eval_steps = config.base.training.eval_steps.max(1);
    let mut total_loss = 0.0;
    let mut total_cosine = 0.0;
    let mut total_full_loss = 0.0;
    let mut total_full_cosine = 0.0;
    let mut total_teacher_cosine = 0.0;
    let mut mask_metrics = None;
    let mut mask_ms = 0;
    let mut teacher_forward_ms = 0;
    let mut student_forward_ms = 0;
    let mut loss_ms = 0;
    let mask_config = if let Some(trial) = trial {
        Some(mask_for_trial(
            &config.base,
            trial.mask_policy,
            trial.density,
            config.target_density,
            trial.seed,
        )?)
    } else {
        None
    };
    for step in 0..eval_steps {
        let batch = load_experiment_batch::<B>(
            dataset.as_ref(),
            &config.base.dataset,
            student.config(),
            device,
            step * config.base.training.batch_size,
            config.base.training.batch_size,
        )?;
        let teacher_start = Instant::now();
        let teacher_tokens = teacher
            .encode_video(batch.teacher.clone(), None)
            .tokens
            .detach();
        teacher_forward_ms += teacher_start.elapsed().as_millis();
        total_teacher_cosine +=
            cosine_from_tensors(teacher_tokens.clone(), teacher_tokens.clone())?;
        let batch_size = batch.student.shape().dims::<5>()[0];
        let [_, _, frames, height, width] = batch.student.shape().dims::<5>();
        let actual_grid = video_token_grid(student.config(), frames, height, width)?;
        let mask_start = Instant::now();
        let resolved_masks = if let Some(mask) = &mask_config {
            Some(mask.resolve_masks_with_metadata(
                &batch.student,
                student.config(),
                actual_grid,
                &batch.metadata,
            )?)
        } else {
            None
        };
        mask_ms += mask_start.elapsed().as_millis();
        if mask_metrics.is_none()
            && let Some((context_mask, target_mask)) = &resolved_masks
        {
            mask_metrics = Some(mask_metric_values(context_mask, target_mask));
        }
        let student_start = Instant::now();
        let mut state = student.fresh_state();
        let student_tokens = if trial.is_some() {
            student
                .forward_single_frame_rollout(
                    batch.student,
                    Some(teacher_tokens.clone()),
                    &mut state,
                )?
                .tokens
        } else {
            teacher_tokens.clone()
        };
        student_forward_ms += student_start.elapsed().as_millis();
        let loss_start = Instant::now();
        let full_feature_loss = (student_tokens.clone() - teacher_tokens.clone())
            .powf_scalar(2.0)
            .mean()
            .mul_scalar(config.base.loss.feature_loss_weight as f64);
        let feature_loss = feature_loss_for_masks(
            student_tokens.clone(),
            teacher_tokens.clone(),
            resolved_masks.as_ref().map(|(_, target_mask)| target_mask),
            batch_size,
            device,
            config.base.loss.feature_loss_weight,
        );
        let primary_cosine = cosine_for_masks(
            student_tokens.clone(),
            teacher_tokens.clone(),
            resolved_masks.as_ref().map(|(_, target_mask)| target_mask),
            batch_size,
            device,
        )?;
        let loss = if config.base.loss.predictor_loss_weight > 0.0
            && let Some((context_mask, target_mask)) = resolved_masks
        {
            let context_tokens = apply_token_mask(
                student_tokens.clone(),
                context_mask.to_tensor::<B>(batch_size, device),
            );
            let target_tokens = apply_token_mask(
                teacher_tokens.clone(),
                target_mask.to_tensor::<B>(batch_size, device),
            );
            let predictions = student.forward_predictor_sparse(
                context_tokens,
                &context_mask,
                &target_mask,
                actual_grid,
                0,
            )?;
            let target_tokens = if predictions.target_predictions.shape().dims::<3>()
                == target_tokens.shape().dims::<3>()
            {
                target_tokens
            } else {
                let teacher_context_tokens = apply_token_mask(
                    teacher_tokens.clone(),
                    context_mask.to_tensor::<B>(batch_size, device),
                );
                teacher
                    .predictor
                    .forward_sparse(
                        teacher_context_tokens,
                        &context_mask,
                        &target_mask,
                        actual_grid,
                        0,
                    )?
                    .target_predictions
                    .detach()
            };
            feature_loss
                + (predictions.target_predictions - target_tokens)
                    .powf_scalar(2.0)
                    .mean()
                    .mul_scalar(config.base.loss.predictor_loss_weight as f64)
        } else {
            feature_loss
        };
        total_loss += tensor_scalar(loss.detach())?;
        total_cosine += primary_cosine;
        total_full_loss += tensor_scalar(full_feature_loss.detach())?;
        total_full_cosine += cosine_from_tensors(student_tokens, teacher_tokens)?;
        loss_ms += loss_start.elapsed().as_millis();
    }
    let samples = eval_steps * config.base.training.batch_size;
    let elapsed_ms = start.elapsed().as_millis();
    let eval_ms = eval_start.elapsed().as_millis();
    let (mask_context_tokens, mask_target_tokens, mask_context_density, mask_target_density) =
        mask_metrics.unwrap_or((None, None, None, None));
    Ok(EvalSummary {
        loss: total_loss / eval_steps as f64,
        cosine: total_cosine / eval_steps as f64,
        full_loss: total_full_loss / eval_steps as f64,
        full_cosine: total_full_cosine / eval_steps as f64,
        teacher_self_cosine: total_teacher_cosine / eval_steps as f64,
        mask_context_tokens,
        mask_target_tokens,
        mask_context_density,
        mask_target_density,
        elapsed_ms,
        setup_ms,
        eval_ms,
        mask_ms,
        teacher_forward_ms,
        student_forward_ms,
        loss_ms,
        samples_per_second: samples_per_second(samples, elapsed_ms),
    })
}

fn trial_train_config(
    config: &ExperimentConfig,
    trial: ExperimentTrial,
) -> Result<BurnJepaTrainConfig> {
    let mut train_config = config.base.clone();
    config.ttt_layer_set(trial).apply_to(&mut train_config);
    train_config.training.mask = Some(mask_for_trial(
        &train_config,
        trial.mask_policy,
        trial.density,
        config.target_density,
        trial.seed,
    )?);
    Ok(train_config)
}

fn trial_layer_set_name(config: &ExperimentConfig, trial: ExperimentTrial) -> String {
    config.ttt_layer_set(trial).name.clone()
}

fn trial_encoder_layers(config: &ExperimentConfig, trial: ExperimentTrial) -> Vec<usize> {
    config
        .ttt_layer_set(trial)
        .resolved_encoder_layers(&config.base)
}

fn mask_for_trial(
    config: &BurnJepaTrainConfig,
    policy: ExperimentMaskPolicy,
    density: f32,
    target_density: f32,
    seed: u64,
) -> Result<crate::TrainingMaskConfig> {
    let model_config = model_config(config)?;
    let grid = experiment_token_grid(config, &model_config)?;
    let dense_len = grid.len().max(2);
    let context_tokens = density_tokens(dense_len, density).min(dense_len - 1);
    let target_tokens = density_tokens(dense_len, target_density).min(dense_len - context_tokens);
    Ok(match policy {
        ExperimentMaskPolicy::FullFrame => crate::TrainingMaskConfig::FullFrame { target_tokens },
        ExperimentMaskPolicy::KeepRatio => crate::TrainingMaskConfig::KeepRatio {
            context_keep_ratio: density,
        },
        ExperimentMaskPolicy::RandomSparse => crate::TrainingMaskConfig::RandomSparse {
            context_tokens,
            target_tokens,
            seed,
        },
        ExperimentMaskPolicy::PatchDiff => crate::TrainingMaskConfig::PatchDiff {
            threshold: 0.0,
            context_tokens,
            target_tokens,
            dilation: 0,
        },
        ExperimentMaskPolicy::AutogazeSparse => crate::TrainingMaskConfig::AutogazeSparse {
            image_grid: crate::TrainingImageTokenGrid::new(grid.height.max(1), grid.width.max(1)),
            context_tokens,
            target_tokens,
            source: crate::TrainingAutogazeTokenSource::default(),
            frame_tokens: None,
            dilation: 0,
        },
        ExperimentMaskPolicy::PrecomputedMasks => {
            let context = (0..context_tokens).collect::<Vec<_>>();
            let target = (context_tokens..context_tokens + target_tokens).collect::<Vec<_>>();
            crate::TrainingMaskConfig::PrecomputedMasks {
                context_indices: context,
                target_indices: target,
            }
        }
        ExperimentMaskPolicy::ManifestPrecomputedMasks => {
            crate::TrainingMaskConfig::ManifestPrecomputedMasks
        }
    })
}

fn experiment_token_grid(
    config: &BurnJepaTrainConfig,
    model_config: &VJepaConfig,
) -> Result<TokenGridShape> {
    let frames = round_up_to_multiple(
        config.dataset.frames.max(model_config.tubelet_size.max(1)),
        model_config.tubelet_size.max(1),
    );
    let image_size = round_up_to_multiple(
        config
            .dataset
            .image_size
            .max(model_config.patch_size.max(1)),
        model_config.patch_size.max(1),
    );
    video_token_grid(model_config, frames, image_size, image_size)
}

fn load_model<B: Backend>(
    config: &BurnJepaTrainConfig,
    device: &B::Device,
) -> Result<VJepa2_1Model<B>> {
    if let Some(checkpoint_dir) = config
        .model
        .teacher_checkpoint_dir
        .as_ref()
        .or(config.model.checkpoint_dir.as_ref())
    {
        let mut options = VJepaLoadOptions::default();
        if let Some(weights_name) = &config.model.weights_name {
            options.weights_name = weights_name.clone();
        }
        let (model, _, _) = options.load_model(checkpoint_dir, device)?;
        return Ok(model);
    }
    Ok(VJepa2_1Model::new(&model_config(config)?, device))
}

fn model_config(config: &BurnJepaTrainConfig) -> Result<VJepaConfig> {
    if let Some(config_path) = &config.model.config_path {
        VJepaConfig::from_json_file(config_path)
    } else if let Some(checkpoint_dir) = config
        .model
        .teacher_checkpoint_dir
        .as_ref()
        .or(config.model.checkpoint_dir.as_ref())
    {
        crate::load_config_from_hf_dir(checkpoint_dir, &VJepaLoadOptions::default().config_name)
    } else {
        Ok(VJepaConfig::tiny_for_tests())
    }
}

fn load_experiment_batch<B: Backend>(
    dataset: &dyn JepaDataset,
    dataset_config: &JepaDatasetConfig,
    model_config: &VJepaConfig,
    device: &B::Device,
    start_index: usize,
    batch_size: usize,
) -> Result<JepaTensorBatch<B>> {
    let mut students = Vec::with_capacity(batch_size);
    let mut teachers = Vec::with_capacity(batch_size);
    let mut metadata = Vec::with_capacity(batch_size);
    for offset in 0..batch_size {
        let sample = dataset.sample(start_index + offset)?;
        let batch = load_jepa_tensor_batch::<B>(&sample, dataset_config, model_config, device)?;
        students.push(batch.student);
        teachers.push(batch.teacher);
        metadata.extend(batch.metadata);
    }
    Ok(JepaTensorBatch {
        student: Tensor::cat(students, 0),
        teacher: Tensor::cat(teachers, 0),
        metadata,
    })
}

#[derive(Clone, Debug)]
struct FrameClip {
    clip_id: String,
    source: PathBuf,
    frames: Vec<PathBuf>,
}

fn collect_frame_clips(input: &Path) -> Result<Vec<FrameClip>> {
    let mut clips = Vec::new();
    let input = input.canonicalize().unwrap_or_else(|_| input.to_path_buf());
    if input.is_file() {
        bail!("single video input requires extract_videos=true and a directory input for now");
    }
    collect_frame_clips_recursive(&input, &mut clips)?;
    Ok(clips)
}

fn collect_frame_clips_recursive(input: &Path, clips: &mut Vec<FrameClip>) -> Result<()> {
    let image_files = image_files_in_dir(input)?;
    if !image_files.is_empty() {
        clips.push(FrameClip {
            clip_id: clip_id(input),
            source: input.to_path_buf(),
            frames: image_files,
        });
        return Ok(());
    }

    let mut dirs = fs::read_dir(input)
        .with_context(|| format!("read {}", input.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    dirs.retain(|path| path.is_dir());
    dirs.sort();
    for path in dirs {
        collect_frame_clips_recursive(&path, clips)?;
    }
    Ok(())
}

fn image_files_in_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(dir)
        .with_context(|| format!("read {}", dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|path| {
        path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                .unwrap_or(false)
    });
    paths.sort();
    Ok(paths)
}

fn extract_videos_to_frame_dirs(input: &Path, output_dir: &Path) -> Result<usize> {
    let videos = video_files(input)?;
    let frames_root = output_dir.join("frames");
    fs::create_dir_all(&frames_root)?;
    let mut extracted = 0;
    for video in videos {
        let clip = clip_id(&video);
        let clip_dir = frames_root.join(&clip);
        fs::create_dir_all(&clip_dir)?;
        let pattern = clip_dir.join("%08d.jpg");
        let status = Command::new("ffmpeg")
            .arg("-y")
            .arg("-i")
            .arg(&video)
            .arg("-q:v")
            .arg("2")
            .arg(&pattern)
            .status()
            .with_context(|| format!("run ffmpeg for {}", video.display()))?;
        ensure!(status.success(), "ffmpeg failed for {}", video.display());
        extracted += 1;
    }
    Ok(extracted)
}

fn video_files(input: &Path) -> Result<Vec<PathBuf>> {
    let mut videos = Vec::new();
    for entry in fs::read_dir(input).with_context(|| format!("read {}", input.display()))? {
        let path = entry?.path();
        if path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "mp4" | "mov" | "mkv" | "webm"
                    )
                })
                .unwrap_or(false)
        {
            videos.push(path);
        }
    }
    videos.sort();
    Ok(videos)
}

fn clip_domain(clip: &FrameClip, config: &ExperimentDataConfig) -> Option<String> {
    config
        .domain
        .clone()
        .or_else(|| config.domain_from_parent.then(|| parent_domain(clip)))
        .or_else(|| {
            config
                .domain_from_clip_prefix
                .then(|| clip_prefix_domain(&clip.clip_id))
        })
}

fn clip_prefix_domain(clip_id: &str) -> String {
    clip_id
        .split_once('_')
        .map(|(prefix, _)| prefix)
        .filter(|prefix| !prefix.trim().is_empty())
        .unwrap_or(clip_id)
        .to_string()
}

fn parent_domain(clip: &FrameClip) -> String {
    clip.source
        .parent()
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string()
}

fn clip_rows(
    clip: &FrameClip,
    window_frames: usize,
    window_stride: usize,
    domain: Option<String>,
) -> Vec<JepaManifestRow> {
    let window_frames = window_frames.max(1);
    let window_stride = window_stride.max(1);
    let mut rows = Vec::new();
    let mut start = 0;
    while start < clip.frames.len() {
        let end = (start + window_frames).min(clip.frames.len());
        rows.push(JepaManifestRow {
            clip_id: Some(clip.clip_id.clone()),
            domain: domain.clone(),
            start_frame: Some(start),
            fps: None,
            duration: None,
            caption: None,
            source: Some(clip.source.display().to_string()),
            image: None,
            frames: Some(clip.frames[start..end].to_vec()),
            frame_dir: None,
            teacher_frames: None,
            teacher_frame_dir: None,
            precomputed_context_indices: None,
            precomputed_target_indices: None,
            original_stream: None,
            cache_id: None,
        });
        if end == clip.frames.len() {
            break;
        }
        start += window_stride;
    }
    rows
}

fn prepare_autogaze_masks_if_configured(
    config: &ExperimentConfig,
    train_rows: &mut [JepaManifestRow],
    eval_rows: &mut [JepaManifestRow],
) -> Result<usize> {
    let Some(mask_config) = config.data.autogaze_masks.as_ref() else {
        return Ok(0);
    };
    let _ = (&train_rows, &eval_rows);
    match mask_config.backend.unwrap_or(config.base.training.backend) {
        JepaTrainBackend::NdArray => {
            #[cfg(feature = "autogaze-ndarray")]
            {
                let device = Default::default();
                prepare_autogaze_masks_for_rows::<burn::backend::NdArray<f32>>(
                    config,
                    mask_config,
                    train_rows,
                    eval_rows,
                    &device,
                )
            }
            #[cfg(not(feature = "autogaze-ndarray"))]
            {
                bail!("AutoGaze ndarray mask preparation requires the autogaze-ndarray feature")
            }
        }
        JepaTrainBackend::Flex | JepaTrainBackend::Dispatch => {
            #[cfg(feature = "autogaze-ndarray")]
            {
                let device = Default::default();
                prepare_autogaze_masks_for_rows::<burn::backend::NdArray<f32>>(
                    config,
                    mask_config,
                    train_rows,
                    eval_rows,
                    &device,
                )
            }
            #[cfg(not(feature = "autogaze-ndarray"))]
            {
                bail!(
                    "AutoGaze mask preparation for flex/dispatch training uses serialized masks and requires the autogaze-ndarray feature or an explicit AutoGaze mask backend"
                )
            }
        }
        JepaTrainBackend::Cuda => {
            #[cfg(all(feature = "autogaze-cuda", feature = "cuda"))]
            {
                let device = Default::default();
                prepare_autogaze_masks_for_rows::<burn::backend::Cuda<f32, i32>>(
                    config,
                    mask_config,
                    train_rows,
                    eval_rows,
                    &device,
                )
            }
            #[cfg(not(all(feature = "autogaze-cuda", feature = "cuda")))]
            {
                bail!("AutoGaze CUDA mask preparation requires the autogaze-cuda and cuda features")
            }
        }
        JepaTrainBackend::Wgpu => {
            #[cfg(all(feature = "autogaze-webgpu", feature = "wgpu"))]
            {
                let device = Default::default();
                prepare_autogaze_masks_for_rows::<burn::backend::Wgpu<f32, i32>>(
                    config,
                    mask_config,
                    train_rows,
                    eval_rows,
                    &device,
                )
            }
            #[cfg(not(all(feature = "autogaze-webgpu", feature = "wgpu")))]
            {
                bail!(
                    "AutoGaze WGPU mask preparation requires the autogaze-webgpu and wgpu features"
                )
            }
        }
        JepaTrainBackend::WebGpu => {
            #[cfg(all(feature = "autogaze-webgpu", feature = "webgpu"))]
            {
                let device = Default::default();
                prepare_autogaze_masks_for_rows::<burn::backend::WebGpu<f32, i32>>(
                    config,
                    mask_config,
                    train_rows,
                    eval_rows,
                    &device,
                )
            }
            #[cfg(not(all(feature = "autogaze-webgpu", feature = "webgpu")))]
            {
                bail!(
                    "AutoGaze WebGPU mask preparation requires the autogaze-webgpu and webgpu features"
                )
            }
        }
    }
}

#[cfg(feature = "autogaze")]
fn prepare_autogaze_masks_for_rows<B: Backend>(
    config: &ExperimentConfig,
    mask_config: &ExperimentAutogazeMaskConfig,
    train_rows: &mut [JepaManifestRow],
    eval_rows: &mut [JepaManifestRow],
    device: &B::Device,
) -> Result<usize> {
    use burn_autogaze::{AutoGazePipeline, AutoGazeStreamingCache};

    let model_config = model_config(&config.base)?;
    let frames = round_up_to_multiple(
        config
            .base
            .dataset
            .frames
            .max(model_config.tubelet_size.max(1)),
        model_config.tubelet_size.max(1),
    );
    let image_size = round_up_to_multiple(
        config
            .base
            .dataset
            .image_size
            .max(model_config.patch_size.max(1)),
        model_config.patch_size.max(1),
    );
    let grid = video_token_grid(&model_config, frames, image_size, image_size)?;
    let context_density = mask_config
        .context_density
        .unwrap_or_else(|| config.densities.first().copied().unwrap_or(0.2));
    let target_density = mask_config.target_density.unwrap_or(config.target_density);
    let target_tokens = density_tokens(grid.len(), target_density).max(1);
    let mut autogaze = AutoGazePipeline::<B>::from_hf_dir(&mask_config.checkpoint_dir, device)
        .with_context(|| {
            format!(
                "load AutoGaze checkpoint {}",
                mask_config.checkpoint_dir.display()
            )
        })?;
    if let Some(max_tokens) = mask_config.max_gaze_tokens_each_frame {
        autogaze = autogaze.with_max_gaze_tokens_each_frame(max_tokens);
    }
    autogaze = autogaze.with_task_loss_requirement(mask_config.task_loss_requirement);
    let plan = crate::AutogazeSparseJepaWindowConfig::new(
        frames,
        model_config.tubelet_size,
        model_config.patch_size,
        image_size,
        image_size,
        autogaze.config().num_vision_tokens_each_frame,
        context_density,
        target_tokens,
        autogaze.max_gaze_tokens_each_frame(),
    )
    .with_top_k_overfetch(mask_config.top_k_overfetch)
    .with_dilation(mask_config.dilation)
    .build()?;

    let mut rows_written = 0usize;
    let mut streaming_caches = BTreeMap::<String, AutoGazeStreamingCache<B>>::new();
    let mut streaming_starts = BTreeMap::<String, usize>::new();
    let train_rows_len = train_rows.len();
    for (row_offset, row) in train_rows
        .iter_mut()
        .chain(eval_rows.iter_mut())
        .enumerate()
    {
        if row_offset == train_rows_len {
            streaming_caches.clear();
            streaming_starts.clear();
        }
        let stream_key = autogaze_mask_stream_key(row, rows_written);
        let start_frame = row.start_frame.unwrap_or(rows_written);
        let reset_stream = streaming_starts
            .get(&stream_key)
            .is_some_and(|previous| start_frame <= *previous);
        if reset_stream {
            streaming_caches.remove(&stream_key);
        }
        streaming_starts.insert(stream_key.clone(), start_frame);
        let sample = row.to_sample(Path::new("."), config.base.dataset.sample_kind)?;
        let batch =
            load_jepa_tensor_batch::<B>(&sample, &config.base.dataset, &model_config, device)?;
        let generated = if mask_config.streaming {
            let cache = streaming_caches
                .entry(stream_key)
                .or_insert_with(|| AutoGazeStreamingCache::new(frames));
            plan.generate_streaming(&autogaze, batch.student, cache)
        } else {
            plan.generate(&autogaze, batch.student)
        };
        let masks = plan.project_generated_masks(&generated)?;
        row.precomputed_context_indices = Some(masks.context_mask.indices().to_vec());
        row.precomputed_target_indices = Some(masks.target_mask.indices().to_vec());
        rows_written += 1;
    }
    B::sync(device).context("sync AutoGaze mask preparation backend")?;
    Ok(rows_written)
}

#[cfg(feature = "autogaze")]
fn autogaze_mask_stream_key(row: &JepaManifestRow, fallback_index: usize) -> String {
    row.clip_id
        .as_ref()
        .or(row.source.as_ref())
        .map(|value| {
            format!(
                "{}:{}",
                row.domain.as_deref().unwrap_or("unknown-domain"),
                value
            )
        })
        .unwrap_or_else(|| format!("anonymous-row-{fallback_index}"))
}

#[cfg(not(feature = "autogaze"))]
#[allow(dead_code)]
fn prepare_autogaze_masks_for_rows<B: Backend>(
    _config: &ExperimentConfig,
    _mask_config: &ExperimentAutogazeMaskConfig,
    _train_rows: &mut [JepaManifestRow],
    _eval_rows: &mut [JepaManifestRow],
    _device: &B::Device,
) -> Result<usize> {
    bail!("AutoGaze mask preparation requires an autogaze-* feature")
}

fn write_jsonl(path: &Path, rows: &[JepaManifestRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = String::new();
    for row in rows {
        text.push_str(&serde_json::to_string(row)?);
        text.push('\n');
    }
    fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

fn write_trial_csv(path: &Path, trials: &[ExperimentTrialReport]) -> Result<()> {
    write_csv(path, trial_csv_columns(), trials)
}

fn write_analysis(path: &Path, report: &ExperimentRunReport) -> Result<()> {
    let mut text = format!(
        "# TTT Experiment Analysis\n\nCompleted {}/{} trials in {} ms.\n\n",
        report.completed_trials, report.trial_count, report.elapsed_ms
    );
    push_markdown_table(
        &mut text,
        "Success Criteria",
        success_criteria_columns(),
        &success_criteria_rows(&report.success_criteria),
    );
    push_markdown_table(
        &mut text,
        "By Model Variant",
        model_summary_columns(),
        &summarize_by_model(&report.trials),
    );
    push_markdown_table(
        &mut text,
        "By Mask Policy",
        mask_summary_columns(),
        &summarize_by_mask(&report.trials),
    );
    push_markdown_table(
        &mut text,
        "By TTT Layer Set",
        layer_set_summary_columns(),
        &summarize_by_layer_set(&report.trials),
    );
    push_markdown_table(
        &mut text,
        "By Model And Mask",
        model_mask_summary_columns(),
        &summarize_by_model_mask(&report.trials),
    );
    push_markdown_table(
        &mut text,
        "Matched TTT Delta",
        matched_delta_columns(),
        &matched_ttt_deltas(&report.trials),
    );
    text.push_str("\n## Verdict Gate\n\n");
    text.push_str(
        "Continue this direction only if TTT variants beat `single_frame_no_ttt` on held-out eval loss/cosine at matched data, steps, and seed. Prefer AutoGaze over patch diff only when it improves quality at matched total e2e cost. Synthetic smoke results only validate wiring; set `require_real_checkpoint=true` and `require_real_dataset=true` for a real experiment gate.\n",
    );
    fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

struct CsvColumn<T> {
    name: &'static str,
    value: fn(&T) -> String,
}

fn write_csv<T>(path: &Path, columns: Vec<CsvColumn<T>>, rows: &[T]) -> Result<()> {
    let mut text = String::new();
    text.push_str(
        &columns
            .iter()
            .map(|column| column.name)
            .collect::<Vec<_>>()
            .join(","),
    );
    text.push('\n');
    for row in rows {
        text.push_str(
            &columns
                .iter()
                .map(|column| csv_escape(&(column.value)(row)))
                .collect::<Vec<_>>()
                .join(","),
        );
        text.push('\n');
    }
    fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

fn trial_csv_columns() -> Vec<CsvColumn<ExperimentTrialReport>> {
    vec![
        CsvColumn {
            name: "trial_id",
            value: |trial| trial.trial_id.clone(),
        },
        CsvColumn {
            name: "model_variant",
            value: |trial| format!("{:?}", trial.model_variant),
        },
        CsvColumn {
            name: "mask_policy",
            value: |trial| format!("{:?}", trial.mask_policy),
        },
        CsvColumn {
            name: "ttt_layer_set",
            value: |trial| trial.ttt_layer_set.clone(),
        },
        CsvColumn {
            name: "ttt_encoder_layers",
            value: |trial| format_usize_list(&trial.ttt_encoder_layers),
        },
        CsvColumn {
            name: "ttt_predictor_layers",
            value: |trial| format_usize_list(&trial.ttt_predictor_layers),
        },
        CsvColumn {
            name: "density",
            value: |trial| format!("{:.6}", trial.density),
        },
        CsvColumn {
            name: "target_density",
            value: |trial| format!("{:.6}", trial.target_density),
        },
        CsvColumn {
            name: "seed",
            value: |trial| trial.seed.to_string(),
        },
        CsvColumn {
            name: "status",
            value: |trial| format!("{:?}", trial.status),
        },
        CsvColumn {
            name: "train_final_loss",
            value: |trial| fmt_opt8(trial.train_final_loss),
        },
        CsvColumn {
            name: "train_best_loss",
            value: |trial| fmt_opt8(trial.train_best_loss),
        },
        CsvColumn {
            name: "eval_loss",
            value: |trial| fmt_opt8(trial.eval_loss),
        },
        CsvColumn {
            name: "eval_cosine",
            value: |trial| fmt_opt8(trial.eval_cosine),
        },
        CsvColumn {
            name: "eval_full_loss",
            value: |trial| fmt_opt8(trial.eval_full_loss),
        },
        CsvColumn {
            name: "eval_full_cosine",
            value: |trial| fmt_opt8(trial.eval_full_cosine),
        },
        CsvColumn {
            name: "pre_train_eval_loss",
            value: |trial| fmt_opt8(trial.pre_train_eval_loss),
        },
        CsvColumn {
            name: "pre_train_eval_cosine",
            value: |trial| fmt_opt8(trial.pre_train_eval_cosine),
        },
        CsvColumn {
            name: "pre_train_full_eval_loss",
            value: |trial| fmt_opt8(trial.pre_train_full_eval_loss),
        },
        CsvColumn {
            name: "pre_train_full_eval_cosine",
            value: |trial| fmt_opt8(trial.pre_train_full_eval_cosine),
        },
        CsvColumn {
            name: "mask_context_tokens",
            value: |trial| fmt_opt_usize(trial.mask_context_tokens),
        },
        CsvColumn {
            name: "mask_target_tokens",
            value: |trial| fmt_opt_usize(trial.mask_target_tokens),
        },
        CsvColumn {
            name: "mask_context_density",
            value: |trial| fmt_opt8(trial.mask_context_density.map(f64::from)),
        },
        CsvColumn {
            name: "mask_target_density",
            value: |trial| fmt_opt8(trial.mask_target_density.map(f64::from)),
        },
        CsvColumn {
            name: "ttt_fast_weight_bytes_f32",
            value: |trial| fmt_opt_usize(trial.ttt_fast_weight_bytes_f32),
        },
        CsvColumn {
            name: "ttt_trainable_param_bytes_f32",
            value: |trial| fmt_opt_usize(trial.ttt_trainable_param_bytes_f32),
        },
        CsvColumn {
            name: "elapsed_ms",
            value: |trial| trial.elapsed_ms.to_string(),
        },
        CsvColumn {
            name: "setup_ms",
            value: |trial| trial.timing.setup_ms.to_string(),
        },
        CsvColumn {
            name: "train_ms",
            value: |trial| trial.timing.train_ms.to_string(),
        },
        CsvColumn {
            name: "eval_ms",
            value: |trial| trial.timing.eval_ms.to_string(),
        },
        CsvColumn {
            name: "stage_mask_ms",
            value: |trial| trial.timing.mask_ms.to_string(),
        },
        CsvColumn {
            name: "stage_teacher_forward_ms",
            value: |trial| trial.timing.teacher_forward_ms.to_string(),
        },
        CsvColumn {
            name: "stage_student_forward_ms",
            value: |trial| trial.timing.student_forward_ms.to_string(),
        },
        CsvColumn {
            name: "stage_loss_ms",
            value: |trial| trial.timing.loss_ms.to_string(),
        },
        CsvColumn {
            name: "stage_backward_optim_ms",
            value: |trial| trial.timing.backward_optim_ms.to_string(),
        },
        CsvColumn {
            name: "stage_backward_ms",
            value: |trial| trial.timing.backward_ms.to_string(),
        },
        CsvColumn {
            name: "stage_optimizer_ms",
            value: |trial| trial.timing.optimizer_ms.to_string(),
        },
        CsvColumn {
            name: "teacher_cache_hits",
            value: |trial| trial.timing.teacher_cache_hits.to_string(),
        },
        CsvColumn {
            name: "teacher_cache_misses",
            value: |trial| trial.timing.teacher_cache_misses.to_string(),
        },
        CsvColumn {
            name: "samples_per_second",
            value: |trial| fmt_opt8(trial.samples_per_second),
        },
        CsvColumn {
            name: "error",
            value: |trial| trial.error.clone().unwrap_or_default(),
        },
    ]
}

#[derive(Clone, Copy)]
enum MarkdownAlign {
    Left,
    Right,
}

struct MarkdownColumn<T> {
    title: &'static str,
    align: MarkdownAlign,
    value: Box<dyn Fn(&T) -> String>,
}

fn markdown_column<T, F>(title: &'static str, align: MarkdownAlign, value: F) -> MarkdownColumn<T>
where
    F: Fn(&T) -> String + 'static,
{
    MarkdownColumn {
        title,
        align,
        value: Box::new(value),
    }
}

fn push_markdown_table<T>(
    text: &mut String,
    title: &str,
    columns: Vec<MarkdownColumn<T>>,
    rows: &[T],
) {
    text.push_str(&format!("## {title}\n\n"));
    text.push('|');
    for column in &columns {
        text.push(' ');
        text.push_str(column.title);
        text.push_str(" |");
    }
    text.push('\n');
    text.push('|');
    for column in &columns {
        text.push_str(match column.align {
            MarkdownAlign::Left => "---",
            MarkdownAlign::Right => "---:",
        });
        text.push('|');
    }
    text.push('\n');
    for row in rows {
        text.push('|');
        for column in &columns {
            text.push(' ');
            text.push_str(&markdown_escape(&(column.value)(row)));
            text.push_str(" |");
        }
        text.push('\n');
    }
    text.push('\n');
}

#[derive(Clone, Debug)]
struct SuccessCriteriaRow {
    gate: &'static str,
    value: String,
}

fn success_criteria_rows(criteria: &ExperimentSuccessCriteria) -> Vec<SuccessCriteriaRow> {
    vec![
        SuccessCriteriaRow {
            gate: "All trials completed",
            value: criteria.all_trials_completed.to_string(),
        },
        SuccessCriteriaRow {
            gate: "Real checkpoint configured",
            value: criteria.real_checkpoint_configured.to_string(),
        },
        SuccessCriteriaRow {
            gate: "Real dataset configured",
            value: criteria.real_dataset_configured.to_string(),
        },
        SuccessCriteriaRow {
            gate: "Full model matrix",
            value: criteria.full_model_matrix.to_string(),
        },
        SuccessCriteriaRow {
            gate: "Full mask matrix",
            value: criteria.full_mask_matrix.to_string(),
        },
        SuccessCriteriaRow {
            gate: "Target-mask feature/eval enabled",
            value: criteria.mask_loss_enabled.to_string(),
        },
        SuccessCriteriaRow {
            gate: "Density count",
            value: criteria.density_count.to_string(),
        },
        SuccessCriteriaRow {
            gate: "TTT layer set count",
            value: criteria.ttt_layer_set_count.to_string(),
        },
        SuccessCriteriaRow {
            gate: "Matched TTT trials",
            value: criteria.matched_ttt_trials.to_string(),
        },
        SuccessCriteriaRow {
            gate: "TTT loss improved trials",
            value: criteria.ttt_loss_improved_trials.to_string(),
        },
        SuccessCriteriaRow {
            gate: "TTT cosine improved trials",
            value: criteria.ttt_cosine_improved_trials.to_string(),
        },
    ]
}

fn success_criteria_columns() -> Vec<MarkdownColumn<SuccessCriteriaRow>> {
    vec![
        markdown_column("Gate", MarkdownAlign::Left, |row: &SuccessCriteriaRow| {
            row.gate.to_string()
        }),
        markdown_column("Value", MarkdownAlign::Right, |row: &SuccessCriteriaRow| {
            row.value.clone()
        }),
    ]
}

#[derive(Clone, Debug)]
struct ExperimentGroupSummary {
    label: String,
    trials: usize,
    mean_eval_loss: Option<f64>,
    mean_eval_cosine: Option<f64>,
    mean_full_loss: Option<f64>,
    mean_full_cosine: Option<f64>,
    mean_context_density: Option<f64>,
    mean_target_density: Option<f64>,
    mean_samples_per_second: Option<f64>,
    mean_train_ms: Option<f64>,
    mean_eval_ms: Option<f64>,
    mean_teacher_ms: Option<f64>,
    mean_student_ms: Option<f64>,
    mean_fast_memory_mib: Option<f64>,
}

impl ExperimentGroupSummary {
    fn from_trials(label: String, trials: &[&ExperimentTrialReport]) -> Self {
        Self {
            label,
            trials: trials.len(),
            mean_eval_loss: mean_f64(trials.iter().filter_map(|trial| trial.eval_loss)),
            mean_eval_cosine: mean_f64(trials.iter().filter_map(|trial| trial.eval_cosine)),
            mean_full_loss: mean_f64(trials.iter().filter_map(|trial| trial.eval_full_loss)),
            mean_full_cosine: mean_f64(trials.iter().filter_map(|trial| trial.eval_full_cosine)),
            mean_context_density: mean_f64(
                trials
                    .iter()
                    .filter_map(|trial| trial.mask_context_density.map(f64::from)),
            ),
            mean_target_density: mean_f64(
                trials
                    .iter()
                    .filter_map(|trial| trial.mask_target_density.map(f64::from)),
            ),
            mean_samples_per_second: mean_f64(
                trials.iter().filter_map(|trial| trial.samples_per_second),
            ),
            mean_train_ms: mean_u128(trials.iter().map(|trial| trial.timing.train_ms)),
            mean_eval_ms: mean_u128(trials.iter().map(|trial| trial.timing.eval_ms)),
            mean_teacher_ms: mean_u128(trials.iter().map(|trial| trial.timing.teacher_forward_ms)),
            mean_student_ms: mean_u128(trials.iter().map(|trial| trial.timing.student_forward_ms)),
            mean_fast_memory_mib: mean_f64(
                trials
                    .iter()
                    .filter_map(|trial| trial.ttt_fast_weight_bytes_f32.map(bytes_to_mib)),
            ),
        }
    }
}

fn summarize_by_model(trials: &[ExperimentTrialReport]) -> Vec<ExperimentGroupSummary> {
    summarize_groups(trials, |trial| format!("{:?}", trial.model_variant))
}

fn summarize_by_mask(trials: &[ExperimentTrialReport]) -> Vec<ExperimentGroupSummary> {
    summarize_groups(trials, |trial| format!("{:?}", trial.mask_policy))
}

fn summarize_by_layer_set(trials: &[ExperimentTrialReport]) -> Vec<ExperimentGroupSummary> {
    summarize_groups(trials, |trial| trial.ttt_layer_set.clone())
}

fn summarize_by_model_mask(trials: &[ExperimentTrialReport]) -> Vec<ExperimentGroupSummary> {
    summarize_groups(trials, |trial| {
        format!("{:?}/{:?}", trial.model_variant, trial.mask_policy)
    })
}

fn summarize_groups(
    trials: &[ExperimentTrialReport],
    key: fn(&ExperimentTrialReport) -> String,
) -> Vec<ExperimentGroupSummary> {
    let mut groups: BTreeMap<String, Vec<&ExperimentTrialReport>> = BTreeMap::new();
    for trial in trials {
        groups.entry(key(trial)).or_default().push(trial);
    }
    groups
        .into_iter()
        .map(|(label, trials)| ExperimentGroupSummary::from_trials(label, &trials))
        .collect()
}

fn model_summary_columns() -> Vec<MarkdownColumn<ExperimentGroupSummary>> {
    vec![
        group_label_column("Model variant"),
        group_trials_column(),
        group_f64_column("Mean eval loss", |row| row.mean_eval_loss),
        group_f64_column("Mean eval cosine", |row| row.mean_eval_cosine),
        group_f64_column("Mean full loss", |row| row.mean_full_loss),
        group_f64_column("Mean full cosine", |row| row.mean_full_cosine),
        group_f64_column("Mean samples/sec", |row| row.mean_samples_per_second),
        group_f64_column("Mean train ms", |row| row.mean_train_ms),
        group_f64_column("Mean eval ms", |row| row.mean_eval_ms),
        group_f64_column("Mean teacher ms", |row| row.mean_teacher_ms),
        group_f64_column("Mean student ms", |row| row.mean_student_ms),
        group_mib_column("Mean fast memory MiB", |row| row.mean_fast_memory_mib),
    ]
}

fn mask_summary_columns() -> Vec<MarkdownColumn<ExperimentGroupSummary>> {
    vec![
        group_label_column("Mask policy"),
        group_trials_column(),
        group_f64_column("Mean eval loss", |row| row.mean_eval_loss),
        group_f64_column("Mean eval cosine", |row| row.mean_eval_cosine),
        group_f64_column("Mean full loss", |row| row.mean_full_loss),
        group_f64_column("Mean full cosine", |row| row.mean_full_cosine),
        group_f64_column("Mean context density", |row| row.mean_context_density),
        group_f64_column("Mean target density", |row| row.mean_target_density),
        group_f64_column("Mean samples/sec", |row| row.mean_samples_per_second),
        group_f64_column("Mean train ms", |row| row.mean_train_ms),
        group_f64_column("Mean eval ms", |row| row.mean_eval_ms),
    ]
}

fn layer_set_summary_columns() -> Vec<MarkdownColumn<ExperimentGroupSummary>> {
    vec![
        group_label_column("TTT layer set"),
        group_trials_column(),
        group_f64_column("Mean eval loss", |row| row.mean_eval_loss),
        group_f64_column("Mean eval cosine", |row| row.mean_eval_cosine),
        group_f64_column("Mean full loss", |row| row.mean_full_loss),
        group_f64_column("Mean full cosine", |row| row.mean_full_cosine),
        group_f64_column("Mean samples/sec", |row| row.mean_samples_per_second),
        group_f64_column("Mean train ms", |row| row.mean_train_ms),
        group_mib_column("Mean fast memory MiB", |row| row.mean_fast_memory_mib),
    ]
}

fn model_mask_summary_columns() -> Vec<MarkdownColumn<ExperimentGroupSummary>> {
    vec![
        group_label_column("Model/mask"),
        group_trials_column(),
        group_f64_column("Mean eval loss", |row| row.mean_eval_loss),
        group_f64_column("Mean eval cosine", |row| row.mean_eval_cosine),
        group_f64_column("Mean full loss", |row| row.mean_full_loss),
        group_f64_column("Mean full cosine", |row| row.mean_full_cosine),
        group_f64_column("Mean teacher ms", |row| row.mean_teacher_ms),
        group_f64_column("Mean student ms", |row| row.mean_student_ms),
        group_f64_column("Mean samples/sec", |row| row.mean_samples_per_second),
    ]
}

fn group_label_column(title: &'static str) -> MarkdownColumn<ExperimentGroupSummary> {
    markdown_column(
        title,
        MarkdownAlign::Left,
        |row: &ExperimentGroupSummary| row.label.clone(),
    )
}

fn group_trials_column() -> MarkdownColumn<ExperimentGroupSummary> {
    markdown_column(
        "Trials",
        MarkdownAlign::Right,
        |row: &ExperimentGroupSummary| row.trials.to_string(),
    )
}

fn group_f64_column(
    title: &'static str,
    value: fn(&ExperimentGroupSummary) -> Option<f64>,
) -> MarkdownColumn<ExperimentGroupSummary> {
    markdown_column(title, MarkdownAlign::Right, move |row| fmt_opt6(value(row)))
}

fn group_mib_column(
    title: &'static str,
    value: fn(&ExperimentGroupSummary) -> Option<f64>,
) -> MarkdownColumn<ExperimentGroupSummary> {
    markdown_column(title, MarkdownAlign::Right, move |row| fmt_opt3(value(row)))
}

#[derive(Clone, Debug)]
struct MatchedTttDeltaRow {
    ttt_variant: String,
    ttt_layer_set: String,
    mask: String,
    density: f32,
    seed: u64,
    eval_loss_delta: Option<f64>,
    eval_cosine_delta: Option<f64>,
    full_loss_delta: Option<f64>,
    full_cosine_delta: Option<f64>,
    train_loss_delta: Option<f64>,
    fast_memory_mib: Option<f64>,
}

fn matched_ttt_deltas(trials: &[ExperimentTrialReport]) -> Vec<MatchedTttDeltaRow> {
    trials
        .iter()
        .filter(|trial| {
            matches!(
                trial.model_variant,
                ExperimentModelVariant::TttTeacherFinal | ExperimentModelVariant::TttSelfHidden
            ) && trial.status == ExperimentTrialStatus::Completed
        })
        .filter_map(|trial| {
            let baseline = trials.iter().find(|candidate| {
                candidate.model_variant == ExperimentModelVariant::SingleFrameNoTtt
                    && candidate.mask_policy == trial.mask_policy
                    && candidate.seed == trial.seed
                    && (candidate.density - trial.density).abs() <= f32::EPSILON
                    && candidate.ttt_layer_set == trial.ttt_layer_set
                    && candidate.status == ExperimentTrialStatus::Completed
            })?;
            Some(MatchedTttDeltaRow {
                ttt_variant: format!("{:?}", trial.model_variant),
                ttt_layer_set: trial.ttt_layer_set.clone(),
                mask: format!("{:?}", trial.mask_policy),
                density: trial.density,
                seed: trial.seed,
                eval_loss_delta: opt_delta(trial.eval_loss, baseline.eval_loss),
                eval_cosine_delta: opt_delta(trial.eval_cosine, baseline.eval_cosine),
                full_loss_delta: opt_delta(trial.eval_full_loss, baseline.eval_full_loss),
                full_cosine_delta: opt_delta(trial.eval_full_cosine, baseline.eval_full_cosine),
                train_loss_delta: opt_delta(trial.eval_loss, trial.pre_train_eval_loss),
                fast_memory_mib: trial.ttt_fast_weight_bytes_f32.map(bytes_to_mib),
            })
        })
        .collect()
}

fn matched_delta_columns() -> Vec<MarkdownColumn<MatchedTttDeltaRow>> {
    vec![
        markdown_column(
            "TTT variant",
            MarkdownAlign::Left,
            |row: &MatchedTttDeltaRow| row.ttt_variant.clone(),
        ),
        markdown_column("Mask", MarkdownAlign::Left, |row: &MatchedTttDeltaRow| {
            row.mask.clone()
        }),
        markdown_column(
            "TTT layer set",
            MarkdownAlign::Left,
            |row: &MatchedTttDeltaRow| row.ttt_layer_set.clone(),
        ),
        markdown_column(
            "Density",
            MarkdownAlign::Right,
            |row: &MatchedTttDeltaRow| format!("{:.4}", row.density),
        ),
        markdown_column("Seed", MarkdownAlign::Right, |row: &MatchedTttDeltaRow| {
            row.seed.to_string()
        }),
        delta_column("Eval loss delta", |row| row.eval_loss_delta),
        delta_column("Eval cosine delta", |row| row.eval_cosine_delta),
        delta_column("Full loss delta", |row| row.full_loss_delta),
        delta_column("Full cosine delta", |row| row.full_cosine_delta),
        delta_column("Train loss delta", |row| row.train_loss_delta),
        markdown_column(
            "Fast memory MiB",
            MarkdownAlign::Right,
            |row: &MatchedTttDeltaRow| fmt_opt3(row.fast_memory_mib),
        ),
    ]
}

fn delta_column(
    title: &'static str,
    value: fn(&MatchedTttDeltaRow) -> Option<f64>,
) -> MarkdownColumn<MatchedTttDeltaRow> {
    markdown_column(title, MarkdownAlign::Right, move |row| fmt_opt6(value(row)))
}

fn trial_id(trial: ExperimentTrial) -> String {
    format!(
        "{:?}_{:?}_l{}_d{:.4}_s{}",
        trial.model_variant,
        trial.mask_policy,
        trial.ttt_layer_set_index,
        trial.density,
        trial.seed
    )
    .replace("::", "_")
    .replace('.', "p")
}

fn density_tokens(dense_len: usize, density: f32) -> usize {
    ((dense_len as f32) * density).ceil().max(1.0) as usize
}

fn round_up_to_multiple(value: usize, multiple: usize) -> usize {
    let multiple = multiple.max(1);
    value.max(1).div_ceil(multiple) * multiple
}

fn eval_every(eval_ratio: f32) -> usize {
    if eval_ratio <= 0.0 {
        usize::MAX
    } else {
        (1.0 / eval_ratio.clamp(0.01, 1.0)).round().max(1.0) as usize
    }
}

fn clip_id(path: &Path) -> String {
    path.file_stem()
        .or_else(|| path.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("clip")
        .replace(
            |ch: char| !ch.is_ascii_alphanumeric() && ch != '-' && ch != '_',
            "_",
        )
}

fn feature_loss_for_masks<B: Backend>(
    student_tokens: Tensor<B, 3>,
    teacher_tokens: Tensor<B, 3>,
    target_mask: Option<&SparseTokenMask>,
    batch_size: usize,
    device: &B::Device,
    weight: f32,
) -> Tensor<B, 1> {
    let (student_tokens, teacher_tokens) = if let Some(mask) = target_mask {
        let indices = mask.to_tensor::<B>(batch_size, device);
        (
            apply_token_mask(student_tokens, indices.clone()),
            apply_token_mask(teacher_tokens, indices),
        )
    } else {
        (student_tokens, teacher_tokens)
    };
    (student_tokens - teacher_tokens)
        .powf_scalar(2.0)
        .mean()
        .mul_scalar(weight as f64)
}

fn cosine_for_masks<B: Backend>(
    student_tokens: Tensor<B, 3>,
    teacher_tokens: Tensor<B, 3>,
    target_mask: Option<&SparseTokenMask>,
    batch_size: usize,
    device: &B::Device,
) -> Result<f64> {
    let (student_tokens, teacher_tokens) = if let Some(mask) = target_mask {
        let indices = mask.to_tensor::<B>(batch_size, device);
        (
            apply_token_mask(student_tokens, indices.clone()),
            apply_token_mask(teacher_tokens, indices),
        )
    } else {
        (student_tokens, teacher_tokens)
    };
    cosine_from_tensors(student_tokens, teacher_tokens)
}

fn mask_metric_values(
    context: &SparseTokenMask,
    target: &SparseTokenMask,
) -> (Option<usize>, Option<usize>, Option<f32>, Option<f32>) {
    let dense_tokens = context.dense_len().max(1);
    (
        Some(context.len()),
        Some(target.len()),
        Some(context.len() as f32 / dense_tokens as f32),
        Some(target.len() as f32 / dense_tokens as f32),
    )
}

fn tensor_scalar<B: Backend>(tensor: burn::tensor::Tensor<B, 1>) -> Result<f64> {
    let values = tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .context("read scalar tensor")?;
    Ok(values.first().copied().unwrap_or_default() as f64)
}

fn cosine_from_tensors<B: Backend, const D: usize>(
    left: burn::tensor::Tensor<B, D>,
    right: burn::tensor::Tensor<B, D>,
) -> Result<f64> {
    let left = left
        .into_data()
        .to_vec::<f32>()
        .context("read left tensor")?;
    let right = right
        .into_data()
        .to_vec::<f32>()
        .context("read right tensor")?;
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    for (left, right) in left.iter().zip(right.iter()) {
        let left = *left as f64;
        let right = *right as f64;
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        Ok(0.0)
    } else {
        Ok(dot / (left_norm.sqrt() * right_norm.sqrt()))
    }
}

fn samples_per_second(samples: usize, elapsed_ms: u128) -> f64 {
    if elapsed_ms == 0 {
        samples as f64
    } else {
        samples as f64 / (elapsed_ms as f64 / 1000.0)
    }
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn markdown_escape(value: &str) -> String {
    value.replace('|', "\\|")
}

fn fmt_opt8(value: Option<f64>) -> String {
    format_opt(value, 8)
}

fn fmt_opt6(value: Option<f64>) -> String {
    format_opt(value, 6)
}

fn fmt_opt3(value: Option<f64>) -> String {
    format_opt(value, 3)
}

fn format_opt(value: Option<f64>, precision: usize) -> String {
    value
        .map(|value| format!("{value:.precision$}"))
        .unwrap_or_default()
}

fn fmt_opt_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn format_usize_list(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(";")
}

fn opt_delta(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    Some(left? - right?)
}

fn mean_f64(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

fn mean_u128(values: impl Iterator<Item = u128>) -> Option<f64> {
    mean_f64(values.map(|value| value as f64))
}

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn unique_density_count(trials: &[ExperimentTrialReport]) -> usize {
    let mut densities = trials
        .iter()
        .map(|trial| (trial.density * 1_000_000.0).round() as i64)
        .collect::<Vec<_>>();
    densities.sort_unstable();
    densities.dedup();
    densities.len()
}

fn unique_layer_set_count(trials: &[ExperimentTrialReport]) -> usize {
    let mut layer_sets = trials
        .iter()
        .map(|trial| trial.ttt_layer_set.as_str())
        .collect::<Vec<_>>();
    layer_sets.sort_unstable();
    layer_sets.dedup();
    layer_sets.len()
}

fn stable_hash(value: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn git_sha() -> Result<String> {
    let output = Command::new("git").arg("rev-parse").arg("HEAD").output()?;
    ensure!(output.status.success(), "git rev-parse HEAD failed");
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
