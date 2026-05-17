use std::cell::Cell;

use burn::backend::wgpu;
use burn::tensor::Tensor;
use js_sys::{Array, Uint8Array};
use serde::Serialize;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::{
    BurnJepaPackageModelKind, BurnJepaPipelinePackageManifest, SparseTokenMask, VJepa2_1Model,
    VJepaConfig, VJepaEncoderOutput, VJepaRgbaVideoShape, VJepaTttModel, load_ttt_burnpack_parts,
    load_vjepa_burnpack_parts, rgba_video_to_tensor,
};

type WasmBackend = burn::backend::WebGpu<f32, i32>;
type WasmDevice = wgpu::WgpuDevice;

thread_local! {
    static WEBGPU_RUNTIME_READY: Cell<bool> = const { Cell::new(false) };
}

enum WasmJepaModel {
    Base(VJepa2_1Model<WasmBackend>),
    Ttt(VJepaTttModel<WasmBackend>),
}

#[wasm_bindgen]
pub struct WasmVJepa {
    model: WasmJepaModel,
    #[allow(dead_code)]
    config: VJepaConfig,
    device: WasmDevice,
}

#[derive(Serialize)]
struct WasmEmbedSummary {
    shape: Vec<usize>,
    grid: [usize; 3],
    sample_count: usize,
    sample_mean: f32,
    sample_min: f32,
    sample_max: f32,
    sample: Vec<f32>,
}

#[wasm_bindgen]
impl WasmVJepa {
    #[wasm_bindgen]
    pub async fn create(config_json: &str) -> Result<WasmVJepa, JsValue> {
        console_error_panic_hook::set_once();
        let config: VJepaConfig = serde_json::from_str(config_json)
            .map_err(|err| js_error(format!("failed to parse V-JEPA config: {err}")))?;
        let device = webgpu_device().await;
        let model = WasmJepaModel::Base(VJepa2_1Model::<WasmBackend>::new(&config, &device));
        Ok(Self {
            model,
            config,
            device,
        })
    }

    #[wasm_bindgen(js_name = createFromBpkParts)]
    pub async fn create_from_bpk_parts(
        manifest_json: &str,
        parts: Array,
    ) -> Result<WasmVJepa, JsValue> {
        console_error_panic_hook::set_once();
        let manifest = BurnJepaPipelinePackageManifest::from_json_str(manifest_json)
            .map_err(|err| js_error(format!("failed to parse package manifest: {err:#}")))?;
        let parts = js_array_to_parts(parts)?;
        Self::create_from_manifest_and_parts(manifest, &parts).await
    }

    #[wasm_bindgen(js_name = createFromManifestUrl)]
    pub async fn create_from_manifest_url(manifest_url: &str) -> Result<WasmVJepa, JsValue> {
        console_error_panic_hook::set_once();
        let manifest_bytes = fetch_url_bytes(manifest_url).await?;
        let manifest: BurnJepaPipelinePackageManifest =
            serde_json::from_slice(manifest_bytes.as_slice()).map_err(|err| {
                js_error(format!(
                    "failed to parse burn_jepa manifest {manifest_url}: {err}"
                ))
            })?;
        let parts_manifest_url = join_url(
            manifest_url_parent(manifest_url).as_str(),
            manifest.parts_manifest.as_str(),
        );
        let part_bytes = fetch_parts_bundle(parts_manifest_url.as_str()).await?;
        Self::create_from_manifest_and_parts(manifest, &part_bytes).await
    }

    #[wasm_bindgen]
    pub fn embed_rgba_shape(
        &self,
        rgba: &[u8],
        batch: usize,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<usize>, JsValue> {
        let out = self.embed_rgba(rgba, batch, frames, height, width)?;
        Ok(out.tokens.shape().dims::<3>().to_vec())
    }

    #[wasm_bindgen(js_name = embedRgbaSummaryJson)]
    pub async fn embed_rgba_summary_json(
        &self,
        rgba: &[u8],
        batch: usize,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<String, JsValue> {
        let out = self.embed_rgba(rgba, batch, frames, height, width)?;
        let [batch, tokens, dim] = out.tokens.shape().dims::<3>();
        let sample_tokens = tokens.min(4);
        let sample_dim = dim.min(8);
        let sample = out
            .tokens
            .slice([0..batch.min(1), 0..sample_tokens, 0..sample_dim]);
        let values = sample
            .into_data_async()
            .await
            .map_err(|err| js_error(format!("failed to read token sample: {err:?}")))?
            .to_vec::<f32>()
            .map_err(|err| js_error(format!("failed to read token sample: {err:?}")))?;
        let (sample_min, sample_max, sample_mean) = summarize_values(&values);
        let summary = WasmEmbedSummary {
            shape: vec![batch, tokens, dim],
            grid: [out.grid.depth, out.grid.height, out.grid.width],
            sample_count: values.len(),
            sample_mean,
            sample_min,
            sample_max,
            sample: values,
        };
        serde_json::to_string(&summary)
            .map_err(|err| js_error(format!("failed to serialize summary: {err}")))
    }
}

impl WasmVJepa {
    async fn create_from_manifest_and_parts(
        mut manifest: BurnJepaPipelinePackageManifest,
        parts: &[Vec<u8>],
    ) -> Result<WasmVJepa, JsValue> {
        let device = webgpu_device().await;
        let config = manifest.jepa_config.clone();
        let model = match manifest.model_kind {
            BurnJepaPackageModelKind::Base => {
                let (model, result) =
                    load_vjepa_burnpack_parts::<WasmBackend>(&config, parts, &device).map_err(
                        |err| js_error(format!("failed to load V-JEPA burnpack parts: {err:#}")),
                    )?;
                ensure_no_apply_errors(result)?;
                WasmJepaModel::Base(model)
            }
            BurnJepaPackageModelKind::Ttt => {
                let ttt_config = manifest.ttt_config.take().ok_or_else(|| {
                    js_error("TTT burn_jepa package manifest is missing ttt_config")
                })?;
                let (model, result) =
                    load_ttt_burnpack_parts::<WasmBackend>(&config, ttt_config, parts, &device)
                        .map_err(|err| {
                            js_error(format!("failed to load TTT V-JEPA burnpack parts: {err:#}"))
                        })?;
                ensure_no_apply_errors(result)?;
                WasmJepaModel::Ttt(model)
            }
        };
        Ok(Self {
            model,
            config,
            device,
        })
    }

    fn embed_rgba(
        &self,
        rgba: &[u8],
        batch: usize,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Result<VJepaEncoderOutput<WasmBackend>, JsValue> {
        let video = rgba_video_to_tensor::<WasmBackend>(
            rgba,
            VJepaRgbaVideoShape::new(batch, frames, height, width),
            &self.device,
        )
        .map_err(|err| js_error(format!("failed to build V-JEPA input tensor: {err:#}")))?;
        self.encode_video(video, None)
    }

    fn encode_video(
        &self,
        video: Tensor<WasmBackend, 5>,
        mask: Option<&SparseTokenMask>,
    ) -> Result<VJepaEncoderOutput<WasmBackend>, JsValue> {
        match &self.model {
            WasmJepaModel::Base(model) => Ok(model.encode_video(video, mask)),
            WasmJepaModel::Ttt(model) => model
                .encode_video(video, mask)
                .map_err(|err| js_error(format!("failed to run TTT V-JEPA: {err:#}"))),
        }
    }
}

async fn webgpu_device() -> WasmDevice {
    let device = WasmDevice::default();
    if WEBGPU_RUNTIME_READY.with(Cell::get) {
        return device;
    }
    wgpu::init_setup_async::<wgpu::graphics::WebGpu>(&device, Default::default()).await;
    WEBGPU_RUNTIME_READY.with(|flag| flag.set(true));
    device
}

async fn fetch_parts_bundle(parts_manifest_url: &str) -> Result<Vec<Vec<u8>>, JsValue> {
    let manifest_bytes = fetch_url_bytes(parts_manifest_url).await?;
    let manifest: crate::BurnpackPartsManifest = serde_json::from_slice(manifest_bytes.as_slice())
        .map_err(|err| {
            js_error(format!(
                "failed to parse burnpack parts manifest {parts_manifest_url}: {err}"
            ))
        })?;
    if manifest.parts.is_empty() {
        return Err(js_error(format!(
            "burnpack parts manifest is empty: {parts_manifest_url}"
        )));
    }
    let parent = manifest_url_parent(parts_manifest_url);
    let mut parts = Vec::with_capacity(manifest.parts.len());
    for part in manifest.parts {
        let url = join_url(parent.as_str(), part.path.as_str());
        parts.push(fetch_url_bytes(url.as_str()).await?);
    }
    Ok(parts)
}

async fn fetch_url_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().ok_or_else(|| js_error("window is unavailable"))?;
    let response_value = JsFuture::from(window.fetch_with_str(url))
        .await
        .map_err(|err| js_error(format!("fetch failed for {url}: {err:?}")))?;
    let response: web_sys::Response = response_value
        .dyn_into()
        .map_err(|_| js_error(format!("fetch returned non-response value for {url}")))?;
    if !response.ok() {
        return Err(js_error(format!(
            "HTTP {} while fetching {url}",
            response.status()
        )));
    }
    let buffer =
        JsFuture::from(response.array_buffer().map_err(|err| {
            js_error(format!("failed to read response bytes for {url}: {err:?}"))
        })?)
        .await
        .map_err(|err| js_error(format!("arrayBuffer failed for {url}: {err:?}")))?;
    Ok(Uint8Array::new(&buffer).to_vec())
}

fn js_array_to_parts(parts: Array) -> Result<Vec<Vec<u8>>, JsValue> {
    let mut out = Vec::with_capacity(parts.length() as usize);
    for value in parts.iter() {
        out.push(Uint8Array::new(&value).to_vec());
    }
    if out.is_empty() {
        return Err(js_error("burnpack parts array is empty"));
    }
    Ok(out)
}

fn ensure_no_apply_errors(result: burn_store::ApplyResult) -> Result<(), JsValue> {
    if result.errors.is_empty() {
        return Ok(());
    }
    Err(js_error(format!(
        "burnpack load reported tensor errors: {:?}",
        result.errors
    )))
}

fn summarize_values(values: &[f32]) -> (f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0;
    for value in values {
        min = min.min(*value);
        max = max.max(*value);
        sum += *value;
    }
    (min, max, sum / values.len() as f32)
}

fn manifest_url_parent(url: &str) -> String {
    url.rsplit_once('/')
        .map(|(parent, _)| parent.to_string())
        .unwrap_or_else(|| ".".to_string())
}

fn join_url(base: &str, child: &str) -> String {
    if child.starts_with("http://") || child.starts_with("https://") {
        return child.to_string();
    }
    let left = base.trim_end_matches('/');
    let right = child.trim_start_matches('/');
    if left.is_empty() {
        return format!("/{right}");
    }
    format!("{left}/{right}")
}

fn js_error(message: impl AsRef<str>) -> JsValue {
    JsValue::from_str(message.as_ref())
}
