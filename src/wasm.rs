use crate::{VJepaConfig, VJepaPipeline, VJepaRgbaVideoShape, rgba_video_to_tensor};
use burn::backend::{WebGpu, wgpu};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmVJepa {
    pipeline: VJepaPipeline<WebGpu>,
    device: wgpu::WgpuDevice,
}

#[wasm_bindgen]
impl WasmVJepa {
    #[wasm_bindgen]
    pub async fn create(config_json: &str) -> Result<WasmVJepa, JsValue> {
        console_error_panic_hook::set_once();
        let config: VJepaConfig =
            serde_json::from_str(config_json).map_err(|err| JsValue::from_str(&err.to_string()))?;
        let device = wgpu::WgpuDevice::default();
        wgpu::init_setup::<wgpu::graphics::AutoGraphicsApi>(&device, Default::default());
        let pipeline = VJepaPipeline::<WebGpu>::random(config, &device);
        Ok(Self { pipeline, device })
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
        let video = rgba_video_to_tensor::<WebGpu>(
            rgba,
            VJepaRgbaVideoShape::new(batch, frames, height, width),
            &self.device,
        )
        .map_err(|err| JsValue::from_str(&err.to_string()))?;
        let out = self.pipeline.embed_video(video);
        Ok(out.tokens.shape().dims::<3>().to_vec())
    }
}
