use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QuantizationMode {
    #[default]
    None,
    SymmetricInt8,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedTensorData {
    pub values: Vec<i8>,
    pub scale: f32,
    pub shape: Vec<usize>,
}

pub fn symmetric_quantize(values: &[f32], shape: &[usize]) -> QuantizedTensorData {
    let max_abs = values
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let values = values
        .iter()
        .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    QuantizedTensorData {
        values,
        scale,
        shape: shape.to_vec(),
    }
}

pub fn symmetric_dequantize(data: &QuantizedTensorData) -> Result<Vec<f32>> {
    ensure!(
        data.scale.is_finite() && data.scale > 0.0,
        "invalid quantization scale"
    );
    Ok(data
        .values
        .iter()
        .map(|value| *value as f32 * data.scale)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_int8_roundtrip_stays_close() {
        let values = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let q = symmetric_quantize(&values, &[5]);
        assert_eq!(q.values, vec![-127, -64, 0, 64, 127]);
        let restored = symmetric_dequantize(&q).expect("dequantize");
        for (a, b) in values.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }
}
