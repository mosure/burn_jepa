use crate::TokenGridShape;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparsePosition {
    pub frame: f32,
    pub row: f32,
    pub col: f32,
}

pub fn token_index_to_coords(index: usize, grid: TokenGridShape) -> (usize, usize, usize) {
    let tokens_per_frame = grid.tokens_per_frame().max(1);
    let frame = index / tokens_per_frame;
    let rem = index - frame * tokens_per_frame;
    let row = rem / grid.width.max(1);
    let col = rem - row * grid.width.max(1);
    (frame, row, col)
}

pub fn coords_to_token_index(frame: usize, row: usize, col: usize, grid: TokenGridShape) -> usize {
    (frame * grid.tokens_per_frame()) + (row * grid.width) + col
}

pub fn get_1d_sincos_pos_embed(embed_dim: usize, grid_size: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(grid_size * embed_dim);
    for pos in 0..grid_size {
        out.extend(one_position_sincos(embed_dim, pos as f32));
    }
    out
}

pub fn get_2d_sincos_pos_embed(embed_dim: usize, height: usize, width: usize) -> Vec<f32> {
    let h_dim = embed_dim / 2;
    let w_dim = embed_dim - h_dim;
    let mut out = Vec::with_capacity(height * width * embed_dim);
    for row in 0..height {
        for col in 0..width {
            out.extend(one_position_sincos(h_dim, row as f32));
            out.extend(one_position_sincos(w_dim, col as f32));
        }
    }
    out
}

pub fn get_3d_sincos_pos_embed(embed_dim: usize, grid: TokenGridShape) -> Vec<f32> {
    sparse_3d_sincos_pos_embed(embed_dim, grid, &(0..grid.len()).collect::<Vec<_>>())
}

pub fn sparse_3d_sincos_pos_embed(
    embed_dim: usize,
    grid: TokenGridShape,
    indices: &[usize],
) -> Vec<f32> {
    let d_dim = even_dim(embed_dim / 2);
    let h_dim = even_dim(embed_dim / 4);
    let w_dim = embed_dim.saturating_sub(d_dim + h_dim);
    let w_dim = even_dim(w_dim);
    let rem = embed_dim.saturating_sub(d_dim + h_dim + w_dim);
    let mut out = Vec::with_capacity(indices.len() * embed_dim);
    for &index in indices {
        let (frame, row, col) = token_index_to_coords(index, grid);
        out.extend(one_position_sincos(d_dim, frame as f32));
        out.extend(one_position_sincos(h_dim, row as f32));
        out.extend(one_position_sincos(w_dim, col as f32));
        out.extend(std::iter::repeat_n(0.0, rem));
    }
    out
}

fn even_dim(dim: usize) -> usize {
    dim - (dim % 2)
}

fn one_position_sincos(embed_dim: usize, pos: f32) -> Vec<f32> {
    if embed_dim == 0 {
        return Vec::new();
    }
    let half = embed_dim / 2;
    let mut values = Vec::with_capacity(embed_dim);
    for i in 0..half {
        let omega = 1.0 / 10000_f32.powf(i as f32 / half.max(1) as f32);
        values.push((pos * omega).sin());
    }
    for i in 0..half {
        let omega = 1.0 / 10000_f32.powf(i as f32 / half.max(1) as f32);
        values.push((pos * omega).cos());
    }
    if embed_dim % 2 == 1 {
        values.push(0.0);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_index_round_trip_uses_depth_height_width_order() {
        let grid = TokenGridShape::new(3, 4, 5);
        let index = coords_to_token_index(2, 3, 4, grid);
        assert_eq!(index, 59);
        assert_eq!(token_index_to_coords(index, grid), (2, 3, 4));
    }

    #[test]
    fn sparse_positional_encoding_has_expected_shape() {
        let grid = TokenGridShape::new(2, 2, 2);
        let values = sparse_3d_sincos_pos_embed(10, grid, &[0, 7]);
        assert_eq!(values.len(), 20);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[3], 1.0);
    }
}
