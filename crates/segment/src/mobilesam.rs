//! MobileSAM (ViT-T + point prompt decoder) driven through an external accelerator.
//!
//! The image encoder runs on the Hexagon NPU and the small prompt decoder on the QNN CPU backend
//! (the HTP decoder is numerically unreliable for this model). This module owns the parts that are
//! independent of the runtime: MobileSAM's preprocessing (resize the longest side to 1024 and pad
//! with the SAM pixel mean), prompt coordinates, and mapping the 256 x 256 mask back to the photo.
//!
//! The model sees the photo with its aspect ratio preserved (and padded), unlike SAM 3's stretch,
//! so the mask is cropped to the content region when it is mapped to LightCraft's square grid.

use std::sync::Arc;

use crate::{Click, Error, Prediction, Result};

/// Side of the encoder input.
pub const IMAGE: usize = 1024;
/// Side of the low-resolution mask the decoder produces.
pub const MASK: usize = 256;
/// Prompt slots the decoder was exported with (unused slots are padded with label −1).
pub const MAX_POINTS: usize = 8;
/// Side of the square logits grid handed to [`crate::SegMask`] (matches SAM 3's mask side).
pub const GRID: usize = crate::MASK_SIDE;
/// SAM's pixel mean in 0..1 (RGB); the padding is filled with it so the encoder sees a zero there.
const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// The external accelerator: the encoder on the NPU and the prompt decoder on the CPU backend.
pub trait MobileSamRuntime: Send + Sync {
    /// `image`: 1024 × 1024 × 3 float32, NHWC, 0..1 (mean/std are inside the model).
    /// Returns the 64 × 64 × 256 embedding tensor, NHWC.
    fn encode(&self, image: &[f32]) -> std::result::Result<Vec<f32>, String>;

    /// `embeddings`: the encoder output. `points`: up to [`MAX_POINTS`] `(x, y, positive)` in the
    /// 1024 frame. Returns the 256 × 256 mask logits and the model's IoU logit.
    fn decode(&self, embeddings: &[f32], points: &[(f32, f32, bool)]) -> std::result::Result<Vec<(Vec<f32>, f32)>, String>;
}

/// A MobileSAM model backed by an external runtime.
pub struct MobileSam {
    runtime: Arc<dyn MobileSamRuntime>,
}

/// One photo after the encoder: the embeddings plus where the photo sits in the 1024 frame.
pub struct Encoded {
    embeddings: Vec<f32>,
    /// Content size inside the 1024 × 1024 encoder input (the rest is padding).
    width: usize,
    height: usize,
}

impl MobileSam {
    pub fn new(runtime: Arc<dyn MobileSamRuntime>) -> Self {
        Self { runtime }
    }

    /// Resize the photo to MobileSAM's input, encode it and keep the embeddings for clicks.
    pub fn encode(&self, rgb: &[u8], w: usize, h: usize) -> Result<Encoded> {
        let (width, height) = target_size(w, h);
        if w == 0 || h == 0 || rgb.len() < w * h * 3 {
            return Err(Error::Model(format!("image buffer of {} bytes for {w} x {h}", rgb.len())));
        }
        let image = padded_input(rgb, w, h, width, height);
        let embeddings = self.runtime.encode(&image).map_err(Error::Model)?;
        if embeddings.len() != 64 * 64 * 256 {
            return Err(Error::Model(format!("the MobileSAM encoder returned {} values, expected {}", embeddings.len(), 64 * 64 * 256)));
        }
        Ok(Encoded { embeddings, width, height })
    }

    /// Segment from clicks (normalized to the photo; the first [`MAX_POINTS`] are used).
    ///
    /// Returns the decoder's mask alternatives (the single-mask token is dropped, leaving SAM's
    /// three multimask outputs); the caller picks one, usually the highest score.
    pub fn segment_clicks(&self, enc: &Encoded, clicks: &[Click]) -> Result<Vec<Prediction>> {
        if !clicks.iter().any(|c| c.positive) {
            return Err(Error::Model("no positive click".into()));
        }
        let used: Vec<&Click> = clicks.iter().take(MAX_POINTS).collect();
        let points: Vec<(f32, f32, bool)> =
            used.iter().map(|c| (c.x.clamp(0.0, 1.0) * enc.width as f32, c.y.clamp(0.0, 1.0) * enc.height as f32, c.positive)).collect();
        let decoded = self.runtime.decode(&enc.embeddings, &points).map_err(Error::Model)?;
        if decoded.is_empty() {
            return Err(Error::Model("the MobileSAM decoder returned no masks".into()));
        }
        // The first output is SAM's single-mask token; the rest are the multimask alternatives.
        let skip = if decoded.len() > 1 { 1 } else { 0 };
        let mut out = Vec::with_capacity(decoded.len() - skip);
        for (mask, score) in decoded.iter().skip(skip) {
            if mask.len() != MASK * MASK {
                return Err(Error::Model(format!("the MobileSAM decoder returned {} values, expected {}", mask.len(), MASK * MASK)));
            }
            out.push(Prediction { logits: to_grid(mask, enc.width, enc.height), score: sigmoid(*score) });
        }
        Ok(out)
    }

    /// The embeddings (tests).
    pub fn embeddings<'a>(&self, enc: &'a Encoded) -> &'a [f32] {
        &enc.embeddings
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Content size after resizing the longest side to [`IMAGE`].
fn target_size(w: usize, h: usize) -> (usize, usize) {
    let scale = IMAGE as f64 / w.max(h).max(1) as f64;
    (((w as f64 * scale).round() as usize).clamp(1, IMAGE), ((h as f64 * scale).round() as usize).clamp(1, IMAGE))
}

/// Bilinear resize (align_corners = false) to `rw × rh`, 0..1, and pad to 1024 × 1024 with the
/// pixel mean, NHWC.
fn padded_input(rgb: &[u8], w: usize, h: usize, rw: usize, rh: usize) -> Vec<f32> {
    let mut out = vec![0f32; IMAGE * IMAGE * 3];
    for y in 0..IMAGE {
        for x in 0..IMAGE {
            let o = (y * IMAGE + x) * 3;
            out[o..o + 3].copy_from_slice(&PIXEL_MEAN);
        }
    }
    let sx = w as f64 / rw as f64;
    let sy = h as f64 / rh as f64;
    for y in 0..rh {
        let fy = ((y as f64 + 0.5) * sy - 0.5).clamp(0.0, (h - 1) as f64);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(h - 1);
        let wy = (fy - y0 as f64) as f32;
        for x in 0..rw {
            let fx = ((x as f64 + 0.5) * sx - 0.5).clamp(0.0, (w - 1) as f64);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let wx = (fx - x0 as f64) as f32;
            let o = (y * IMAGE + x) * 3;
            for c in 0..3 {
                let p00 = f32::from(rgb[(y0 * w + x0) * 3 + c]);
                let p10 = f32::from(rgb[(y0 * w + x1) * 3 + c]);
                let p01 = f32::from(rgb[(y1 * w + x0) * 3 + c]);
                let p11 = f32::from(rgb[(y1 * w + x1) * 3 + c]);
                let top = p00 + (p10 - p00) * wx;
                let bottom = p01 + (p11 - p01) * wx;
                out[o + c] = (top + (bottom - top) * wy) / 255.0;
            }
        }
    }
    out
}

/// Sample the 256 × 256 mask over the content region into a [`GRID`] × [`GRID`] grid covering the
/// photo, with bilinear interpolation (edges clamped).
fn to_grid(mask: &[f32], content_w: usize, content_h: usize) -> Vec<f32> {
    let mut out = vec![0f32; GRID * GRID];
    // Content region in the 256 mask's pixel coordinates.
    let mx = content_w as f32 / IMAGE as f32 * MASK as f32;
    let my = content_h as f32 / IMAGE as f32 * MASK as f32;
    for gy in 0..GRID {
        for gx in 0..GRID {
            let u = ((gx as f32 + 0.5) / GRID as f32 * mx - 0.5).clamp(0.0, (MASK - 1) as f32);
            let v = ((gy as f32 + 0.5) / GRID as f32 * my - 0.5).clamp(0.0, (MASK - 1) as f32);
            let x0 = u.floor() as usize;
            let y0 = v.floor() as usize;
            let x1 = (x0 + 1).min(MASK - 1);
            let y1 = (y0 + 1).min(MASK - 1);
            let wx = u - x0 as f32;
            let wy = v - y0 as f32;
            let top = mask[y0 * MASK + x0] * (1.0 - wx) + mask[y0 * MASK + x1] * wx;
            let bottom = mask[y1 * MASK + x0] * (1.0 - wx) + mask[y1 * MASK + x1] * wx;
            out[gy * GRID + gx] = top * (1.0 - wy) + bottom * wy;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocessing_pads_with_the_pixel_mean_and_content_size_follows_the_aspect() {
        let rgb = vec![255u8; 64 * 32 * 3];
        let input = padded_input(&rgb, 64, 32, 1024, 512);
        assert_eq!(input.len(), 1024 * 1024 * 3);
        // top-left is the white image, the bottom half is the mean padding
        assert!((input[0] - 1.0).abs() < 1e-6);
        let pad = (700 * 1024 + 10) * 3;
        assert!((input[pad] - PIXEL_MEAN[0]).abs() < 1e-6);
        assert_eq!(target_size(2000, 1000), (1024, 512));
        assert_eq!(target_size(500, 1000), (512, 1024));
    }

    #[test]
    fn grid_keeps_the_mask_upright_and_bounded() {
        let mut mask = vec![-10f32; MASK * MASK];
        for y in 0..MASK {
            for x in 0..MASK / 2 {
                mask[y * MASK + x] = 10.0;
            }
        }
        let grid = to_grid(&mask, 1024, 1024);
        assert_eq!(grid.len(), GRID * GRID);
        assert!(grid[GRID * GRID / 2] > 0.0);
        assert!(grid[GRID * GRID - 1] < 0.0);
    }

    struct MockRuntime {
        image: std::sync::Mutex<Vec<f32>>,
        points: std::sync::Mutex<Vec<(f32, f32, bool)>>,
    }

    impl MobileSamRuntime for MockRuntime {
        fn encode(&self, image: &[f32]) -> std::result::Result<Vec<f32>, String> {
            *self.image.lock().unwrap() = image.to_vec();
            Ok(vec![0.0; 64 * 64 * 256])
        }

        fn decode(
            &self,
            _embeddings: &[f32],
            points: &[(f32, f32, bool)],
        ) -> std::result::Result<Vec<(Vec<f32>, f32)>, String> {
            *self.points.lock().unwrap() = points.to_vec();
            Ok((0..4).map(|i| (vec![i as f32; MASK * MASK], i as f32)).collect())
        }
    }

    #[test]
    fn letterboxes_a_wide_photo_and_scales_prompts_to_the_content() {
        let mock = std::sync::Arc::new(MockRuntime {
            image: std::sync::Mutex::new(Vec::new()),
            points: std::sync::Mutex::new(Vec::new()),
        });
        let model = MobileSam::new(mock.clone());
        let rgb = vec![255u8; 200 * 100 * 3];
        let enc = model.encode(&rgb, 200, 100).unwrap();
        {
            let image = mock.image.lock().unwrap();
            assert_eq!(image.len(), IMAGE * IMAGE * 3);
            // the content is white and sits at the top; the bottom half is the pixel-mean pad
            assert!((image[0] - 1.0).abs() < 1e-6);
            let pad = (600 * IMAGE + 10) * 3;
            assert!((image[pad] - PIXEL_MEAN[0]).abs() < 1e-6);
        }
        let predictions = model.segment_clicks(&enc, &[Click { x: 0.5, y: 0.25, positive: true }]).unwrap();
        // four outputs, the first is the single-mask token: three alternatives remain
        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0].logits.len(), GRID * GRID);
        let points = mock.points.lock().unwrap();
        let (x, y, positive) = points[0];
        assert!((x - 512.0).abs() < 1e-3, "x {x}");
        assert!((y - 128.0).abs() < 1e-3, "y {y}");
        assert!(positive);
    }
}
