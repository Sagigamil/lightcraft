//! Verify the Hexagon NPU encoder against the CPU backbone.
//!
//! The NPU chain (`tools/ai/qnn_chain`, used by the Android AI build) returns the same token
//! tensor the ViT produces. This compares the two numerically and runs the same clicks through
//! the unchanged neck/tracker on both, reporting the mask IoU.
//!
//! ```text
//! LIGHTCRAFT_SAM3_DIR=<model dir> NPU_TOKENS=<tokens.raw> [NPU_IMAGE=<image.raw>] \
//!   cargo run -p lightcraft-segment --release --example npu_check -- 0.5,0.5 0.2,0.7
//! ```
//!
//! `image.raw` is the preprocessed `[1, 3, 1008, 1008]` float32 tensor (the QNN runner's input,
//! produced by `mask::preprocess`); `tokens.raw` is the runner's output, float32 NHWC.

use std::path::PathBuf;

use candle_core::{Device, Tensor};
use lightcraft_segment::{Click, Sam3};

fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn main() {
    let dir = PathBuf::from(std::env::var("LIGHTCRAFT_SAM3_DIR").expect("set LIGHTCRAFT_SAM3_DIR"));
    let tokens_path = PathBuf::from(std::env::var("NPU_TOKENS").expect("set NPU_TOKENS"));
    let image_path = std::env::var("NPU_IMAGE").ok().map(PathBuf::from);
    let clicks: Vec<Click> = std::env::args()
        .skip(1)
        .map(|a| {
            let (x, y) = a.split_once(',').expect("clicks are x,y");
            Click { x: x.parse().expect("x"), y: y.parse().expect("y"), positive: !a.starts_with('-') }
        })
        .collect();

    let mut model = Sam3::load(&dir).expect("load the model");
    let tokens = read_f32(&tokens_path);
    let (grid, hidden) = (72usize, 1024usize);
    assert_eq!(tokens.len(), grid * grid * hidden, "token count");
    // NHWC -> NCHW
    let mut chw = vec![0f32; tokens.len()];
    for i in 0..grid * grid {
        for c in 0..hidden {
            chw[c * grid * grid + i] = tokens[i * hidden + c];
        }
    }
    let npu = Tensor::from_vec(chw.clone(), (1, hidden, grid, grid), &Device::Cpu).expect("tokens");
    let mut enc_npu = lightcraft_segment::Encoded::from_backbone(npu.clone());

    let mut enc_cpu = None;
    if let Some(path) = &image_path {
        let pixels = read_f32(path);
        assert_eq!(pixels.len(), 3 * 1008 * 1008, "image pixel count");
        let pixels = Tensor::from_vec(pixels, (1, 3, 1008, 1008), &Device::Cpu).expect("image");
        let t = std::time::Instant::now();
        let enc = model.encode_pixels(&pixels).expect("CPU encode");
        eprintln!("CPU backbone: {:?}", t.elapsed());
        let cpu = enc.backbone().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let a = &cpu;
        let b = &chw;
        // cosine similarity and relative error over the whole tensor
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let na: f64 = a.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| f64::from(*x) * f64::from(*x)).sum::<f64>().sqrt();
        let max_err = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let mean_err = a.iter().zip(b).map(|(x, y)| f64::from((x - y).abs())).sum::<f64>() / a.len() as f64;
        eprintln!(
            "tokens vs CPU: cosine {:.6}, max abs err {max_err:.3e}, mean abs err {mean_err:.3e}",
            dot / (na * nb)
        );
        enc_cpu = Some(enc);
    }

    if clicks.is_empty() {
        return;
    }
    let mask_of = |m: &mut Sam3, enc: &mut lightcraft_segment::Encoded| -> (Vec<f32>, f32) {
        let p = m.segment_clicks(enc, &clicks).expect("segment");
        (p.logits, p.score)
    };
    let (logits_npu, score_npu) = mask_of(&mut model, &mut enc_npu);
    eprintln!("NPU mask: score {score_npu:.4}");
    let Some(mut enc_cpu) = enc_cpu else { return };
    let (logits_cpu, score_cpu) = mask_of(&mut model, &mut enc_cpu);
    eprintln!("CPU mask: score {score_cpu:.4}");
    let inter = logits_cpu.iter().zip(&logits_npu).filter(|(a, b)| **a > 0.0 && **b > 0.0).count();
    let union = logits_cpu.iter().zip(&logits_npu).filter(|(a, b)| **a > 0.0 || **b > 0.0).count();
    let iou = inter as f64 / union.max(1) as f64;
    let agree = logits_cpu.iter().zip(&logits_npu).filter(|(a, b)| (**a > 0.0) == (**b > 0.0)).count() as f64 / logits_cpu.len() as f64;
    eprintln!("mask IoU {iou:.4}, pixel agreement {:.2} %", agree * 100.0);
}
