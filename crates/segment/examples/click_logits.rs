// Temporary: SAM 3 mask logits for one click, for comparing with MobileSAM.

use candle_core::{Device, Tensor};
use lightcraft_segment::{Click, Sam3};
use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(std::env::var("LIGHTCRAFT_SAM3_DIR").expect("set LIGHTCRAFT_SAM3_DIR"));
    let pixels_path = PathBuf::from(std::env::var("SAM3_PIXELS").expect("set SAM3_PIXELS"));
    let out = PathBuf::from(std::env::var("SAM3_LOGITS").expect("set SAM3_LOGITS"));
    let click: Vec<f32> = std::env::var("SAM3_CLICK")
        .expect("set SAM3_CLICK=x,y")
        .split(',')
        .map(|v| v.parse().expect("click"))
        .collect();
    let bytes = std::fs::read(&pixels_path).expect("read pixels");
    let raw: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    assert_eq!(raw.len(), 3 * 1008 * 1008);
    let pixels = Tensor::from_vec(raw, (1, 3, 1008, 1008), &Device::Cpu).expect("pixels");

    let mut model = Sam3::load(&dir).expect("load");
    let mut enc = model.encode_pixels(&pixels).expect("encode");
    let p = model
        .segment_clicks(&mut enc, &[Click { x: click[0], y: click[1], positive: true }])
        .expect("segment");
    let bytes: Vec<u8> = p.logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&out, bytes).expect("write");
    eprintln!("score {} logits {}", p.score, p.logits.len());
}
