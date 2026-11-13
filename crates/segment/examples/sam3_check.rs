//! Smoke-test a SAM 3 checkpoint directory: load it, encode a synthetic image, and run one
//! point prompt. Exits non-zero on any error or invalid output.
//!
//! `cargo run -p lightcraft-segment --release --example sam3_check -- <model dir>`

use std::path::Path;
use std::time::Instant;

use lightcraft_segment::{Click, Sam3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).ok_or("usage: sam3_check <model dir>")?;
    let dir = Path::new(&dir);
    let t = Instant::now();
    let mut model = Sam3::load(dir)?;
    eprintln!("loaded in {:?} on {:?}", t.elapsed(), model.device());

    let (w, h) = (1008usize, 1008usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            rgb[i] = (x % 256) as u8;
            rgb[i + 1] = (y % 256) as u8;
            rgb[i + 2] = 128;
        }
    }
    let t = Instant::now();
    let mut encoded = model.encode(&rgb, w, h)?;
    eprintln!("encoded in {:?}", t.elapsed());
    let t = Instant::now();
    let p = model.segment_clicks(&mut encoded, &[Click { x: 0.5, y: 0.5, positive: true }])?;
    eprintln!("click in {:?}", t.elapsed());

    let side = lightcraft_segment::MASK_SIDE;
    if p.logits.len() != side * side {
        return Err(format!("mask has {} logits, expected {}", p.logits.len(), side * side).into());
    }
    if !p.logits.iter().all(|v| v.is_finite()) {
        return Err("mask contains non-finite logits".into());
    }
    let positive = p.logits.iter().filter(|v| **v > 0.0).count();
    if positive == 0 {
        return Err("mask is empty".into());
    }
    println!("OK: score={:.4}, {positive}/{} pixels positive", p.score, p.logits.len());
    Ok(())
}
