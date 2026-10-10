//! Load a `.nam` file, run a synthetic DI through both sizes and print the
//! output level. Usage: `cargo run --release --example process -- amp.nam`
use robine_nam::{NamFile, Size};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: process <model.nam>")?;
    let file = NamFile::read(&path)?;
    println!(
        "{} | {} | {} Hz | sizes: {}",
        file.metadata.display_name("unnamed"),
        file.architecture,
        file.sample_rate,
        file.sizes()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    if let Some(gain) = file.metadata.normalization_gain_db(-18.0) {
        println!("normalization to -18 dBFS: {gain:+.2} dB");
    }

    let input: Vec<f32> = (0..48_000)
        .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 110.0 * i as f32 / 48_000.0).sin())
        .collect();
    for size in [Size::Small, Size::Large] {
        let Some(_) = file.variant(size) else {
            continue;
        };
        let mut model = file.build(size)?;
        let mut output = vec![0.0; input.len()];
        model.process(&input, &mut output);
        let rms = (output.iter().map(|v| v * v).sum::<f32>() / output.len() as f32).sqrt();
        println!("{size}: rms {:.2} dBFS", 20.0 * rms.max(1e-9).log10());
    }
    Ok(())
}
