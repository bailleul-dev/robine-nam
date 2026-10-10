//! # robine-nam
//!
//! A small, allocation-free, audio-driver-agnostic runtime for
//! [Neural Amp Modeler](https://www.neuralampmodeler.com) **A2** ("nam2")
//! models. It deliberately supports only the two A2 variants:
//!
//! * [`Size::Small`]: 3 channels (A2-Lite / "nano"),
//! * [`Size::Large`]: 8 channels (A2-Full / "standard"),
//!
//! either as a bare `WaveNet` file or packed in a `SlimmableContainer`
//! (the format TONE3000 and the trainer produce). Anything else (LSTM,
//! A1 WaveNets, FiLM/gated layers, custom shapes) is rejected at load time.
//!
//! ```no_run
//! use robine_nam::{Model, Size};
//!
//! let mut model = Model::load("amp.nam")?;              // largest variant
//! let mut small = Model::load_size("amp.nam", Size::Small)?;
//! let embedded = Model::parse_bytes(include_bytes!("../tests/fixtures/A2.nam"))?;
//! assert_eq!(model.sample_rate(), 48_000.0);
//!
//! let input = vec![0.0f32; 512];
//! let mut output = vec![0.0f32; 512];
//! model.process(&input, &mut output);                   // RT-safe
//! # Ok::<(), robine_nam::Error>(())
//! ```
//!
//! Input can be a path (`load`, `NamFile::read`), text (`parse`), bytes
//! (`parse_bytes`, e.g. an embedded or downloaded model) or any `Read`
//! (`from_reader`). To pick a size from bytes, go through [`NamFile`].
//!
//! The crate does no resampling, no gain staging and no I/O beyond reading
//! the `.nam` JSON; those belong to the host. [`Metadata`] exposes the
//! loudness and calibration levels (plus [`Metadata::normalization_gain_db`]
//! and [`Metadata::input_calibration_gain_db`]) so a host can implement the
//! usual NAM gain staging.

// `unsafe` is confined to the SIMD intrinsics wrapper in `simd.rs`.
#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod arch;
mod engine;
mod error;
mod file;
mod model;
mod simd;

pub use arch::Size;
pub use error::Error;
pub use file::{db_to_linear, Metadata, NamFile, Variant, DEFAULT_SAMPLE_RATE};
pub use model::Model;
