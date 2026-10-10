//! `.nam` file reading: JSON parsing, A2 shape detection and container
//! submodel selection. Nothing here runs on the audio thread.

use crate::arch::{Size, DILATIONS, HEAD_KERNEL, KERNEL_SIZES, LEAKY_SLOPE, NUM_LAYERS};
use crate::error::Error;
use crate::model::Model;
use serde_json::Value;
use std::path::Path;

/// Default sample rate assumed when a file does not carry one.
pub const DEFAULT_SAMPLE_RATE: f32 = 48_000.0;

/// Descriptive metadata embedded in a `.nam` file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metadata {
    /// Capture name.
    pub name: Option<String>,
    /// Author of the capture.
    pub modeled_by: Option<String>,
    /// Manufacturer of the captured gear.
    pub gear_make: Option<String>,
    /// Model of the captured gear.
    pub gear_model: Option<String>,
    /// Gear category (`amp`, `pedal`, `amp_cab`, ...).
    pub gear_type: Option<String>,
    /// Tone category (`clean`, `crunch`, ...).
    pub tone_type: Option<String>,
    /// Input calibration level in dBu, when the capture was calibrated.
    pub input_level_dbu: Option<f32>,
    /// Output calibration level in dBu, when the capture was calibrated.
    pub output_level_dbu: Option<f32>,
    /// Loudness of the model output for the standard training signal, in dB.
    pub loudness_db: Option<f32>,
    /// Output/input gain ratio measured during training.
    pub gain: Option<f32>,
    /// Validation error-to-signal ratio reported by the trainer.
    pub validation_esr: Option<f32>,
    /// Full metadata object, for anything not mapped above.
    pub raw: Value,
}

impl Metadata {
    fn from_value(value: Value) -> Self {
        let training_esr = value
            .pointer("/training/validation_esr")
            .and_then(Value::as_f64)
            .map(|v| v as f32);
        Self {
            name: string_field(&value, "name"),
            modeled_by: string_field(&value, "modeled_by"),
            gear_make: string_field(&value, "gear_make"),
            gear_model: string_field(&value, "gear_model"),
            gear_type: string_field(&value, "gear_type"),
            tone_type: string_field(&value, "tone_type"),
            input_level_dbu: f32_field(&value, "input_level_dbu"),
            output_level_dbu: f32_field(&value, "output_level_dbu"),
            loudness_db: f32_field(&value, "loudness"),
            gain: f32_field(&value, "gain"),
            validation_esr: training_esr,
            raw: value,
        }
    }

    /// `self` with every unset field taken from `fallback`.
    fn with_fallback(mut self, fallback: &Metadata) -> Self {
        macro_rules! fill {
            ($($field:ident),*) => { $( if self.$field.is_none() { self.$field = fallback.$field.clone(); } )* };
        }
        fill!(
            name,
            modeled_by,
            gear_make,
            gear_model,
            gear_type,
            tone_type,
            input_level_dbu,
            output_level_dbu,
            loudness_db,
            gain,
            validation_esr
        );
        self
    }

    /// Best human-readable name: `name`, else `gear_model`, else `fallback`.
    pub fn display_name<'a>(&'a self, fallback: &'a str) -> &'a str {
        self.name
            .as_deref()
            .or(self.gear_model.as_deref())
            .unwrap_or(fallback)
    }

    /// Output gain, in dB, that brings the model's reported loudness to
    /// `target_db` (the NAM plugin "normalize" switch uses -18 dBFS).
    /// `None` when the file carries no loudness.
    pub fn normalization_gain_db(&self, target_db: f32) -> Option<f32> {
        self.loudness_db.map(|loudness| target_db - loudness)
    }

    /// Input gain, in dB, to apply when the host's interface is calibrated
    /// at `interface_input_dbu` so that the model sees the level it was
    /// trained with. `None` when the capture is not calibrated.
    pub fn input_calibration_gain_db(&self, interface_input_dbu: f32) -> Option<f32> {
        self.input_level_dbu
            .map(|model| interface_input_dbu - model)
    }

    /// Whether the capture carries input calibration data.
    pub fn is_calibrated(&self) -> bool {
        self.input_level_dbu.is_some()
    }
}

/// Convert decibels to a linear gain factor.
pub fn db_to_linear(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// One concrete A2 model found in a file (a `SlimmableContainer` carries one
/// per size, a bare `WaveNet` carries exactly one).
#[derive(Clone, Debug)]
pub struct Variant {
    /// Which A2 size this variant is.
    pub size: Size,
    /// Upper bound of the container "slim" range this variant covers
    /// (`1.0` for a bare WaveNet).
    pub max_value: f64,
    /// `.nam` format version string.
    pub version: String,
    /// Training sample rate in Hz.
    pub sample_rate: f32,
    /// Metadata (submodel metadata, else the file-level one).
    pub metadata: Metadata,
    /// Flat weight vector in upstream order.
    pub weights: Vec<f32>,
}

/// A parsed `.nam` file restricted to its A2 small/large content.
#[derive(Clone, Debug)]
pub struct NamFile {
    /// `.nam` format version string.
    pub version: String,
    /// Top-level architecture (`WaveNet` or `SlimmableContainer`).
    pub architecture: String,
    /// Training sample rate in Hz.
    pub sample_rate: f32,
    /// Metadata (submodel metadata, else the file-level one).
    pub metadata: Metadata,
    /// Variants sorted by ascending `max_value` (so by ascending size in
    /// practice).
    pub variants: Vec<Variant>,
    /// Submodels of a container that were skipped, with the reason.
    pub skipped: Vec<String>,
}

impl NamFile {
    /// Parse `.nam` JSON text.
    pub fn parse(text: &str) -> Result<Self, Error> {
        let root: Value = serde_json::from_str(text)?;
        Self::from_value(root)
    }

    /// Parse `.nam` bytes (UTF-8 JSON).
    pub fn parse_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
        Self::parse(text)
    }

    /// Read a whole `.nam` document from any `Read` (file, socket, embedded
    /// resource) and parse it. The document is buffered first: a `.nam` is
    /// a few hundred kilobytes of JSON that must be complete before parsing,
    /// so true streaming would buy nothing.
    pub fn from_reader(mut reader: impl std::io::Read) -> Result<Self, Error> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::parse_bytes(&bytes)
    }

    /// Read and parse a `.nam` file.
    pub fn read(path: impl AsRef<Path>) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    /// Build from an already-parsed JSON document.
    pub fn from_value(root: Value) -> Result<Self, Error> {
        let version = string_field(&root, "version").ok_or(Error::MissingField("version"))?;
        let architecture =
            string_field(&root, "architecture").ok_or(Error::MissingField("architecture"))?;
        let sample_rate = f32_field(&root, "sample_rate").unwrap_or(DEFAULT_SAMPLE_RATE);
        let metadata = Metadata::from_value(root.get("metadata").cloned().unwrap_or(Value::Null));

        let mut variants = Vec::new();
        let mut skipped = Vec::new();
        match architecture.as_str() {
            "WaveNet" => {
                variants.push(parse_wavenet(&root, 1.0, &version, sample_rate, &metadata)?);
            }
            "SlimmableContainer" => {
                let submodels = root
                    .pointer("/config/submodels")
                    .and_then(Value::as_array)
                    .ok_or(Error::MissingField("config.submodels"))?;
                for (index, entry) in submodels.iter().enumerate() {
                    let max_value = entry
                        .get("max_value")
                        .and_then(Value::as_f64)
                        .ok_or(Error::MissingField("submodels[].max_value"))?;
                    let model = entry
                        .get("model")
                        .ok_or(Error::MissingField("submodels[].model"))?;
                    let sub_version =
                        string_field(model, "version").unwrap_or_else(|| version.clone());
                    let sub_rate = f32_field(model, "sample_rate").unwrap_or(sample_rate);
                    // Submodel metadata wins field by field; the file-level
                    // metadata (what TONE3000 writes: name, gear, calibration)
                    // fills in what the submodel does not carry.
                    let sub_metadata = match model.get("metadata") {
                        Some(Value::Null) | None => metadata.clone(),
                        Some(value) => Metadata::from_value(value.clone()).with_fallback(&metadata),
                    };
                    let sub_arch = string_field(model, "architecture").unwrap_or_default();
                    if sub_arch != "WaveNet" {
                        skipped.push(format!("submodel {index}: architecture `{sub_arch}`"));
                        continue;
                    }
                    match parse_wavenet(model, max_value, &sub_version, sub_rate, &sub_metadata) {
                        Ok(variant) => variants.push(variant),
                        Err(Error::UnsupportedShape(why)) => {
                            skipped.push(format!("submodel {index}: {why}"));
                        }
                        Err(other) => return Err(other),
                    }
                }
                if variants.is_empty() {
                    return Err(Error::NoSupportedVariant(skipped));
                }
                variants.sort_by(|a, b| a.max_value.total_cmp(&b.max_value));
            }
            other => return Err(Error::UnsupportedArchitecture(other.to_string())),
        }

        Ok(Self {
            version,
            architecture,
            sample_rate,
            metadata,
            variants,
            skipped,
        })
    }

    /// Variant of a given size, if present.
    pub fn variant(&self, size: Size) -> Option<&Variant> {
        self.variants.iter().find(|v| v.size == size)
    }

    /// Largest (highest `max_value`) variant.
    pub fn largest(&self) -> &Variant {
        self.variants
            .last()
            .expect("NamFile always has at least one variant")
    }

    /// Smallest (lowest `max_value`) variant.
    pub fn smallest(&self) -> &Variant {
        self.variants
            .first()
            .expect("NamFile always has at least one variant")
    }

    /// Variant selected the way upstream `ContainerModel::SetSlimmableSize`
    /// does: the first variant whose `max_value` is strictly greater than
    /// `slim`, else the last one.
    pub fn for_slim(&self, slim: f64) -> &Variant {
        self.variants
            .iter()
            .find(|v| slim < v.max_value)
            .unwrap_or_else(|| self.largest())
    }

    /// Sizes available in this file, in ascending order.
    pub fn sizes(&self) -> impl Iterator<Item = Size> + '_ {
        self.variants.iter().map(|v| v.size)
    }

    /// Build a runnable model of the given size with the default block size.
    pub fn build(&self, size: Size) -> Result<Model, Error> {
        self.build_with(size, Model::DEFAULT_MAX_BLOCK)
    }

    /// Build a runnable model of the given size. `max_block` is the largest
    /// slice `Model::process` handles in one pass; longer slices are chunked.
    pub fn build_with(&self, size: Size, max_block: usize) -> Result<Model, Error> {
        let variant = self.variant(size).ok_or(Error::SizeNotAvailable)?;
        Model::from_variant(variant, max_block)
    }

    /// Build the largest variant.
    pub fn build_largest(&self) -> Result<Model, Error> {
        Model::from_variant(self.largest(), Model::DEFAULT_MAX_BLOCK)
    }
}

fn parse_wavenet(
    model: &Value,
    max_value: f64,
    version: &str,
    sample_rate: f32,
    metadata: &Metadata,
) -> Result<Variant, Error> {
    let config = model.get("config").ok_or(Error::MissingField("config"))?;
    let channels = a2_channels(config).map_err(|why| Error::UnsupportedShape(why.to_string()))?;
    let size = Size::from_channels(channels)
        .ok_or_else(|| Error::UnsupportedShape(format!("{channels} channels")))?;
    let raw_weights = model
        .get("weights")
        .and_then(Value::as_array)
        .ok_or(Error::MissingField("weights"))?;
    let mut weights = Vec::with_capacity(raw_weights.len());
    for (i, w) in raw_weights.iter().enumerate() {
        let w = w.as_f64().ok_or(Error::NonFiniteWeight(i))? as f32;
        if !w.is_finite() {
            return Err(Error::NonFiniteWeight(i));
        }
        weights.push(w);
    }
    Ok(Variant {
        size,
        max_value,
        version: version.to_string(),
        sample_rate,
        metadata: metadata.clone(),
        weights,
    })
}

/// Strict A2 shape detector. Returns the channel count on success and a
/// human-readable reason on failure. Mirrors upstream `is_a2_shape`.
pub fn a2_channels(config: &Value) -> Result<usize, &'static str> {
    let layers = config
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("missing `layers`")?;
    if layers.len() != 1 {
        return Err("more than one layer array");
    }
    if !is_null_or_absent(config.get("head")) {
        return Err("post-stack head present");
    }
    if !is_null_or_absent(config.get("condition_dsp")) {
        return Err("condition_dsp present");
    }
    if !config.get("head_scale").is_some_and(Value::is_number) {
        return Err("missing numeric `head_scale`");
    }
    if usize_field(config, "in_channels").unwrap_or(1) != 1 {
        return Err("in_channels != 1");
    }

    let layer = &layers[0];
    if usize_field(layer, "input_size") != Some(1) {
        return Err("input_size != 1");
    }
    if usize_field(layer, "condition_size") != Some(1) {
        return Err("condition_size != 1");
    }
    let channels = usize_field(layer, "channels").ok_or("missing `channels`")?;
    if usize_field(layer, "bottleneck") != Some(channels) {
        return Err("bottleneck != channels");
    }
    if Size::from_channels(channels).is_none() {
        return Err("channels is neither 3 nor 8");
    }
    if !usize_array_equals(layer.get("kernel_sizes"), &KERNEL_SIZES) {
        return Err("kernel_sizes differ from A2");
    }
    if !usize_array_equals(layer.get("dilations"), &DILATIONS) {
        return Err("dilations differ from A2");
    }
    if !all_leaky_relu(layer.get("activation")) {
        return Err("activation is not LeakyReLU(0.01) everywhere");
    }
    if !all_strings_or_absent(layer.get("gating_mode"), "none") {
        return Err("gating_mode is not `none`");
    }
    if layer.get("gated").and_then(Value::as_bool).unwrap_or(false) {
        return Err("legacy `gated` is true");
    }
    if !all_null_or_absent(layer.get("secondary_activation")) {
        return Err("secondary_activation present");
    }
    if object_bool(layer.get("head1x1"), "active").unwrap_or(false) {
        return Err("head1x1 active");
    }
    let layer_1x1 = layer.get("layer1x1").ok_or("missing `layer1x1`")?;
    if !object_bool(Some(layer_1x1), "active").unwrap_or(false) {
        return Err("layer1x1 inactive");
    }
    if object_usize(Some(layer_1x1), "groups").unwrap_or(1) != 1 {
        return Err("layer1x1 groups != 1");
    }
    let head = layer.get("head").ok_or("missing layer-array `head`")?;
    if object_usize(Some(head), "out_channels") != Some(1) {
        return Err("head out_channels != 1");
    }
    if object_usize(Some(head), "kernel_size") != Some(HEAD_KERNEL) {
        return Err("head kernel_size != 16");
    }
    if object_usize(Some(head), "head_dilation").unwrap_or(1) != 1 {
        return Err("head_dilation != 1");
    }
    if !object_bool(Some(head), "bias").unwrap_or(false) {
        return Err("head bias != true");
    }
    for key in [
        "conv_pre_film",
        "conv_post_film",
        "input_mixin_pre_film",
        "input_mixin_post_film",
        "activation_pre_film",
        "activation_post_film",
        "layer1x1_post_film",
        "head1x1_post_film",
    ] {
        if !film_inactive(layer.get(key)) {
            return Err("FiLM active");
        }
    }
    if usize_field(layer, "groups_input").unwrap_or(1) != 1 {
        return Err("groups_input != 1");
    }
    if usize_field(layer, "groups_input_mixin").unwrap_or(1) != 1 {
        return Err("groups_input_mixin != 1");
    }
    if !is_null_or_absent(layer.get("slimmable")) {
        return Err("slimmable layer array");
    }
    Ok(channels)
}

fn is_null_or_absent(value: Option<&Value>) -> bool {
    matches!(value, None | Some(Value::Null))
}

fn string_field(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
}

fn f32_field(value: &Value, key: &str) -> Option<f32> {
    value.get(key).and_then(Value::as_f64).map(|v| v as f32)
}

fn usize_field(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
}

fn object_usize(value: Option<&Value>, key: &str) -> Option<usize> {
    value.and_then(|v| usize_field(v, key))
}

fn object_bool(value: Option<&Value>, key: &str) -> Option<bool> {
    value.and_then(|v| v.get(key)).and_then(Value::as_bool)
}

fn usize_array_equals(value: Option<&Value>, expected: &[usize]) -> bool {
    let Some(values) = value.and_then(Value::as_array) else {
        return false;
    };
    values.len() == expected.len()
        && values
            .iter()
            .zip(expected)
            .all(|(v, e)| v.as_u64().is_some_and(|v| v == *e as u64))
}

fn all_leaky_relu(value: Option<&Value>) -> bool {
    let Some(values) = value.and_then(Value::as_array) else {
        return false;
    };
    values.len() == NUM_LAYERS
        && values.iter().all(|a| {
            a.get("type").and_then(Value::as_str) == Some("LeakyReLU")
                && a.get("negative_slope")
                    .and_then(Value::as_f64)
                    .is_some_and(|s| (s as f32 - LEAKY_SLOPE).abs() <= 1e-7)
        })
}

fn all_strings_or_absent(value: Option<&Value>, expected: &str) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::Array(values)) => {
            values.len() == NUM_LAYERS && values.iter().all(|v| v.as_str() == Some(expected))
        }
        Some(Value::String(s)) => s == expected,
        _ => false,
    }
}

fn all_null_or_absent(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::Array(values)) => {
            values.len() == NUM_LAYERS && values.iter().all(Value::is_null)
        }
        _ => false,
    }
}

fn film_inactive(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::Bool(active)) => !active,
        Some(Value::Object(object)) => !object
            .get("active")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        _ => false,
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use serde_json::json;

    pub fn a2_config(channels: usize) -> Value {
        json!({
            "in_channels": 1,
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": channels,
                "bottleneck": channels,
                "kernel_sizes": KERNEL_SIZES,
                "dilations": DILATIONS,
                "activation": vec![json!({"type": "LeakyReLU", "negative_slope": 0.01}); NUM_LAYERS],
                "gating_mode": vec!["none"; NUM_LAYERS],
                "secondary_activation": vec![Value::Null; NUM_LAYERS],
                "head": {"out_channels": 1, "kernel_size": HEAD_KERNEL, "bias": true},
                "layer1x1": {"active": true, "groups": 1},
                "head1x1": {"active": false, "out_channels": 1, "groups": 1},
                "groups_input": 1,
                "groups_input_mixin": 1
            }],
            "head": null,
            "head_scale": 0.01
        })
    }

    pub fn wavenet_json(channels: usize, weights: Vec<f32>) -> String {
        json!({
            "version": "0.7.0",
            "architecture": "WaveNet",
            "config": a2_config(channels),
            "weights": weights,
            "sample_rate": 48000.0,
            "metadata": {"name": "test", "gear_type": "amp"}
        })
        .to_string()
    }

    pub fn container_json(entries: &[(f64, usize, Vec<f32>)]) -> String {
        let submodels: Vec<Value> = entries
            .iter()
            .map(|(max_value, channels, weights)| {
                json!({
                    "max_value": max_value,
                    "model": {
                        "version": "0.7.0",
                        "architecture": "WaveNet",
                        "config": a2_config(*channels),
                        "weights": weights,
                        "sample_rate": 48000.0,
                        "metadata": {"name": format!("test-{channels}")}
                    }
                })
            })
            .collect();
        json!({
            "version": "0.7.0",
            "architecture": "SlimmableContainer",
            "config": {"submodels": submodels},
            "weights": [],
            "sample_rate": 48000.0,
            "metadata": {"name": "container"}
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn parses_bare_wavenet() {
        let file = NamFile::parse(&wavenet_json(8, vec![0.0; Size::Large.weight_count()])).unwrap();
        assert_eq!(file.variants.len(), 1);
        assert_eq!(file.largest().size, Size::Large);
        assert_eq!(file.sample_rate, 48_000.0);
        assert_eq!(file.metadata.display_name("?"), "test");
    }

    #[test]
    fn parses_container_and_sorts_by_max_value() {
        let text = container_json(&[
            (1.0, 8, vec![0.0; Size::Large.weight_count()]),
            (0.5, 3, vec![0.0; Size::Small.weight_count()]),
        ]);
        let file = NamFile::parse(&text).unwrap();
        assert_eq!(
            file.sizes().collect::<Vec<_>>(),
            vec![Size::Small, Size::Large]
        );
        assert_eq!(file.for_slim(0.0).size, Size::Small);
        assert_eq!(file.for_slim(0.49).size, Size::Small);
        assert_eq!(file.for_slim(0.5).size, Size::Large);
        assert_eq!(file.for_slim(1.0).size, Size::Large);
        assert_eq!(
            file.variant(Size::Small).unwrap().metadata.name.as_deref(),
            Some("test-3")
        );
    }

    #[test]
    fn skips_unsupported_submodels_but_keeps_supported_ones() {
        let mut root: Value = serde_json::from_str(&container_json(&[
            (0.5, 3, vec![0.0; Size::Small.weight_count()]),
            (1.0, 8, vec![0.0; Size::Large.weight_count()]),
        ]))
        .unwrap();
        root["config"]["submodels"][1]["model"]["architecture"] = Value::from("LSTM");
        let file = NamFile::from_value(root).unwrap();
        assert_eq!(file.variants.len(), 1);
        assert_eq!(file.skipped.len(), 1);
    }

    #[test]
    fn reads_upstream_example_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/A2.nam");
        let file = NamFile::read(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let again = NamFile::parse_bytes(&bytes).unwrap();
        assert_eq!(
            file.sizes().collect::<Vec<_>>(),
            again.sizes().collect::<Vec<_>>()
        );
        assert_eq!(file.architecture, "SlimmableContainer");
        assert_eq!(
            file.sizes().collect::<Vec<_>>(),
            vec![Size::Small, Size::Large]
        );
        let large = file.variant(Size::Large).unwrap();
        assert!(large.metadata.is_calibrated());
        assert!((large.metadata.input_level_dbu.unwrap() - 14.76).abs() < 0.01);
        let gain = large.metadata.normalization_gain_db(-18.0).unwrap();
        assert!((gain - (-18.0 - large.metadata.loudness_db.unwrap())).abs() < 1e-6);
        assert!((db_to_linear(-6.0) - 0.501_187).abs() < 1e-5);
    }

    #[test]
    fn rejects_lstm() {
        let err = NamFile::parse(
            r#"{"version":"0.5.4","architecture":"LSTM","config":{},"weights":[0.1]}"#,
        )
        .unwrap_err();
        assert!(matches!(err, Error::UnsupportedArchitecture(_)));
    }

    #[test]
    fn rejects_a1_wavenet() {
        let text = r#"{"version":"0.5.4","architecture":"WaveNet","config":{"layers":[{"input_size":1,"condition_size":1,"head_size":2,"channels":16,"kernel_size":3,"dilations":[1,2],"activation":"Tanh","gated":false,"head_bias":false}],"head":null,"head_scale":0.02},"weights":[0.1]}"#;
        let err = NamFile::parse(text).unwrap_err();
        assert!(matches!(err, Error::UnsupportedShape(_)), "{err}");
    }

    #[test]
    fn rejects_wrong_weight_count_at_build() {
        let file =
            NamFile::parse(&wavenet_json(3, vec![0.0; Size::Small.weight_count() + 1])).unwrap();
        let err = file.build(Size::Small).unwrap_err();
        assert!(matches!(err, Error::WeightCount { .. }), "{err}");
    }
}
