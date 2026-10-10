use std::fmt;

/// Errors produced while reading or building a model.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The `.nam` text is not valid JSON.
    Json(serde_json::Error),
    /// I/O failure while reading a file.
    Io(std::io::Error),
    /// A required JSON field is missing or has the wrong type.
    MissingField(&'static str),
    /// Top-level architecture is neither `WaveNet` nor `SlimmableContainer`.
    UnsupportedArchitecture(String),
    /// The file is a WaveNet but not an A2 small/large shape.
    UnsupportedShape(String),
    /// A container holds no supported submodel. The payload lists why each
    /// submodel was rejected.
    NoSupportedVariant(Vec<String>),
    /// The requested size is not present in the file.
    SizeNotAvailable,
    /// The weight vector length does not match the architecture.
    WeightCount {
        /// Weights required by the architecture.
        expected: usize,
        /// Weights found in the file.
        found: usize,
    },
    /// A weight is not a finite number.
    NonFiniteWeight(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Json(e) => write!(f, "invalid .nam JSON: {e}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::MissingField(name) => write!(f, "missing or invalid field `{name}`"),
            Error::UnsupportedArchitecture(a) => write!(f, "unsupported architecture `{a}`"),
            Error::UnsupportedShape(why) => {
                write!(f, "WaveNet is not an A2 small/large model: {why}")
            }
            Error::NoSupportedVariant(reasons) => {
                write!(f, "container has no supported A2 submodel")?;
                for r in reasons {
                    write!(f, "; {r}")?;
                }
                Ok(())
            }
            Error::SizeNotAvailable => write!(f, "requested model size is not in the file"),
            Error::WeightCount { expected, found } => {
                write!(f, "expected {expected} weights, found {found}")
            }
            Error::NonFiniteWeight(i) => write!(f, "weight {i} is not finite"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Json(e) => Some(e),
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
