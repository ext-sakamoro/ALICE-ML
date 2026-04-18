//! Safetensors format parser for ALICE-ML.
//!
//! Reads safetensors files (HuggingFace standard) and converts
//! BF16/FP16/FP32 tensors for ternary quantization.

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap as HashMap, string::String, vec, vec::Vec};

// ─── Half-precision conversion ──────────────────────────────────────────────

/// Convert BF16 (Brain Float 16) to f32.
/// BF16 is simply the upper 16 bits of an f32.
#[inline]
#[must_use]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Convert IEEE 754 half-precision (FP16) to f32.
#[inline]
#[must_use]
pub fn fp16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exponent = ((h >> 10) & 0x1f) as u32;
    let mantissa = (h & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 0u32;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let exp = 127 - 15 - e + 1;
            let man = (m & 0x3ff) << 13;
            f32::from_bits((sign << 31) | (exp << 23) | man)
        }
    } else if exponent == 31 {
        f32::from_bits((sign << 31) | (0xff << 23) | (mantissa << 13))
    } else {
        let exp = exponent + 127 - 15;
        let man = mantissa << 13;
        f32::from_bits((sign << 31) | (exp << 23) | man)
    }
}

// ─── Data types ─────────────────────────────────────────────────────────────

/// Tensor data type in safetensors format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32-bit float
    F32,
    /// 16-bit float (IEEE 754)
    F16,
    /// Brain Float 16
    BF16,
    /// 64-bit float
    F64,
    /// Unknown type
    Other,
}

impl DType {
    fn from_str(s: &str) -> Self {
        match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "F64" => Self::F64,
            _ => Self::Other,
        }
    }

    /// Bytes per element.
    #[must_use]
    pub fn element_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::Other => 0,
        }
    }
}

// ─── Tensor descriptor ──────────────────────────────────────────────────────

/// Descriptor for a single tensor within a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    /// Data type (BF16, F16, F32, etc.)
    pub dtype: DType,
    /// Shape (e.g., [4096, 4096] for a weight matrix)
    pub shape: Vec<usize>,
    /// Byte offset range [start, end) within the data section
    pub data_start: usize,
    /// Byte offset end within the data section
    pub data_end: usize,
}

impl TensorDesc {
    /// Total number of elements.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Data size in bytes.
    #[must_use]
    pub fn data_size(&self) -> usize {
        self.data_end - self.data_start
    }
}

// ─── Minimal JSON parser for safetensors header ─────────────────────────────

struct JsonParser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.data.len()
            && matches!(self.data[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    fn consume(&mut self, expected: u8) -> bool {
        self.skip_whitespace();
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn parse_string(&mut self) -> Option<String> {
        self.skip_whitespace();
        if !self.consume(b'"') {
            return None;
        }
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'"' {
            if self.data[self.pos] == b'\\' {
                self.pos += 1;
            }
            self.pos += 1;
        }
        let s = String::from_utf8_lossy(&self.data[start..self.pos]).into_owned();
        self.pos += 1; // closing quote
        Some(s)
    }

    fn parse_number(&mut self) -> Option<usize> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.data[start..self.pos]).ok()?;
        s.parse().ok()
    }

    fn parse_number_array(&mut self) -> Option<Vec<usize>> {
        self.skip_whitespace();
        if !self.consume(b'[') {
            return None;
        }
        let mut arr = Vec::new();
        self.skip_whitespace();
        if self.peek() != Some(b']') {
            arr.push(self.parse_number()?);
            while self.consume(b',') {
                arr.push(self.parse_number()?);
            }
        }
        self.consume(b']');
        Some(arr)
    }

    fn skip_value(&mut self) {
        self.skip_whitespace();
        match self.peek() {
            Some(b'"') => {
                let _ = self.parse_string();
            }
            Some(b'{') => {
                self.pos += 1;
                let mut depth = 1;
                while self.pos < self.data.len() && depth > 0 {
                    match self.data[self.pos] {
                        b'{' => depth += 1,
                        b'}' => depth -= 1,
                        b'"' => {
                            self.pos += 1;
                            while self.pos < self.data.len() && self.data[self.pos] != b'"' {
                                if self.data[self.pos] == b'\\' {
                                    self.pos += 1;
                                }
                                self.pos += 1;
                            }
                        }
                        _ => {}
                    }
                    self.pos += 1;
                }
            }
            Some(b'[') => {
                self.pos += 1;
                let mut depth = 1;
                while self.pos < self.data.len() && depth > 0 {
                    match self.data[self.pos] {
                        b'[' => depth += 1,
                        b']' => depth -= 1,
                        _ => {}
                    }
                    self.pos += 1;
                }
            }
            _ => {
                while self.pos < self.data.len()
                    && !matches!(self.data[self.pos], b',' | b'}' | b']')
                {
                    self.pos += 1;
                }
            }
        }
    }

    /// Parse tensor descriptor object: {"dtype":"BF16","shape":[4096,4096],"data_offsets":[0,33554432]}
    fn parse_tensor_desc(&mut self) -> Option<TensorDesc> {
        if !self.consume(b'{') {
            return None;
        }

        let mut dtype = DType::Other;
        let mut shape = Vec::new();
        let mut data_start = 0usize;
        let mut data_end = 0usize;

        loop {
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                break;
            }

            let key = self.parse_string()?;
            self.consume(b':');

            match key.as_str() {
                "dtype" => {
                    let dt_str = self.parse_string()?;
                    dtype = DType::from_str(&dt_str);
                }
                "shape" => {
                    shape = self.parse_number_array()?;
                }
                "data_offsets" => {
                    let offsets = self.parse_number_array()?;
                    if offsets.len() >= 2 {
                        data_start = offsets[0];
                        data_end = offsets[1];
                    }
                }
                _ => self.skip_value(),
            }

            self.consume(b',');
        }

        Some(TensorDesc {
            dtype,
            shape,
            data_start,
            data_end,
        })
    }

    /// Parse the top-level object: {"__metadata__":{...}, "tensor_name":{...}, ...}
    fn parse_header(&mut self) -> Option<HashMap<String, TensorDesc>> {
        if !self.consume(b'{') {
            return None;
        }

        let mut tensors = HashMap::new();

        loop {
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                break;
            }

            let key = self.parse_string()?;
            self.consume(b':');

            if key == "__metadata__" {
                self.skip_value();
            } else if let Some(desc) = self.parse_tensor_desc() {
                tensors.insert(key, desc);
            } else {
                self.skip_value();
            }

            self.consume(b',');
        }

        Some(tensors)
    }
}

// ─── Safetensors file ───────────────────────────────────────────────────────

/// A parsed safetensors file with zero-copy data access.
pub struct SafetensorsFile<'a> {
    /// Tensor descriptors keyed by name.
    pub tensors: HashMap<String, TensorDesc>,
    /// Raw tensor data section.
    data: &'a [u8],
}

impl<'a> SafetensorsFile<'a> {
    /// Parse a safetensors file from a byte slice.
    ///
    /// Format: [header_size: u64 LE][header: JSON][tensor_data...]
    pub fn parse(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < 8 {
            return None;
        }

        let header_size = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as usize;

        if 8 + header_size > bytes.len() {
            return None;
        }

        let header_bytes = &bytes[8..8 + header_size];
        let data = &bytes[8 + header_size..];

        let mut parser = JsonParser::new(header_bytes);
        let tensors = parser.parse_header()?;

        Some(Self { tensors, data })
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|k| k.as_str()).collect()
    }

    /// Get raw bytes for a tensor.
    pub fn tensor_bytes(&self, name: &str) -> Option<&'a [u8]> {
        let desc = self.tensors.get(name)?;
        if desc.data_end > self.data.len() {
            return None;
        }
        Some(&self.data[desc.data_start..desc.data_end])
    }

    /// Read a tensor as f32, converting from BF16/FP16 if needed.
    pub fn tensor_to_f32(&self, name: &str) -> Option<Vec<f32>> {
        let desc = self.tensors.get(name)?;
        let bytes = self.tensor_bytes(name)?;
        let n = desc.n_elements();

        let mut out = Vec::with_capacity(n);

        match desc.dtype {
            DType::F32 => {
                for i in 0..n {
                    let off = i * 4;
                    out.push(f32::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                        bytes[off + 2],
                        bytes[off + 3],
                    ]));
                }
            }
            DType::BF16 => {
                for i in 0..n {
                    let off = i * 2;
                    out.push(bf16_to_f32(u16::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                    ])));
                }
            }
            DType::F16 => {
                for i in 0..n {
                    let off = i * 2;
                    out.push(fp16_to_f32(u16::from_le_bytes([
                        bytes[off],
                        bytes[off + 1],
                    ])));
                }
            }
            _ => return None,
        }

        Some(out)
    }

    /// Get tensor descriptor.
    pub fn tensor_desc(&self, name: &str) -> Option<&TensorDesc> {
        self.tensors.get(name)
    }

    /// Number of tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the file contains no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_to_f32() {
        // 1.0 in BF16 = 0x3F80
        assert!((bf16_to_f32(0x3F80) - 1.0).abs() < 1e-6);
        // -1.0 in BF16 = 0xBF80
        assert!((bf16_to_f32(0xBF80) - (-1.0)).abs() < 1e-6);
        // 0.0 in BF16 = 0x0000
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        // 2.0 in BF16 = 0x4000
        assert!((bf16_to_f32(0x4000) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_fp16_to_f32() {
        assert_eq!(fp16_to_f32(0x0000), 0.0);
        assert!((fp16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        assert!((fp16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        assert!((fp16_to_f32(0x4000) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::F32.element_size(), 4);
        assert_eq!(DType::F16.element_size(), 2);
        assert_eq!(DType::BF16.element_size(), 2);
        assert_eq!(DType::F64.element_size(), 8);
    }

    fn make_safetensors_bytes(header_json: &str, data: &[u8]) -> Vec<u8> {
        let header_bytes = header_json.as_bytes();
        let header_size = header_bytes.len() as u64;
        let mut buf = Vec::new();
        buf.extend_from_slice(&header_size.to_le_bytes());
        buf.extend_from_slice(header_bytes);
        buf.extend_from_slice(data);
        buf
    }

    #[test]
    fn test_parse_safetensors_f32() {
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let header = r#"{"weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let bytes = make_safetensors_bytes(header, &data);

        let sf = SafetensorsFile::parse(&bytes).unwrap();
        assert_eq!(sf.len(), 1);

        let desc = sf.tensor_desc("weight").unwrap();
        assert_eq!(desc.dtype, DType::F32);
        assert_eq!(desc.shape, vec![2, 2]);
        assert_eq!(desc.n_elements(), 4);

        let values = sf.tensor_to_f32("weight").unwrap();
        assert_eq!(values.len(), 4);
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_safetensors_bf16() {
        // 1.0 = 0x3F80, 2.0 = 0x4000 in BF16
        let data: Vec<u8> = [0x3F80u16, 0x4000]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();

        let header = r#"{"vec":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#;
        let bytes = make_safetensors_bytes(header, &data);

        let sf = SafetensorsFile::parse(&bytes).unwrap();
        let values = sf.tensor_to_f32("vec").unwrap();
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_with_metadata() {
        let header = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let data = 42.0f32.to_le_bytes();
        let bytes = make_safetensors_bytes(header, &data);

        let sf = SafetensorsFile::parse(&bytes).unwrap();
        assert_eq!(sf.len(), 1);
        let v = sf.tensor_to_f32("w").unwrap();
        assert!((v[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_multiple_tensors() {
        let header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
        let mut data = Vec::new();
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&2.0f32.to_le_bytes());
        data.extend_from_slice(&3.0f32.to_le_bytes());
        data.extend_from_slice(&4.0f32.to_le_bytes());

        let bytes = make_safetensors_bytes(header, &data);
        let sf = SafetensorsFile::parse(&bytes).unwrap();

        assert_eq!(sf.len(), 2);
        let a = sf.tensor_to_f32("a").unwrap();
        assert!((a[0] - 1.0).abs() < 1e-6);
        assert!((a[1] - 2.0).abs() < 1e-6);
        let b = sf.tensor_to_f32("b").unwrap();
        assert!((b[0] - 3.0).abs() < 1e-6);
        assert!((b[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_file() {
        let bytes = [0u8; 4]; // too short
        assert!(SafetensorsFile::parse(&bytes).is_none());
    }

    #[test]
    fn test_tensor_names() {
        let header = r#"{"alpha":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"beta":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#;
        let data = vec![0u8; 8];
        let bytes = make_safetensors_bytes(header, &data);
        let sf = SafetensorsFile::parse(&bytes).unwrap();
        let mut names: Vec<&str> = sf.tensor_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_missing_tensor() {
        let header = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let data = 1.0f32.to_le_bytes();
        let bytes = make_safetensors_bytes(header, &data);
        let sf = SafetensorsFile::parse(&bytes).unwrap();
        assert!(sf.tensor_to_f32("nonexistent").is_none());
    }
}
