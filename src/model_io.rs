//! Model I/O — 重みのシリアライズ/デシリアライズ。
//!
//! `TernaryWeight` をバイト列に変換してファイル保存/読込するための
//! 軽量バイナリフォーマット。外部依存ゼロ。

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::ops::TernaryWeight;

/// モデルファイルのマジックナンバー。
const MAGIC: [u8; 4] = *b"ATML";

/// フォーマットバージョン。
const FORMAT_VERSION: u32 = 1;

/// シリアライズされたモデルレイヤー。
#[derive(Debug, Clone)]
pub struct SerializedLayer {
    /// レイヤー名ハッシュ (FNV-1a)。
    pub name_hash: u64,
    /// 出力特徴数。
    pub out_features: usize,
    /// 入力特徴数。
    pub in_features: usize,
    /// スケール係数。
    pub scale: f32,
    /// パック済み重み (4 weights per byte)。
    pub packed: Vec<u8>,
}

/// モデル全体のシリアライズコンテナ。
#[derive(Debug, Clone)]
pub struct ModelArchive {
    /// レイヤー一覧。
    layers: Vec<SerializedLayer>,
}

impl ModelArchive {
    /// 空のアーカイブを作成。
    #[must_use]
    pub const fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// `TernaryWeight` をレイヤーとして追加。
    pub fn add_layer(&mut self, name: &str, weights: &TernaryWeight) {
        self.layers.push(SerializedLayer {
            name_hash: fnv1a(name.as_bytes()),
            out_features: weights.out_features(),
            in_features: weights.in_features(),
            scale: weights.scale(),
            packed: weights.packed().to_vec(),
        });
    }

    /// レイヤー数。
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// レイヤーをインデックスで取得。
    #[must_use]
    pub fn get_layer(&self, index: usize) -> Option<&SerializedLayer> {
        self.layers.get(index)
    }

    /// レイヤーを名前ハッシュで検索。
    #[must_use]
    pub fn find_layer(&self, name: &str) -> Option<&SerializedLayer> {
        let hash = fnv1a(name.as_bytes());
        self.layers.iter().find(|l| l.name_hash == hash)
    }

    /// バイト列にシリアライズ。
    ///
    /// フォーマット:
    /// - `[4]` マジック "ATML"
    /// - `[4]` バージョン (u32 LE)
    /// - `[4]` レイヤー数 (u32 LE)
    /// - 各レイヤー:
    ///   - `[8]` 名前ハッシュ (u64 LE)
    ///   - `[4]` `out_features` (u32 LE)
    ///   - `[4]` `in_features` (u32 LE)
    ///   - `[4]` scale (f32 LE)
    ///   - `[4]` packed バイト数 (u32 LE)
    ///   - `[N]` packed データ
    #[must_use]
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&MAGIC);
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());

        for layer in &self.layers {
            buf.extend_from_slice(&layer.name_hash.to_le_bytes());
            buf.extend_from_slice(&(layer.out_features as u32).to_le_bytes());
            buf.extend_from_slice(&(layer.in_features as u32).to_le_bytes());
            buf.extend_from_slice(&layer.scale.to_le_bytes());
            buf.extend_from_slice(&(layer.packed.len() as u32).to_le_bytes());
            buf.extend_from_slice(&layer.packed);
        }

        buf
    }

    /// バイト列からデシリアライズ。
    ///
    /// # Errors
    ///
    /// フォーマットエラー時は `None` を返す。
    #[must_use]
    pub fn deserialize(data: &[u8]) -> Option<Self> {
        if data.len() < 12 {
            return None;
        }
        if data[..4] != MAGIC {
            return None;
        }
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != FORMAT_VERSION {
            return None;
        }
        let layer_count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        let mut offset = 12;
        let mut layers = Vec::with_capacity(layer_count);

        for _ in 0..layer_count {
            if offset + 24 > data.len() {
                return None;
            }
            let name_hash = u64::from_le_bytes(data[offset..offset + 8].try_into().ok()?);
            offset += 8;
            let out_features =
                u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
            offset += 4;
            let in_features =
                u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
            offset += 4;
            let scale = f32::from_le_bytes(data[offset..offset + 4].try_into().ok()?);
            offset += 4;
            let packed_len = u32::from_le_bytes(data[offset..offset + 4].try_into().ok()?) as usize;
            offset += 4;

            if offset + packed_len > data.len() {
                return None;
            }
            let packed = data[offset..offset + packed_len].to_vec();
            offset += packed_len;

            layers.push(SerializedLayer {
                name_hash,
                out_features,
                in_features,
                scale,
                packed,
            });
        }

        Some(Self { layers })
    }

    /// `SerializedLayer` から `TernaryWeight` を復元。
    #[must_use]
    pub fn restore_weight(layer: &SerializedLayer) -> TernaryWeight {
        TernaryWeight::from_packed(
            layer.packed.clone(),
            layer.out_features,
            layer.in_features,
            layer.scale,
        )
    }
}

impl Default for ModelArchive {
    fn default() -> Self {
        Self::new()
    }
}

/// FNV-1a ハッシュ (64-bit)。
fn fnv1a(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_archive() {
        let a = ModelArchive::new();
        assert_eq!(a.layer_count(), 0);
    }

    #[test]
    fn default_archive() {
        let a = ModelArchive::default();
        assert_eq!(a.layer_count(), 0);
    }

    #[test]
    fn add_and_find_layer() {
        let mut a = ModelArchive::new();
        let w = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
        a.add_layer("linear0", &w);
        assert_eq!(a.layer_count(), 1);
        assert!(a.find_layer("linear0").is_some());
        assert!(a.find_layer("nonexistent").is_none());
    }

    #[test]
    fn get_layer_index() {
        let mut a = ModelArchive::new();
        let w = TernaryWeight::from_ternary(&[1, 0, -1, 1], 2, 2);
        a.add_layer("layer0", &w);
        let l = a.get_layer(0).unwrap();
        assert_eq!(l.out_features, 2);
        assert_eq!(l.in_features, 2);
        assert!(a.get_layer(1).is_none());
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut a = ModelArchive::new();
        let w1 = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
        let w2 = TernaryWeight::from_ternary(&[1, 0, -1, 0, 1, -1, 0, 1, 0], 3, 3);
        a.add_layer("fc1", &w1);
        a.add_layer("fc2", &w2);

        let bytes = a.serialize();
        let restored = ModelArchive::deserialize(&bytes).unwrap();
        assert_eq!(restored.layer_count(), 2);

        let l0 = restored.get_layer(0).unwrap();
        assert_eq!(l0.out_features, 2);
        assert_eq!(l0.in_features, 2);

        let l1 = restored.get_layer(1).unwrap();
        assert_eq!(l1.out_features, 3);
        assert_eq!(l1.in_features, 3);
    }

    #[test]
    fn restore_weight_from_layer() {
        let mut a = ModelArchive::new();
        let w = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
        a.add_layer("test", &w);

        let bytes = a.serialize();
        let restored = ModelArchive::deserialize(&bytes).unwrap();
        let l = restored.get_layer(0).unwrap();
        let w2 = ModelArchive::restore_weight(l);
        assert_eq!(w2.out_features(), 2);
        assert_eq!(w2.in_features(), 2);
    }

    #[test]
    fn deserialize_invalid_magic() {
        let data = b"XXXX\x01\x00\x00\x00\x00\x00\x00\x00";
        assert!(ModelArchive::deserialize(data).is_none());
    }

    #[test]
    fn deserialize_too_short() {
        assert!(ModelArchive::deserialize(b"ATM").is_none());
    }

    #[test]
    fn deserialize_wrong_version() {
        let mut data = Vec::new();
        data.extend_from_slice(&MAGIC);
        data.extend_from_slice(&99_u32.to_le_bytes());
        data.extend_from_slice(&0_u32.to_le_bytes());
        assert!(ModelArchive::deserialize(&data).is_none());
    }

    #[test]
    fn deserialize_truncated_layer() {
        let mut a = ModelArchive::new();
        let w = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
        a.add_layer("test", &w);
        let bytes = a.serialize();
        // 途中で切断
        let truncated = &bytes[..bytes.len() - 1];
        assert!(ModelArchive::deserialize(truncated).is_none());
    }

    #[test]
    fn serialize_empty_archive() {
        let a = ModelArchive::new();
        let bytes = a.serialize();
        let restored = ModelArchive::deserialize(&bytes).unwrap();
        assert_eq!(restored.layer_count(), 0);
    }

    #[test]
    fn fnv1a_nonzero() {
        assert_ne!(fnv1a(b"hello"), 0);
        assert_ne!(fnv1a(b"hello"), fnv1a(b"world"));
    }

    #[test]
    fn weight_inference_after_roundtrip() {
        // 保存→復元→推論の結果が一致するか検証
        let w = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
        let input = [2.0_f32, 3.0];
        let mut out1 = [0.0_f32; 2];
        crate::ops::ternary_matvec(&input, &w, &mut out1);

        let mut a = ModelArchive::new();
        a.add_layer("fc", &w);
        let bytes = a.serialize();
        let restored = ModelArchive::deserialize(&bytes).unwrap();
        let w2 = ModelArchive::restore_weight(restored.get_layer(0).unwrap());

        let mut out2 = [0.0_f32; 2];
        crate::ops::ternary_matvec(&input, &w2, &mut out2);

        assert!((out1[0] - out2[0]).abs() < 1e-6);
        assert!((out1[1] - out2[1]).abs() < 1e-6);
    }
}
