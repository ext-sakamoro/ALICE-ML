//! Weight Streaming — レイヤー単位のオンデマンドロード。
//!
//! 24B モデル全体をメモリに常駐させず、推論に必要なレイヤーだけを
//! ディスク/ストレージからストリーミングする。8GB RAM のラズパイ 5 で
//! 3GB のモデルを動かす場合でも、KV キャッシュや中間バッファに
//! 十分なメモリを確保できる。
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//! │  Storage     │────▶│ LayerStreamer │────▶│  BitLinear   │
//! │  (disk/mmap) │     │  (Hot/Cold)  │     │  (inference) │
//! └──────────────┘     └──────────────┘     └──────────────┘
//!                            │
//!                            ▼
//!                      ┌──────────────┐
//!                      │ ALICE-Cache  │  (optional, feature flag)
//!                      │ Oracle+Tiered│
//!                      └──────────────┘
//! ```
//!
//! # Example
//!
//! ```rust
//! use alice_ml::streaming::{LayerStreamer, StreamerConfig};
//! use alice_ml::model_io::ModelArchive;
//! use alice_ml::ops::TernaryWeight;
//!
//! // モデルをシリアライズ
//! let mut archive = ModelArchive::new();
//! let w0 = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
//! let w1 = TernaryWeight::from_ternary(&[-1, 0, 1, -1], 2, 2);
//! archive.add_layer("layer.0", &w0);
//! archive.add_layer("layer.1", &w1);
//! let data = archive.serialize();
//!
//! // ストリーマーを構築
//! let config = StreamerConfig { max_hot_layers: 1 };
//! let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();
//!
//! // レイヤー 0 をロード
//! let layer0 = streamer.load_layer(0).unwrap();
//! assert_eq!(layer0.out_features(), 2);
//!
//! // レイヤー 1 をロード（Hot 上限 1 なのでレイヤー 0 は退去）
//! let layer1 = streamer.load_layer(1).unwrap();
//! assert_eq!(layer1.out_features(), 2);
//! assert!(!streamer.is_hot(0));
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::model_io::{ModelArchive, SerializedLayer};
use crate::ops::TernaryWeight;

// ============================================================================
// Configuration
// ============================================================================

/// Weight Streaming の設定。
#[derive(Debug, Clone, Copy)]
pub struct StreamerConfig {
    /// メモリに常駐させる最大レイヤー数 (Hot 層)。
    pub max_hot_layers: usize,
}

impl Default for StreamerConfig {
    fn default() -> Self {
        Self { max_hot_layers: 8 }
    }
}

// ============================================================================
// Layer Slot
// ============================================================================

/// レイヤーの状態。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    /// メモリに常駐 (Hot)。
    Hot,
    /// ディスク上 (Cold)。重みはアンロード済み。
    Cold,
}

/// レイヤーのスロット管理。
#[derive(Debug)]
struct LayerSlot {
    /// スロット状態。
    state: SlotState,
    /// Hot 時のみ保持。Cold なら None。
    weights: Option<TernaryWeight>,
    /// アクセス回数（LRU 代替の簡易カウンタ）。
    access_count: u64,
    /// 最終アクセス順序（単調増加のタイムスタンプ代替）。
    last_access: u64,
}

// ============================================================================
// Streamer Stats
// ============================================================================

/// ストリーマーの統計情報。
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamerStats {
    /// Hot ヒット数 (ロード不要)。
    pub hits: u64,
    /// Cold → Hot ロード数。
    pub loads: u64,
    /// Hot → Cold 退去数。
    pub evictions: u64,
}

impl StreamerStats {
    /// ヒット率 (0.0〜1.0)。
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.loads;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ============================================================================
// LayerStreamer
// ============================================================================

/// レイヤー単位のオンデマンドウェイトストリーマー。
///
/// シリアライズされたモデルアーカイブからレイヤーを必要に応じてロードし、
/// Hot 層の容量を超えた場合は LRU で退去する。
pub struct LayerStreamer {
    /// 全レイヤーのシリアライズデータ（Cold 層のソース）。
    archive: ModelArchive,
    /// レイヤースロット管理。
    slots: Vec<LayerSlot>,
    /// 設定。
    config: StreamerConfig,
    /// 現在 Hot なレイヤー数。
    hot_count: usize,
    /// 単調増加のアクセスカウンタ。
    global_clock: u64,
    /// 統計。
    stats: StreamerStats,
}

impl LayerStreamer {
    /// シリアライズ済みバイト列からストリーマーを構築。
    ///
    /// # Errors
    /// デシリアライズに失敗した場合 `None` を返す。
    #[must_use]
    pub fn from_bytes(data: &[u8], config: StreamerConfig) -> Option<Self> {
        let archive = ModelArchive::deserialize(data)?;
        let layer_count = archive.layer_count();

        let slots = (0..layer_count)
            .map(|_| LayerSlot {
                state: SlotState::Cold,
                weights: None,
                access_count: 0,
                last_access: 0,
            })
            .collect();

        Some(Self {
            archive,
            slots,
            config,
            hot_count: 0,
            global_clock: 0,
            stats: StreamerStats::default(),
        })
    }

    /// `ModelArchive` から直接構築。
    #[must_use]
    pub fn from_archive(archive: ModelArchive, config: StreamerConfig) -> Self {
        let layer_count = archive.layer_count();

        let slots = (0..layer_count)
            .map(|_| LayerSlot {
                state: SlotState::Cold,
                weights: None,
                access_count: 0,
                last_access: 0,
            })
            .collect();

        Self {
            archive,
            slots,
            config,
            hot_count: 0,
            global_clock: 0,
            stats: StreamerStats::default(),
        }
    }

    /// レイヤー総数。
    #[must_use]
    pub const fn layer_count(&self) -> usize {
        self.slots.len()
    }

    /// 設定を取得。
    #[must_use]
    pub const fn config(&self) -> &StreamerConfig {
        &self.config
    }

    /// 統計を取得。
    #[must_use]
    pub const fn stats(&self) -> &StreamerStats {
        &self.stats
    }

    /// 現在 Hot なレイヤー数。
    #[must_use]
    pub const fn hot_count(&self) -> usize {
        self.hot_count
    }

    /// 指定レイヤーが Hot かどうか。
    #[must_use]
    pub fn is_hot(&self, index: usize) -> bool {
        self.slots
            .get(index)
            .is_some_and(|s| s.state == SlotState::Hot)
    }

    /// 指定レイヤーの状態を取得。
    #[must_use]
    pub fn slot_state(&self, index: usize) -> Option<SlotState> {
        self.slots.get(index).map(|s| s.state)
    }

    /// レイヤーをロードし、`TernaryWeight` の参照を返す。
    ///
    /// Hot 層に既にあればそのまま返す（ヒット）。
    /// Cold ならアーカイブからデシリアライズして Hot に昇格。
    /// Hot 層が満杯なら LRU で退去してから昇格。
    ///
    /// # Returns
    /// `None` if index is out of range.
    pub fn load_layer(&mut self, index: usize) -> Option<&TernaryWeight> {
        if index >= self.slots.len() {
            return None;
        }

        self.global_clock += 1;
        let clock = self.global_clock;

        if self.slots[index].state == SlotState::Hot {
            // Hot ヒット
            self.slots[index].access_count += 1;
            self.slots[index].last_access = clock;
            self.stats.hits += 1;
            return self.slots[index].weights.as_ref();
        }

        // Cold → Hot: ロードが必要
        // Hot 層が満杯なら退去
        if self.hot_count >= self.config.max_hot_layers {
            self.evict_lru(index);
        }

        // アーカイブからデシリアライズ
        let layer_data = self.archive.get_layer(index)?;
        let weights = restore_weight(layer_data);

        self.slots[index].state = SlotState::Hot;
        self.slots[index].weights = Some(weights);
        self.slots[index].access_count += 1;
        self.slots[index].last_access = clock;
        self.hot_count += 1;
        self.stats.loads += 1;

        self.slots[index].weights.as_ref()
    }

    /// 指定レイヤーをプリフェッチ（Hot に昇格）。
    ///
    /// 既に Hot なら何もしない。推論ループ中で「次に使うレイヤー」を
    /// 先に読み込んでおくために使う。
    ///
    /// # Returns
    /// `true` if successfully prefetched or already hot.
    pub fn prefetch_layer(&mut self, index: usize) -> bool {
        if index >= self.slots.len() {
            return false;
        }
        if self.slots[index].state == SlotState::Hot {
            return true;
        }
        self.load_layer(index).is_some()
    }

    /// 指定レイヤーを強制退去 (Hot → Cold)。
    pub fn evict_layer(&mut self, index: usize) {
        if index >= self.slots.len() {
            return;
        }
        if self.slots[index].state == SlotState::Hot {
            self.slots[index].state = SlotState::Cold;
            self.slots[index].weights = None;
            self.hot_count -= 1;
            self.stats.evictions += 1;
        }
    }

    /// 全レイヤーを Cold に退去。
    pub fn evict_all(&mut self) {
        for slot in &mut self.slots {
            if slot.state == SlotState::Hot {
                slot.state = SlotState::Cold;
                slot.weights = None;
                self.stats.evictions += 1;
            }
        }
        self.hot_count = 0;
    }

    /// Hot 層のメモリ使用量合計 (bytes)。
    #[must_use]
    pub fn hot_memory_bytes(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state == SlotState::Hot)
            .filter_map(|s| s.weights.as_ref())
            .map(TernaryWeight::memory_bytes)
            .sum()
    }

    /// 全レイヤーの合計メモリ使用量 (bytes)（全て Hot の場合）。
    #[must_use]
    pub fn total_model_bytes(&self) -> usize {
        let mut total = 0;
        for i in 0..self.archive.layer_count() {
            if let Some(layer) = self.archive.get_layer(i) {
                total += layer.packed.len();
            }
        }
        total
    }

    /// LRU 退去: `exclude` 以外で `last_access` が最も古い Hot レイヤーを退去。
    fn evict_lru(&mut self, exclude: usize) {
        let mut victim = None;
        let mut oldest = u64::MAX;

        for (i, slot) in self.slots.iter().enumerate() {
            if i == exclude {
                continue;
            }
            if slot.state == SlotState::Hot && slot.last_access < oldest {
                oldest = slot.last_access;
                victim = Some(i);
            }
        }

        if let Some(idx) = victim {
            self.slots[idx].state = SlotState::Cold;
            self.slots[idx].weights = None;
            self.hot_count -= 1;
            self.stats.evictions += 1;
        }
    }
}

// ============================================================================
// Helper
// ============================================================================

/// `SerializedLayer` から `TernaryWeight` を復元（`ModelArchive::restore_weight` と同等）。
fn restore_weight(layer: &SerializedLayer) -> TernaryWeight {
    TernaryWeight::from_packed(
        layer.packed.clone(),
        layer.out_features,
        layer.in_features,
        layer.scale,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_archive(n_layers: usize) -> (ModelArchive, Vec<u8>) {
        let mut archive = ModelArchive::new();
        for i in 0..n_layers {
            // 各レイヤーは 2x2 のユニーク重み
            let v0 = if i % 2 == 0 { 1i8 } else { -1 };
            let w = TernaryWeight::from_ternary(&[v0, -1, 0, 1], 2, 2);
            let name = format!("layer.{}", i);
            archive.add_layer(&name, &w);
        }
        let data = archive.serialize();
        (archive, data)
    }

    #[test]
    fn test_from_bytes() {
        let (_, data) = make_test_archive(4);
        let config = StreamerConfig { max_hot_layers: 2 };
        let streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        assert_eq!(streamer.layer_count(), 4);
        assert_eq!(streamer.hot_count(), 0);
    }

    #[test]
    fn test_from_bytes_invalid() {
        let result = LayerStreamer::from_bytes(b"garbage", StreamerConfig::default());
        assert!(result.is_none());
    }

    #[test]
    fn test_from_archive() {
        let (archive, _) = make_test_archive(3);
        let streamer = LayerStreamer::from_archive(archive, StreamerConfig::default());
        assert_eq!(streamer.layer_count(), 3);
    }

    #[test]
    fn test_load_single_layer() {
        let (_, data) = make_test_archive(2);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        let w = streamer.load_layer(0).unwrap();
        assert_eq!(w.out_features(), 2);
        assert_eq!(w.in_features(), 2);
        assert!(streamer.is_hot(0));
        assert!(!streamer.is_hot(1));
        assert_eq!(streamer.hot_count(), 1);
    }

    #[test]
    fn test_load_hot_hit() {
        let (_, data) = make_test_archive(2);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        streamer.load_layer(0).unwrap();
        streamer.load_layer(0).unwrap(); // ヒット

        assert_eq!(streamer.stats().hits, 1);
        assert_eq!(streamer.stats().loads, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let (_, data) = make_test_archive(3);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        streamer.load_layer(0).unwrap(); // Hot: [0]
        streamer.load_layer(1).unwrap(); // Hot: [0, 1]
        assert_eq!(streamer.hot_count(), 2);

        // レイヤー 2 をロード → LRU（レイヤー 0）が退去
        streamer.load_layer(2).unwrap(); // Hot: [1, 2], evict 0
        assert!(!streamer.is_hot(0));
        assert!(streamer.is_hot(1));
        assert!(streamer.is_hot(2));
        assert_eq!(streamer.stats().evictions, 1);
    }

    #[test]
    fn test_lru_respects_access_order() {
        let (_, data) = make_test_archive(3);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        streamer.load_layer(0).unwrap(); // clock=1
        streamer.load_layer(1).unwrap(); // clock=2
        streamer.load_layer(0).unwrap(); // clock=3, ヒット → 0 のアクセス更新

        // レイヤー 2 をロード → LRU はレイヤー 1（last_access=2 < 0 の 3）
        streamer.load_layer(2).unwrap();
        assert!(streamer.is_hot(0)); // 最近アクセス → 残る
        assert!(!streamer.is_hot(1)); // 退去
        assert!(streamer.is_hot(2));
    }

    #[test]
    fn test_evict_layer() {
        let (_, data) = make_test_archive(2);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        streamer.load_layer(0).unwrap();
        assert!(streamer.is_hot(0));

        streamer.evict_layer(0);
        assert!(!streamer.is_hot(0));
        assert_eq!(streamer.hot_count(), 0);
    }

    #[test]
    fn test_evict_cold_noop() {
        let (_, data) = make_test_archive(2);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        // Cold レイヤーの退去は何もしない
        streamer.evict_layer(0);
        assert_eq!(streamer.stats().evictions, 0);
    }

    #[test]
    fn test_evict_all() {
        let (_, data) = make_test_archive(3);
        let config = StreamerConfig { max_hot_layers: 3 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        streamer.load_layer(0).unwrap();
        streamer.load_layer(1).unwrap();
        streamer.load_layer(2).unwrap();
        assert_eq!(streamer.hot_count(), 3);

        streamer.evict_all();
        assert_eq!(streamer.hot_count(), 0);
        assert_eq!(streamer.stats().evictions, 3);
    }

    #[test]
    fn test_prefetch_layer() {
        let (_, data) = make_test_archive(2);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        assert!(streamer.prefetch_layer(0));
        assert!(streamer.is_hot(0));

        // 既に Hot なら何もせず true
        assert!(streamer.prefetch_layer(0));
        assert_eq!(streamer.stats().loads, 1); // ロードは 1 回のみ
    }

    #[test]
    fn test_prefetch_out_of_range() {
        let (_, data) = make_test_archive(2);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        assert!(!streamer.prefetch_layer(99));
    }

    #[test]
    fn test_load_out_of_range() {
        let (_, data) = make_test_archive(2);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        assert!(streamer.load_layer(99).is_none());
    }

    #[test]
    fn test_slot_state() {
        let (_, data) = make_test_archive(2);
        let config = StreamerConfig { max_hot_layers: 2 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        assert_eq!(streamer.slot_state(0), Some(SlotState::Cold));
        streamer.load_layer(0).unwrap();
        assert_eq!(streamer.slot_state(0), Some(SlotState::Hot));
        assert_eq!(streamer.slot_state(99), None);
    }

    #[test]
    fn test_hit_rate() {
        let stats = StreamerStats {
            hits: 3,
            loads: 1,
            evictions: 0,
        };
        assert!((stats.hit_rate() - 0.75).abs() < 1e-6);

        let empty = StreamerStats::default();
        assert!((empty.hit_rate() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hot_memory_bytes() {
        let (_, data) = make_test_archive(3);
        let config = StreamerConfig { max_hot_layers: 3 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        assert_eq!(streamer.hot_memory_bytes(), 0);

        streamer.load_layer(0).unwrap();
        let mem1 = streamer.hot_memory_bytes();
        assert!(mem1 > 0);

        streamer.load_layer(1).unwrap();
        let mem2 = streamer.hot_memory_bytes();
        assert!(mem2 > mem1);
    }

    #[test]
    fn test_total_model_bytes() {
        let (_, data) = make_test_archive(3);
        let config = StreamerConfig { max_hot_layers: 3 };
        let streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        // 各レイヤーは 2x2 = 4 weights → 1 packed byte
        assert_eq!(streamer.total_model_bytes(), 3);
    }

    #[test]
    fn test_config_default() {
        let cfg = StreamerConfig::default();
        assert_eq!(cfg.max_hot_layers, 8);
    }

    #[test]
    fn test_reload_after_eviction() {
        let (_, data) = make_test_archive(2);
        let config = StreamerConfig { max_hot_layers: 1 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        streamer.load_layer(0).unwrap();
        streamer.load_layer(1).unwrap(); // 0 退去
        assert!(!streamer.is_hot(0));

        // 再ロード
        let w = streamer.load_layer(0).unwrap();
        assert_eq!(w.out_features(), 2);
        assert!(streamer.is_hot(0));
        assert!(!streamer.is_hot(1)); // 1 が退去
    }

    #[test]
    fn test_inference_through_streamer() {
        // ストリーマー経由でロードした重みで推論
        let mut archive = ModelArchive::new();
        let w = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
        archive.add_layer("fc", &w);
        let data = archive.serialize();

        let config = StreamerConfig { max_hot_layers: 1 };
        let mut streamer = LayerStreamer::from_bytes(&data, config).unwrap();

        let loaded = streamer.load_layer(0).unwrap();

        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];
        crate::ops::ternary_matvec(&input, loaded, &mut output);

        // [1,-1; 0,1] · [2,3] = [-1, 3]
        assert!((output[0] - (-1.0)).abs() < 1e-5);
        assert!((output[1] - 3.0).abs() < 1e-5);
    }
}
