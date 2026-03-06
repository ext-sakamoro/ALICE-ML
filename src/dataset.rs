//! Data Loading — Dataset trait とバッチイテレータ。
//!
//! 学習データの抽象化。メモリ上のデータセットを
//! ミニバッチに分割して供給する。

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// データセットの単一サンプル。
#[derive(Debug, Clone)]
pub struct Sample {
    /// 入力特徴量。
    pub features: Vec<f32>,
    /// ターゲット (ラベル or 回帰値)。
    pub target: Vec<f32>,
}

/// インメモリデータセット。
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    /// サンプル一覧。
    samples: Vec<Sample>,
}

impl InMemoryDataset {
    /// 空のデータセットを作成。
    #[must_use]
    pub const fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// サンプルを追加。
    pub fn push(&mut self, features: Vec<f32>, target: Vec<f32>) {
        self.samples.push(Sample { features, target });
    }

    /// サンプル数。
    #[must_use]
    pub const fn len(&self) -> usize {
        self.samples.len()
    }

    /// 空かどうか。
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// インデックスでサンプル取得。
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Sample> {
        self.samples.get(index)
    }

    /// Fisher-Yates シャッフル (決定的シード)。
    ///
    /// 軽量な線形合同生成器で擬似乱数を生成。
    pub fn shuffle(&mut self, seed: u64) {
        let n = self.samples.len();
        if n < 2 {
            return;
        }
        let mut rng = seed;
        for i in (1..n).rev() {
            rng = lcg_next(rng);
            let j = (rng % (i as u64 + 1)) as usize;
            self.samples.swap(i, j);
        }
    }

    /// バッチイテレータを作成。
    #[must_use]
    pub fn batches(&self, batch_size: usize) -> BatchIterator<'_> {
        BatchIterator {
            dataset: self,
            offset: 0,
            batch_size: batch_size.max(1),
        }
    }

    /// 全サンプルのスライス。
    #[must_use]
    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    /// 入力特徴量の次元数 (最初のサンプルから取得)。
    #[must_use]
    pub fn feature_dim(&self) -> usize {
        self.samples.first().map_or(0, |s| s.features.len())
    }

    /// ターゲットの次元数 (最初のサンプルから取得)。
    #[must_use]
    pub fn target_dim(&self) -> usize {
        self.samples.first().map_or(0, |s| s.target.len())
    }
}

impl Default for InMemoryDataset {
    fn default() -> Self {
        Self::new()
    }
}

/// ミニバッチ。
#[derive(Debug)]
pub struct Batch<'a> {
    /// バッチ内のサンプル参照。
    pub samples: &'a [Sample],
}

impl Batch<'_> {
    /// バッチサイズ。
    #[must_use]
    pub const fn len(&self) -> usize {
        self.samples.len()
    }

    /// 空かどうか。
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// 特徴量をフラットな Vec に展開 (row-major)。
    #[must_use]
    pub fn features_flat(&self) -> Vec<f32> {
        self.samples
            .iter()
            .flat_map(|s| s.features.iter().copied())
            .collect()
    }

    /// ターゲットをフラットな Vec に展開 (row-major)。
    #[must_use]
    pub fn targets_flat(&self) -> Vec<f32> {
        self.samples
            .iter()
            .flat_map(|s| s.target.iter().copied())
            .collect()
    }
}

/// バッチイテレータ。
#[derive(Debug)]
pub struct BatchIterator<'a> {
    /// データセットの参照。
    dataset: &'a InMemoryDataset,
    /// 現在のオフセット。
    offset: usize,
    /// バッチサイズ。
    batch_size: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Batch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.dataset.samples.len() {
            return None;
        }
        let end = (self.offset + self.batch_size).min(self.dataset.samples.len());
        let batch = Batch {
            samples: &self.dataset.samples[self.offset..end],
        };
        self.offset = end;
        Some(batch)
    }
}

/// 線形合同生成器 (LCG)。
const fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dataset() {
        let ds = InMemoryDataset::new();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn default_dataset() {
        let ds = InMemoryDataset::default();
        assert!(ds.is_empty());
    }

    #[test]
    fn push_and_get() {
        let mut ds = InMemoryDataset::new();
        ds.push(vec![1.0, 2.0], vec![0.0]);
        ds.push(vec![3.0, 4.0], vec![1.0]);
        assert_eq!(ds.len(), 2);
        let s = ds.get(0).unwrap();
        assert_eq!(s.features, vec![1.0, 2.0]);
        assert_eq!(s.target, vec![0.0]);
        assert!(ds.get(2).is_none());
    }

    #[test]
    fn feature_and_target_dim() {
        let mut ds = InMemoryDataset::new();
        ds.push(vec![1.0, 2.0, 3.0], vec![0.0, 1.0]);
        assert_eq!(ds.feature_dim(), 3);
        assert_eq!(ds.target_dim(), 2);
    }

    #[test]
    fn empty_dims() {
        let ds = InMemoryDataset::new();
        assert_eq!(ds.feature_dim(), 0);
        assert_eq!(ds.target_dim(), 0);
    }

    #[test]
    fn batch_iterator() {
        let mut ds = InMemoryDataset::new();
        for i in 0..10 {
            ds.push(vec![i as f32], vec![0.0]);
        }
        let batches: Vec<_> = ds.batches(3).collect();
        assert_eq!(batches.len(), 4); // 3+3+3+1
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }

    #[test]
    fn batch_exact_division() {
        let mut ds = InMemoryDataset::new();
        for i in 0..6 {
            ds.push(vec![i as f32], vec![0.0]);
        }
        let batches: Vec<_> = ds.batches(3).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
    }

    #[test]
    fn batch_features_flat() {
        let mut ds = InMemoryDataset::new();
        ds.push(vec![1.0, 2.0], vec![0.0]);
        ds.push(vec![3.0, 4.0], vec![1.0]);
        let batch = ds.batches(2).next().unwrap();
        assert_eq!(batch.features_flat(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(batch.targets_flat(), vec![0.0, 1.0]);
    }

    #[test]
    fn batch_empty_is_empty() {
        let ds = InMemoryDataset::new();
        assert!(ds.batches(5).next().is_none());
    }

    #[test]
    fn shuffle_changes_order() {
        let mut ds = InMemoryDataset::new();
        for i in 0..20 {
            ds.push(vec![i as f32], vec![0.0]);
        }
        let original: Vec<f32> = ds.samples().iter().map(|s| s.features[0]).collect();
        ds.shuffle(42);
        let shuffled: Vec<f32> = ds.samples().iter().map(|s| s.features[0]).collect();
        // シャッフル後は少なくとも1箇所は変わるはず
        assert_ne!(original, shuffled);
        // 要素数は同じ
        assert_eq!(ds.len(), 20);
    }

    #[test]
    fn shuffle_deterministic() {
        let mut ds1 = InMemoryDataset::new();
        let mut ds2 = InMemoryDataset::new();
        for i in 0..10 {
            ds1.push(vec![i as f32], vec![0.0]);
            ds2.push(vec![i as f32], vec![0.0]);
        }
        ds1.shuffle(123);
        ds2.shuffle(123);
        let order1: Vec<f32> = ds1.samples().iter().map(|s| s.features[0]).collect();
        let order2: Vec<f32> = ds2.samples().iter().map(|s| s.features[0]).collect();
        assert_eq!(order1, order2);
    }

    #[test]
    fn shuffle_single_element() {
        let mut ds = InMemoryDataset::new();
        ds.push(vec![1.0], vec![0.0]);
        ds.shuffle(0);
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn samples_slice() {
        let mut ds = InMemoryDataset::new();
        ds.push(vec![1.0], vec![0.0]);
        assert_eq!(ds.samples().len(), 1);
    }

    #[test]
    fn batch_size_zero_becomes_one() {
        let mut ds = InMemoryDataset::new();
        ds.push(vec![1.0], vec![0.0]);
        // batch_size=0 は 1 にクランプ
        assert_eq!(ds.batches(0).count(), 1);
    }

    #[test]
    fn lcg_produces_different_values() {
        let a = lcg_next(0);
        let b = lcg_next(a);
        assert_ne!(a, b);
    }
}
