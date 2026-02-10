//! ALICE-DB bridge: Model weight and training metrics persistence
//!
//! Stores ternary model training metrics (loss, accuracy, sparsity)
//! into ALICE-DB time-series for monitoring and checkpoint analysis.

use alice_db::AliceDB;
use std::io;
use std::path::Path;

/// Training metrics sink backed by ALICE-DB.
pub struct TrainingMetricsSink {
    loss_db: AliceDB,
    accuracy_db: AliceDB,
    sparsity_db: AliceDB,
}

impl TrainingMetricsSink {
    /// Open training metrics databases.
    pub fn open<P: AsRef<Path>>(dir: P) -> io::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        Ok(Self {
            loss_db: AliceDB::open(dir.join("loss"))?,
            accuracy_db: AliceDB::open(dir.join("accuracy"))?,
            sparsity_db: AliceDB::open(dir.join("sparsity"))?,
        })
    }

    /// Record training step metrics.
    pub fn record_step(
        &self,
        step: i64,
        loss: f32,
        accuracy: f32,
        sparsity: f32,
    ) -> io::Result<()> {
        self.loss_db.put(step, loss)?;
        self.accuracy_db.put(step, accuracy)?;
        self.sparsity_db.put(step, sparsity)?;
        Ok(())
    }

    /// Record loss only.
    pub fn record_loss(&self, step: i64, loss: f32) -> io::Result<()> {
        self.loss_db.put(step, loss)
    }

    /// Batch record losses.
    pub fn record_loss_batch(&self, data: &[(i64, f32)]) -> io::Result<()> {
        self.loss_db.put_batch(data)
    }

    /// Query loss history.
    pub fn query_loss(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.loss_db.scan(start, end)
    }

    /// Query accuracy history.
    pub fn query_accuracy(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.accuracy_db.scan(start, end)
    }

    /// Query sparsity history.
    pub fn query_sparsity(&self, start: i64, end: i64) -> io::Result<Vec<(i64, f32)>> {
        self.sparsity_db.scan(start, end)
    }

    /// Flush all databases.
    pub fn flush(&self) -> io::Result<()> {
        self.loss_db.flush()?;
        self.accuracy_db.flush()?;
        self.sparsity_db.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_training_metrics_roundtrip() {
        let dir = tempdir().unwrap();
        let sink = TrainingMetricsSink::open(dir.path()).unwrap();

        for step in 0..50 {
            let loss = 1.0 / (step as f32 + 1.0);
            sink.record_step(step, loss, 0.9, 0.33).unwrap();
        }
        sink.flush().unwrap();

        let losses = sink.query_loss(0, 49).unwrap();
        assert!(!losses.is_empty());
    }
}
