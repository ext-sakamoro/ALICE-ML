//! Probability calibration — measuring it, and fixing it
//!
//! A model that says 0.8 should be right about 80% of the time. Accuracy does
//! not tell you whether it is: a model can pick the right answer often and still
//! be wrong about how sure it was, which is exactly the failure that matters
//! when something downstream routes on confidence
//!
//! # What is here
//!
//! **Measuring** — [`brier_binary`] / [`brier_multiclass`] (proper scoring
//! rules), [`negative_log_likelihood`], and [`ReliabilityBins`] for expected and
//! maximum calibration error
//!
//! **Fixing** — three post-hoc calibrators, in increasing order of what they can
//! move and of how much held-out data they need:
//!
//! | calibrator | parameters | changes the argmax? |
//! |---|---|---|
//! | [`TemperatureScaling`] | 1 | no |
//! | [`VectorScaling`] | `2K` | yes |
//! | [`IsotonicRegression`] | piecewise, binary only | yes |
//!
//! Temperature scaling divides every logit by one number, so it can only make a
//! distribution flatter or sharper — the ranking is untouched. That is usually
//! what you want: it fixes over-confidence without second-guessing which answer
//! the model picked
//!
//! # Why the transcendentals come from `alice_det_math`
//!
//! The platform `exp` and `ln` differ between targets in the last bits. A
//! calibration fitted on one machine and applied on another would then produce
//! slightly different probabilities from the same logits, which defeats the
//! point of having calibrated them at all
//!
//! # Accumulation
//!
//! Sums run in `f64` even though the inputs and outputs are `f32`: a Brier score
//! over a few hundred thousand samples loses digits in `f32` long before the
//! quantity itself stops being meaningful

// `a * b + c` は 2 回丸めのまま置く clippy は `mul_add` を勧めてくるが、
// それは FMA のある target でだけ 1 回丸めになり、同じ logits から違う校正値が
// 出る この module が det_math を使っているのと同じ理由で従わない
// (feedback_mul_add_breaks_bit_exactness)
#![allow(clippy::suboptimal_flops)]

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};
use core::fmt;

/// Probabilities below this are treated as this when a logarithm is taken
///
/// `ln(0)` is `-inf`, and one impossible-but-observed sample would otherwise
/// make the whole average infinite. The bound is the smallest value that still
/// round-trips through `f32` comfortably
pub const MIN_PROBABILITY: f64 = 1e-12;

/// The narrowest and widest temperatures [`TemperatureScaling::fit`] will search
///
/// Below the lower bound the distribution collapses to a point mass; above the
/// upper it is uniform for any realistic logit spread
pub const TEMPERATURE_RANGE: (f32, f32) = (0.05, 20.0);

/// What calibration refuses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CalibrationError {
    /// A probability vector was empty
    EmptyDistribution,
    /// The class index was outside the distribution
    LabelOutOfRange {
        /// The label supplied
        label: usize,
        /// The number of classes
        classes: usize,
    },
    /// A probability or logit was `NaN` or infinite, or a probability was
    /// outside `0.0..=1.0`
    NotAProbability,
    /// Two inputs that must line up had different lengths
    LengthMismatch {
        /// What was expected
        expected: usize,
        /// What arrived
        got: usize,
    },
    /// A temperature was not finite and positive
    TemperatureOutOfRange,
    /// Fewer bins than one, or more than [`ReliabilityBins::MAX_BINS`]
    BinCountOutOfRange {
        /// The count supplied
        got: usize,
    },
    /// There was nothing to fit on
    NoSamples,
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyDistribution => f.write_str("the distribution had no classes"),
            Self::LabelOutOfRange { label, classes } => {
                write!(f, "label {label} is outside a {classes}-class distribution")
            }
            Self::NotAProbability => f.write_str("a value was NaN, infinite, or outside 0.0..=1.0"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "expected {expected} values, got {got}")
            }
            Self::TemperatureOutOfRange => {
                f.write_str("temperature must be finite and greater than zero")
            }
            Self::BinCountOutOfRange { got } => write!(
                f,
                "bin count must be 1..={}, got {got}",
                ReliabilityBins::MAX_BINS
            ),
            Self::NoSamples => f.write_str("there were no samples to fit on"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CalibrationError {}

fn check_distribution(probs: &[f32], label: usize) -> Result<(), CalibrationError> {
    if probs.is_empty() {
        return Err(CalibrationError::EmptyDistribution);
    }
    if label >= probs.len() {
        return Err(CalibrationError::LabelOutOfRange {
            label,
            classes: probs.len(),
        });
    }
    for &p in probs {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(CalibrationError::NotAProbability);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Measuring
// ---------------------------------------------------------------------------

/// Brier score for a yes/no prediction: `(p - y)^2`, so `0.0` is perfect and
/// `1.0` is confidently wrong
///
/// # Errors
///
/// [`CalibrationError::NotAProbability`] if `p` is not a probability
pub fn brier_binary(p: f32, label: bool) -> Result<f32, CalibrationError> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(CalibrationError::NotAProbability);
    }
    let y = f64::from(u8::from(label));
    let d = f64::from(p) - y;
    Ok((d * d) as f32)
}

/// Brier score for a distribution over classes: `sum_k (p_k - y_k)^2`
///
/// The range is `0.0..=2.0` — two, not one, because a confidently wrong answer
/// is wrong twice over: it put one unit of mass where it should not have, and
/// left one unit missing where it should have been
///
/// # Errors
///
/// [`CalibrationError::EmptyDistribution`] for no classes,
/// [`CalibrationError::LabelOutOfRange`] for a label outside the distribution,
/// and [`CalibrationError::NotAProbability`] for a value that is not one
pub fn brier_multiclass(probs: &[f32], label: usize) -> Result<f32, CalibrationError> {
    check_distribution(probs, label)?;
    let mut acc = 0.0_f64;
    for (k, &p) in probs.iter().enumerate() {
        let y = f64::from(u8::from(k == label));
        let d = f64::from(p) - y;
        acc += d * d;
    }
    Ok(acc as f32)
}

/// `-ln(p_label)`, the negative log likelihood of one sample
///
/// Probabilities below [`MIN_PROBABILITY`] are treated as [`MIN_PROBABILITY`],
/// so one impossible-but-observed sample gives a large penalty rather than an
/// infinite one
///
/// # Errors
///
/// As [`brier_multiclass`]
pub fn negative_log_likelihood(probs: &[f32], label: usize) -> Result<f32, CalibrationError> {
    check_distribution(probs, label)?;
    let p = f64::from(probs[label]).max(MIN_PROBABILITY);
    Ok((-alice_det_math::ln64(p)) as f32)
}

/// One bucket of a reliability diagram
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Bin {
    /// How many samples landed here
    pub count: u32,
    /// The mean confidence of those samples
    pub confidence: f32,
    /// The share of them that were right
    pub accuracy: f32,
}

/// Confidence bucketed against outcome — the thing calibration is *about*
///
/// A perfectly calibrated model has `accuracy == confidence` in every bin.
/// [`ReliabilityBins::expected_calibration_error`] is the count-weighted mean of
/// the gap, and [`ReliabilityBins::maximum_calibration_error`] is the worst one
#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityBins {
    /// per-bin (count, confidence sum, correct count), in `f64` to keep the sums
    counts: Vec<u32>,
    confidence_sums: Vec<f64>,
    correct: Vec<u32>,
}

impl ReliabilityBins {
    /// The most bins allowed — beyond this each bin holds too few samples for
    /// its accuracy to mean anything on any realistic evaluation set
    pub const MAX_BINS: usize = 1024;

    /// Builds an empty set of `bins` equal-width buckets over `0.0..=1.0`
    ///
    /// Ten or fifteen is the usual choice
    ///
    /// # Errors
    ///
    /// [`CalibrationError::BinCountOutOfRange`] for zero or more than
    /// [`ReliabilityBins::MAX_BINS`]
    pub fn with_bins(bins: usize) -> Result<Self, CalibrationError> {
        if bins == 0 || bins > Self::MAX_BINS {
            return Err(CalibrationError::BinCountOutOfRange { got: bins });
        }
        Ok(Self {
            counts: vec![0; bins],
            confidence_sums: vec![0.0; bins],
            correct: vec![0; bins],
        })
    }

    /// Records one prediction
    ///
    /// # Errors
    ///
    /// [`CalibrationError::NotAProbability`] if `confidence` is not a probability
    pub fn observe(&mut self, confidence: f32, correct: bool) -> Result<(), CalibrationError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(CalibrationError::NotAProbability);
        }
        let n = self.counts.len();
        // `1.0` belongs in the last bin rather than in a bin of its own
        let raw = (f64::from(confidence) * n as f64) as usize;
        let index = raw.min(n - 1);
        self.counts[index] += 1;
        self.confidence_sums[index] += f64::from(confidence);
        self.correct[index] += u32::from(correct);
        Ok(())
    }

    /// Records the top-1 prediction of a distribution
    ///
    /// # Errors
    ///
    /// As [`brier_multiclass`]
    pub fn observe_distribution(
        &mut self,
        probs: &[f32],
        label: usize,
    ) -> Result<(), CalibrationError> {
        check_distribution(probs, label)?;
        let mut best = 0;
        for (k, p) in probs.iter().enumerate().skip(1) {
            if *p > probs[best] {
                best = k;
            }
        }
        self.observe(probs[best], best == label)
    }

    /// How many samples have been recorded
    #[must_use]
    pub fn total(&self) -> u32 {
        self.counts.iter().sum()
    }

    /// Whether nothing has been recorded
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// The bins, lowest confidence first
    #[must_use]
    pub fn bins(&self) -> Vec<Bin> {
        self.counts
            .iter()
            .zip(&self.confidence_sums)
            .zip(&self.correct)
            .map(|((&count, &confidence_sum), &correct)| {
                if count == 0 {
                    Bin::default()
                } else {
                    let n = f64::from(count);
                    Bin {
                        count,
                        confidence: (confidence_sum / n) as f32,
                        accuracy: (f64::from(correct) / n) as f32,
                    }
                }
            })
            .collect()
    }

    /// Count-weighted mean of `|accuracy - confidence|` over non-empty bins
    ///
    /// `0.0` for a perfectly calibrated model, and `0.0` for an empty set —
    /// there is no error in predictions nobody made
    #[must_use]
    pub fn expected_calibration_error(&self) -> f32 {
        let total = f64::from(self.total());
        if total == 0.0 {
            return 0.0;
        }
        let mut acc = 0.0_f64;
        for ((&count, &confidence_sum), &correct) in self
            .counts
            .iter()
            .zip(&self.confidence_sums)
            .zip(&self.correct)
        {
            if count == 0 {
                continue;
            }
            let n = f64::from(count);
            let gap = (f64::from(correct) / n - confidence_sum / n).abs();
            acc += n * gap;
        }
        (acc / total) as f32
    }

    /// The largest `|accuracy - confidence|` of any non-empty bin
    #[must_use]
    pub fn maximum_calibration_error(&self) -> f32 {
        let mut worst = 0.0_f64;
        for ((&count, &confidence_sum), &correct) in self
            .counts
            .iter()
            .zip(&self.confidence_sums)
            .zip(&self.correct)
        {
            if count == 0 {
                continue;
            }
            let n = f64::from(count);
            let gap = (f64::from(correct) / n - confidence_sum / n).abs();
            if gap > worst {
                worst = gap;
            }
        }
        worst as f32
    }
}

// ---------------------------------------------------------------------------
// Softmax used by the calibrators
// ---------------------------------------------------------------------------

/// `softmax(logits / temperature)` into `out`
fn softmax_scaled(logits: &[f32], temperature: f64, out: &mut [f64]) {
    let mut max = f64::NEG_INFINITY;
    for &z in logits {
        let scaled = f64::from(z) / temperature;
        if scaled > max {
            max = scaled;
        }
    }
    let mut total = 0.0_f64;
    for (o, &z) in out.iter_mut().zip(logits) {
        let e = alice_det_math::exp64(f64::from(z) / temperature - max);
        *o = e;
        total += e;
    }
    if total > 0.0 {
        for o in out.iter_mut() {
            *o /= total;
        }
    }
}

fn check_logits(logits: &[f32], label: usize) -> Result<(), CalibrationError> {
    if logits.is_empty() {
        return Err(CalibrationError::EmptyDistribution);
    }
    if label >= logits.len() {
        return Err(CalibrationError::LabelOutOfRange {
            label,
            classes: logits.len(),
        });
    }
    if logits.iter().any(|z| !z.is_finite()) {
        return Err(CalibrationError::NotAProbability);
    }
    Ok(())
}

/// Golden ratio conjugate 黄金分割探索の縮小率で、書き下しておくことで探索が
/// toolchain に依らず同じ点を評価する
const INV_PHI: f64 = 0.618_033_988_749_894_8;

/// Mean NLL of `samples` under `softmax(logits / t)`
fn mean_nll(samples: &[(&[f32], usize)], t: f64) -> f64 {
    let mut acc = 0.0_f64;
    let mut scratch: Vec<f64> = Vec::new();
    for (logits, label) in samples {
        scratch.clear();
        scratch.resize(logits.len(), 0.0);
        softmax_scaled(logits, t, &mut scratch);
        acc -= alice_det_math::ln64(scratch[*label].max(MIN_PROBABILITY));
    }
    acc / samples.len() as f64
}

// ---------------------------------------------------------------------------
// Temperature scaling
// ---------------------------------------------------------------------------

/// One number that makes every distribution flatter (`t > 1`) or sharper
/// (`t < 1`)
///
/// It cannot change which class wins, because dividing every logit by the same
/// positive number preserves their order. That is the point: it fixes how sure
/// the model is without touching what it chose
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureScaling {
    temperature: f32,
}

impl TemperatureScaling {
    /// `t = 1`, which leaves the distribution exactly as it was
    pub const IDENTITY: Self = Self { temperature: 1.0 };

    /// Wraps a temperature
    ///
    /// # Errors
    ///
    /// [`CalibrationError::TemperatureOutOfRange`] unless `t` is finite and
    /// greater than zero
    pub fn try_new(t: f32) -> Result<Self, CalibrationError> {
        if !t.is_finite() || t <= 0.0 {
            return Err(CalibrationError::TemperatureOutOfRange);
        }
        Ok(Self { temperature: t })
    }

    /// The temperature in force
    #[must_use]
    pub const fn temperature(self) -> f32 {
        self.temperature
    }

    /// Writes `softmax(logits / t)` into `out`
    ///
    /// # Errors
    ///
    /// [`CalibrationError::EmptyDistribution`] for no logits,
    /// [`CalibrationError::NotAProbability`] for a non-finite logit, and
    /// [`CalibrationError::LengthMismatch`] if `out` is a different length
    pub fn apply(&self, logits: &[f32], out: &mut [f32]) -> Result<(), CalibrationError> {
        check_logits(logits, 0)?;
        if out.len() != logits.len() {
            return Err(CalibrationError::LengthMismatch {
                expected: logits.len(),
                got: out.len(),
            });
        }
        let mut scratch = vec![0.0_f64; logits.len()];
        softmax_scaled(logits, f64::from(self.temperature), &mut scratch);
        for (o, s) in out.iter_mut().zip(&scratch) {
            *o = *s as f32;
        }
        Ok(())
    }

    /// Finds the temperature that minimises mean NLL on `samples`
    ///
    /// The NLL of a softmax as a function of `1/t` is convex, so a golden-section
    /// search over [`TEMPERATURE_RANGE`] finds the minimum without derivatives
    /// and without a learning rate to tune
    ///
    /// # Errors
    ///
    /// [`CalibrationError::NoSamples`] for an empty set, plus whatever the
    /// individual samples are rejected for
    pub fn fit(samples: &[(&[f32], usize)]) -> Result<Self, CalibrationError> {
        if samples.is_empty() {
            return Err(CalibrationError::NoSamples);
        }
        for (logits, label) in samples {
            check_logits(logits, *label)?;
        }

        let (lo_bound, hi_bound) = TEMPERATURE_RANGE;
        let (mut lo, mut hi) = (f64::from(lo_bound), f64::from(hi_bound));
        let mut c = hi - (hi - lo) * INV_PHI;
        let mut d = lo + (hi - lo) * INV_PHI;
        let mut fc = mean_nll(samples, c);
        let mut fd = mean_nll(samples, d);
        // 80 iterations shrinks the bracket by 0.618^80, far below f32 resolution
        for _ in 0..80 {
            if fc < fd {
                hi = d;
                d = c;
                fd = fc;
                c = hi - (hi - lo) * INV_PHI;
                fc = mean_nll(samples, c);
            } else {
                lo = c;
                c = d;
                fc = fd;
                d = lo + (hi - lo) * INV_PHI;
                fd = mean_nll(samples, d);
            }
        }
        Self::try_new(lo.midpoint(hi) as f32)
    }
}

// ---------------------------------------------------------------------------
// Vector scaling
// ---------------------------------------------------------------------------

/// A slope and an offset per class: `z'_k = a_k * z_k + b_k`
///
/// Unlike [`TemperatureScaling`] this **can** change which class wins, which is
/// what makes it able to fix a model that is systematically shy about one class
/// — and also what makes it able to overfit. It has `2K` parameters, so it wants
/// a held-out set with room to spare
#[derive(Debug, Clone, PartialEq)]
pub struct VectorScaling {
    slopes: Vec<f32>,
    offsets: Vec<f32>,
}

impl VectorScaling {
    /// `a = 1`, `b = 0` for every class — the identity
    ///
    /// # Errors
    ///
    /// [`CalibrationError::EmptyDistribution`] for zero classes
    pub fn identity(classes: usize) -> Result<Self, CalibrationError> {
        if classes == 0 {
            return Err(CalibrationError::EmptyDistribution);
        }
        Ok(Self {
            slopes: vec![1.0; classes],
            offsets: vec![0.0; classes],
        })
    }

    /// The per-class slopes
    #[must_use]
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }

    /// The per-class offsets
    #[must_use]
    pub fn offsets(&self) -> &[f32] {
        &self.offsets
    }

    /// Writes `softmax(a * logits + b)` into `out`
    ///
    /// # Errors
    ///
    /// [`CalibrationError::LengthMismatch`] if the logits or `out` do not have
    /// one entry per class, plus the usual rejections
    pub fn apply(&self, logits: &[f32], out: &mut [f32]) -> Result<(), CalibrationError> {
        if logits.len() != self.slopes.len() {
            return Err(CalibrationError::LengthMismatch {
                expected: self.slopes.len(),
                got: logits.len(),
            });
        }
        if out.len() != logits.len() {
            return Err(CalibrationError::LengthMismatch {
                expected: logits.len(),
                got: out.len(),
            });
        }
        check_logits(logits, 0)?;
        let mut adjusted = vec![0.0_f32; logits.len()];
        for ((a, o), (&s, &b)) in adjusted
            .iter_mut()
            .zip(logits)
            .zip(self.slopes.iter().zip(&self.offsets))
        {
            *a = s * *o + b;
        }
        let mut scratch = vec![0.0_f64; logits.len()];
        softmax_scaled(&adjusted, 1.0, &mut scratch);
        for (o, s) in out.iter_mut().zip(&scratch) {
            *o = *s as f32;
        }
        Ok(())
    }

    /// Fits the slopes and offsets by gradient descent on mean NLL
    ///
    /// The gradient of the softmax cross-entropy is exactly `(p_k - y_k)`, so
    /// `dL/da_k = (p_k - y_k) * z_k` and `dL/db_k = (p_k - y_k)`, averaged over
    /// the samples. No numerical differentiation is involved
    ///
    /// # Errors
    ///
    /// [`CalibrationError::NoSamples`] for an empty set, and
    /// [`CalibrationError::LengthMismatch`] if the samples disagree on the
    /// number of classes
    pub fn fit(
        samples: &[(&[f32], usize)],
        steps: usize,
        learning_rate: f32,
    ) -> Result<Self, CalibrationError> {
        if samples.is_empty() {
            return Err(CalibrationError::NoSamples);
        }
        let classes = samples[0].0.len();
        for (logits, label) in samples {
            if logits.len() != classes {
                return Err(CalibrationError::LengthMismatch {
                    expected: classes,
                    got: logits.len(),
                });
            }
            check_logits(logits, *label)?;
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(CalibrationError::NotAProbability);
        }

        let mut fitted = Self::identity(classes)?;
        let lr = f64::from(learning_rate);
        let n = samples.len() as f64;
        let mut probs = vec![0.0_f64; classes];
        let mut adjusted = vec![0.0_f32; classes];
        let mut grad_a = vec![0.0_f64; classes];
        let mut grad_b = vec![0.0_f64; classes];

        for _ in 0..steps {
            grad_a.fill(0.0);
            grad_b.fill(0.0);

            for (logits, label) in samples {
                for ((a, o), (&s, &b)) in adjusted
                    .iter_mut()
                    .zip(*logits)
                    .zip(fitted.slopes.iter().zip(&fitted.offsets))
                {
                    *a = s * *o + b;
                }
                softmax_scaled(&adjusted, 1.0, &mut probs);
                for k in 0..classes {
                    let residual = probs[k] - f64::from(u8::from(k == *label));
                    grad_a[k] += residual * f64::from(logits[k]);
                    grad_b[k] += residual;
                }
            }

            for k in 0..classes {
                fitted.slopes[k] -= (lr * grad_a[k] / n) as f32;
                fitted.offsets[k] -= (lr * grad_b[k] / n) as f32;
            }
        }
        Ok(fitted)
    }
}

// ---------------------------------------------------------------------------
// Isotonic regression (pool adjacent violators)
// ---------------------------------------------------------------------------

/// A non-decreasing step function fitted to (score, outcome) pairs
///
/// Where [`TemperatureScaling`] assumes the miscalibration has one shape,
/// isotonic regression assumes only that a higher score should not mean a lower
/// probability, and finds the best non-decreasing fit under squared error. It is
/// binary: one score in, one probability out
#[derive(Debug, Clone, PartialEq)]
pub struct IsotonicRegression {
    /// The right-hand edge of each block, ascending
    edges: Vec<f32>,
    /// The fitted probability of each block, non-decreasing
    values: Vec<f32>,
}

impl IsotonicRegression {
    /// Fits by pool adjacent violators
    ///
    /// The samples are sorted by score, then any adjacent pair that decreases is
    /// merged into one block holding their mean, repeatedly, until the sequence
    /// is non-decreasing. That is the exact least-squares isotonic fit, not an
    /// approximation of it
    ///
    /// # Errors
    ///
    /// [`CalibrationError::NoSamples`] for an empty set, and
    /// [`CalibrationError::NotAProbability`] for a non-finite score
    pub fn fit(samples: &[(f32, bool)]) -> Result<Self, CalibrationError> {
        if samples.is_empty() {
            return Err(CalibrationError::NoSamples);
        }
        if samples.iter().any(|(s, _)| !s.is_finite()) {
            return Err(CalibrationError::NotAProbability);
        }

        let mut sorted: Vec<(f32, bool)> = samples.to_vec();
        // ties keep their input order, so the fit is a function of the input
        sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

        // blocks of (right edge, sum of outcomes, count)
        let mut edges: Vec<f32> = Vec::with_capacity(sorted.len());
        let mut sums: Vec<f64> = Vec::with_capacity(sorted.len());
        let mut counts: Vec<f64> = Vec::with_capacity(sorted.len());

        for (score, outcome) in sorted {
            edges.push(score);
            sums.push(f64::from(u8::from(outcome)));
            counts.push(1.0);
            // pool while the last block is below the one before it
            while sums.len() >= 2 {
                let last = sums.len() - 1;
                let mean_last = sums[last] / counts[last];
                let mean_prev = sums[last - 1] / counts[last - 1];
                if mean_prev <= mean_last {
                    break;
                }
                sums[last - 1] += sums[last];
                counts[last - 1] += counts[last];
                edges[last - 1] = edges[last];
                sums.pop();
                counts.pop();
                edges.pop();
            }
        }

        let values: Vec<f32> = sums
            .iter()
            .zip(&counts)
            .map(|(s, c)| (s / c) as f32)
            .collect();
        Ok(Self { edges, values })
    }

    /// The fitted probability for `score`
    ///
    /// Scores below the first block get the first value and scores above the
    /// last get the last: the fit says nothing about what happens outside the
    /// range it was shown, so it holds the nearest answer rather than
    /// extrapolating
    #[must_use]
    pub fn apply(&self, score: f32) -> f32 {
        self.edges
            .iter()
            .position(|&e| score <= e)
            .map_or_else(|| self.values[self.values.len() - 1], |i| self.values[i])
    }

    /// The number of blocks the fit collapsed to
    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.values.len()
    }

    /// The fitted values, non-decreasing
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}
