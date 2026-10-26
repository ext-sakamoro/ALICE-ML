//! The ATML format's semantics identifier (ALICE-Law G8): the writer records
//! `ML_SEMANTICS_ID`, the reader compares the recorded identifier and names both on a
//! mismatch, and a version 1 file (no identifier) is legacy, or refused when asked.

use alice_ml::model_io::{ArchiveError, LegacyPolicy, ModelArchive, Provenance, FORMAT_VERSION};
use alice_ml::ops::TernaryWeight;
use alice_ml::semantics::ML_SEMANTICS_ID;
use alice_ml::streaming::{LayerStreamer, StreamerConfig};

fn archive() -> ModelArchive {
    let mut a = ModelArchive::new();
    a.add_layer("fc1", &TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2));
    a.add_layer(
        "fc2",
        &TernaryWeight::from_ternary(&[0, 1, -1, 1, 0, -1], 2, 3),
    );
    a
}

/// A version 1 file as 0.3.0 wrote it: magic, version 1, layer count, layers (no
/// identifier). One 1x4 layer `[1, -1, 0, 1]`, name hash of "fc", scale 1.0.
fn v1_file() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(b"ATML");
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&1u32.to_le_bytes());
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in b"fc" {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    let packed = TernaryWeight::from_ternary(&[1, -1, 0, 1], 1, 4)
        .packed()
        .to_vec();
    v.extend_from_slice(&h.to_le_bytes());
    v.extend_from_slice(&1u32.to_le_bytes());
    v.extend_from_slice(&4u32.to_le_bytes());
    v.extend_from_slice(&1.0f32.to_le_bytes());
    v.extend_from_slice(&u32::try_from(packed.len()).expect("short").to_le_bytes());
    v.extend_from_slice(&packed);
    v
}

#[test]
fn the_writer_records_this_builds_identifier() {
    let bytes = archive().serialize();
    assert_eq!(&bytes[..4], b"ATML");
    assert_eq!(
        u32::from_le_bytes(bytes[4..8].try_into().expect("4")),
        FORMAT_VERSION
    );
    assert_eq!(FORMAT_VERSION, 2);
    assert_eq!(&bytes[8..40], &ML_SEMANTICS_ID);
    let (read, provenance) = ModelArchive::read(&bytes, LegacyPolicy::Refuse).expect("own file");
    assert_eq!(provenance, Provenance::Verified);
    assert_eq!(read.layer_count(), 2);
    assert_eq!(read.find_layer("fc2").expect("fc2").in_features, 3);
}

#[test]
fn another_identifier_is_the_mismatch_naming_both() {
    let mut bytes = archive().serialize();
    bytes[8] ^= 0x01;
    let mut stored = ML_SEMANTICS_ID;
    stored[0] ^= 0x01;
    let err = ModelArchive::read(&bytes, LegacyPolicy::Accept).expect_err("another identifier");
    assert_eq!(
        err,
        ArchiveError::SemanticsMismatch {
            stored,
            expected: ML_SEMANTICS_ID
        }
    );
    // the streaming reader returns the same error, not a silent failure
    assert_eq!(
        LayerStreamer::from_bytes(&bytes, StreamerConfig::default(), LegacyPolicy::Accept).err(),
        Some(ArchiveError::SemanticsMismatch {
            stored,
            expected: ML_SEMANTICS_ID
        })
    );
    // and the message names both
    let msg = err.to_string();
    assert!(
        msg.contains(&format!("{:02x}{:02x}", stored[0], stored[1])),
        "{msg}"
    );
    assert!(
        msg.contains(&format!(
            "{:02x}{:02x}",
            ML_SEMANTICS_ID[0], ML_SEMANTICS_ID[1]
        )),
        "{msg}"
    );
}

/// A file written under the identifier with `alice-det-math` 0.5.1 (`a5472370…`, kept in
/// `PREVIOUS_ML_SEMANTICS_IDS`: every pinned kernel gives the same bits) reads back as
/// `VerifiedPrevious` with the same layers, through both readers
#[test]
fn a_file_with_the_previous_identifier_reads_as_verified_previous() {
    let previous: [u8; 32] = core::array::from_fn(|i| {
        u8::from_str_radix(
            &"a547237044e8fdba9ab152f7af13c431f675e5382e4e8cb40aa8c83c46bb606b"[2 * i..2 * i + 2],
            16,
        )
        .expect("hex")
    });
    assert_ne!(previous, ML_SEMANTICS_ID);
    let mut bytes = archive().serialize();
    bytes[8..40].copy_from_slice(&previous);
    let (read, provenance) =
        ModelArchive::read(&bytes, LegacyPolicy::Refuse).expect("a previous identifier is read");
    assert_eq!(provenance, Provenance::VerifiedPrevious(previous));
    assert_eq!(read.layer_count(), 2);
    let fc2 = read.find_layer("fc2").expect("fc2");
    let own = archive();
    let want = own.find_layer("fc2").expect("fc2");
    assert_eq!(fc2.in_features, want.in_features);
    assert_eq!(fc2.packed, want.packed);
    assert!(
        LayerStreamer::from_bytes(&bytes, StreamerConfig::default(), LegacyPolicy::Refuse).is_ok(),
        "the streaming reader accepts the previous identifier too"
    );
}

#[test]
fn a_version_1_file_is_legacy_or_refused() {
    let v1 = v1_file();
    let (read, provenance) = ModelArchive::read(&v1, LegacyPolicy::Accept).expect("v1 accepted");
    assert_eq!(provenance, Provenance::V1Legacy);
    let layer = read.find_layer("fc").expect("fc");
    assert_eq!((layer.out_features, layer.in_features), (1, 4));
    assert_eq!(layer.scale.to_bits(), 1.0f32.to_bits());
    assert_eq!(
        ModelArchive::read(&v1, LegacyPolicy::Refuse).err(),
        Some(ArchiveError::LegacyRefused)
    );
    // through the streaming reader: legacy is visible, and refused when asked
    let (streamer, provenance) =
        LayerStreamer::from_bytes(&v1, StreamerConfig::default(), LegacyPolicy::Accept)
            .expect("v1 accepted");
    assert_eq!(provenance, Provenance::V1Legacy);
    assert_eq!(streamer.layer_count(), 1);
    assert_eq!(
        LayerStreamer::from_bytes(&v1, StreamerConfig::default(), LegacyPolicy::Refuse).err(),
        Some(ArchiveError::LegacyRefused)
    );
}

#[test]
fn malformed_headers() {
    assert_eq!(
        ModelArchive::read(b"ATM", LegacyPolicy::Accept).err(),
        Some(ArchiveError::TooShort)
    );
    assert_eq!(
        ModelArchive::read(b"XXXX\x02\0\0\0", LegacyPolicy::Accept).err(),
        Some(ArchiveError::BadMagic)
    );
    let bytes = archive().serialize();
    // a version 2 header cut inside the identifier
    assert_eq!(
        ModelArchive::read(&bytes[..20], LegacyPolicy::Accept).err(),
        Some(ArchiveError::TooShort)
    );
    assert_eq!(
        ModelArchive::read(&bytes[..bytes.len() - 1], LegacyPolicy::Accept).err(),
        Some(ArchiveError::Truncated)
    );
}
