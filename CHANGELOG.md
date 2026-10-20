# Changelog

All notable changes to ALICE-ML will be documented in this file.

## [Unreleased]

### Added
- `conformance/g11-own.toml`: ALICE-Law G11 の表のうち、この crate 固有の行 (G4 / G5 の golden、G6 の識別子、G7 / G8 の ATML の識別子の照合、比較 0 件は `scripts/run_tests.sh` の試験) 共有の検査器の行は ALICE-Law の `conformance/g11-shared-rows.toml` から生成する
- CI (ALICE-Law G11): test job の Law edge goldens (debug、全 native feature) の `scripts/run_tests.sh` に `ml_semantics_id` / `atml_semantics` / `g7_boundaries` を足した 識別子と ATML の試験の binary が 1 件も走らない時に step が落ちる `tests/determinism_golden.rs` の照合を `Golden::mismatch` に分け、`checker_rejects_a_wrong_digest` が動いた bit (-0 から +0) の digest が記録と一致しないことを確かめる (照合を常に一致にする変異でこの試験だけが失敗する)
- CI (ALICE-Law G11 (c)): ほかの ALICE の crate と同じ bytes で持つ file を ALICE-Law の表と比べる `scripts/shared_files.sh` (試験 `scripts/test_shared_files.py`、ALICE-DetMath などと同じ中身) と `.github/shared-files-pin` を足し、float-environment job (3 OS) と preflight で走らせる 表と検査器は pin の版で取得し SHA-256 を確かめる 差、表に無い共有の path、比較 0 件は失敗
- CI: text の file の行末の検査 `scripts/check_eol.py` (試験 `scripts/test_check_eol.py`) と `.github/workflows/eol.yml` を足した `git ls-files --eol` で text の file が index か working tree で CRLF / 混在なら失敗 (`-text` と binary は除く、比べた数を出し 0 件は失敗)、workflow は ubuntu / macos / windows で走り、preflight にも同じ検査を足した (ALICE の crate で同じ中身の file)
- `alice-law-conformance.toml` の G9 を pass にした 公開版 (crates.io の 0.2.0 と tag alice-ml-v0.2.0 の 1 つだけ、0.3.0 は未公開) と定義を変えた commit で識別子の履歴を確かめ、note に書いた: 0.2.0 には識別子が無く、`ML_SEMANTICS_ID` は 0.4.0 で初めて出る (`a5472370…`、ATML 版 2 が記録する) ので、公開版との比較では追加になる
- 試験 `tests/float_edge_golden.rs` (ALICE-Law G4): Law の site の符号付きゼロとちょうどの打ち消しを bit (`to_bits`) で固定する `law_math` の max / min / max64 / min64 の符号規則と、その上の clamp / relu / max_of / min_of / argmax / tensor_max / tensor_min、等しい logit の softmax (exact / fast、通常・1e30・符号付きゼロ) が各要素ちょうど 1/n、恒等温度の `TemperatureScaling::apply`、`negative_log_likelihood` と `cross_entropy_loss` の 36 件 値は Law の値でもあることを別の試験で確かめる CI の test job (x86_64) と neon job (aarch64) で debug、新しい `.github/workflows/release-profile.yml` (ubuntu-latest x86_64 と macos-latest aarch64、ALICE-Law G5) で release に走らせる (`scripts/run_tests.sh` が件数 0 と下限未満を落とす) x86_64 の leg は AVX2 の lane を被覆するとは言わない (log に CPU の flag を出す) `alice-law-conformance.toml` の G4・G5 を pass にした 出力の bit と識別子は変わらない

- CI (ALICE-Law G9): `scripts/id_move_check.py` (ALICE の識別子を持つ crate で共有、ALICE-Zip と同じ file) を追加 最新の release tag の公開版と作業 tree で `ML_SEMANTICS_ID` の値を読み、動いたのに破壊的な版になっていない / 公開版の値が `PREVIOUS_ML_SEMANTICS_IDS` に無い / CHANGELOG が両方の値を書いていない時に fail する (ci.yml の clippy job と preflight)
- CI: README の ALICE-Law 節の検査 (law-readme-block.yml) を preflight でも同じに走らせる `scripts/law_readme_block_check.py` と試験 `scripts/test_law_readme_block_check.py` を足した (ALICE-DB と同じ中身の共有 file、workflow の check の引数と節の pin の生成器を使い、比較 0 件・件数の行なし・pin が無い・fetch の失敗は失敗、未 commit の入力は検査していないと表示) 試験は CI と preflight で走る
- CI (ALICE-Law G3): fma の結果全体を negate する形の検査を追加 (ALICE-LOL と同一の構文木の前段検査 `tools/no-outer-fneg-of-fma` を `--locked` で、最適化後の LLVM IR の検査 `scripts/fneg_fma_ir_check.py` と試験、47 個の probe) library の source に fma の呼び出しは無く、test に 2 件 (どちらも negate しない)、`assert!` の中の 1 件は理由付きで allowlist に置いた IR の検査は新しい `.github/workflows/g3-ir.yml` が x86_64 (ubuntu-latest) と aarch64 (ubuntu-24.04-arm) の実機 runner で実行する (pyo3 は cross build できない) 実行可否は `scripts/changed_paths.py` (ALICE-DetMath と同一) と `.github/paths/g3-ir.txt` で merge-base(origin/main) から決め、走るべき job が走らなかった時は gate job が red になる `.gitignore` で検査器の `Cargo.lock` を追跡対象にした `alice-law-conformance.toml` の G3 を unverified から pass にした

### Changed
- CI: `.github/workflows/eol.yml` に top-level の concurrency group を足し (同じ branch の古い run は止め、main の run は commit ごとに残す)、action を commit SHA で固定して checkout に `persist-credentials: false` を付け、`workflow_dispatch` で手動でも走らせられるようにした (ALICE の crate で同じ中身の file、workflow_concurrency.py と workflow_pin_check.py を持つ crate の検査に合わせた)

- CI: `.gitattributes` に `* text=auto eol=lf` を置き、text の file を保存も checkout も全 OS で LF にした (共有 file は ALICE-Law の表と byte で比べるので、Windows の CRLF の checkout ではずれになる) binary と既存の行はそのまま、`git add --renormalize` で commit 済みの中身は変わらない
- CI (ALICE-Law G3): `tools/no-outer-fneg-of-fma` と `scripts/fneg_fma_ir_check.py` (と試験) を ALICE-LOL と同じ新しい版にした `expected-src-fma-calls` は src/ の library code の fma 呼び出し数 (`#[test]` と test の時だけ成り立つ cfg の下の item と文、test-only の `mod y;` で読み込む file とその子は数えず、test-only の数を別に表示する) この repository の数は変わらない IR 検査は library の source に fma が無いのに IR に fma がある時を失敗にし、library の fma が 0 の時は共有の probe を target ごとに compile して判定を確かめる (0 件実行は失敗) tool は独自の workspace なので repository 全体の format 検査に入っていなかった CI と preflight に tool の format の検査を足した ci.yml は `ci/**` の branch の push でも走るようにした (他の crate と同じ、main に入れる前に同じ検査を branch で走らせる)

## [0.4.0] - 2026-10-17

ALICE-Law の wave: G1 (pyo3 の lint)、G2 (最大・最小・clamp と tie 規則)、G6 (`ML_SEMANTICS_ID`)、G7 (型が消える境界)、G8 (ATML version 2)、G13 (順序の決まらない集合と並列の禁止) 永続化の形式が変わる (ATML version 2、G9 の規則で major) ので 0.3 から上げる

### Added
- README.md / README_JP.md に ALICE-Law の `gen-law-readme-block.py` が生成する節 (release train、算術の識別子、G1-G14 の状態、feature ごとの保証) を置き、`.github/workflows/law-readme-block.yml` が block に書いた ALICE-Law の revision で再生成して差分を検査する README_JP.md の License の見出しを「ライセンス」にした (生成器が block を置く位置)
- ALICE-Law G7: 型が消える境界 (C ABI 67、Python 13、byte の reader 3、`dyn Any` 1、alice-db の呼び出し 1) を `src/` から数え、それぞれ他の Law を持つ crate の値を受け取らない理由と件数を固定する `tests/g7_boundaries.rs` を足した 新しい境界・`build.rs`・procedural macro・他の alice-* crate の decoder の呼び出しが現れると失敗し、source を 1 つも読まなければ失敗する
- ALICE-Law G13: `clippy.toml` で `std::collections::HashMap` / `HashSet` を禁止した (canary 68 行) 並列の float 経路 `ops::ternary_matmul_batch` (`parallel`) が 1・2・8 thread で直列の matvec と同じ bit になることを確かめ、`src/` に他の rayon の site が現れたら失敗する `tests/parallel_order.rs` を足した (CI は `run_tests.sh parallel_order=2` で件数も見る)
- ALICE-Law G8: ATML の format version 2 を足した version の直後に書いた build の `ML_SEMANTICS_ID` (32 byte) を記録する `ModelArchive::read(data, LegacyPolicy)` が記録された識別子を比べ、自分のものなら `Provenance::Verified`、`PREVIOUS_ML_SEMANTICS_IDS` にあれば `VerifiedPrevious`、それ以外は両方の識別子を持つ `ArchiveError::SemanticsMismatch` を返す 識別子の無い version 1 の file は `Provenance::V1Legacy` として読むか、`LegacyPolicy::Refuse` で `ArchiveError::LegacyRefused` にする `tests/atml_semantics.rs`
- ALICE-Law G6: `alice_ml::semantics::ML_SEMANTICS_ID` を足した `tests/determinism_golden.rs` の 9 本の golden の digest と alice-det-math の `SEMANTICS_ID` を名前順・長さ付きで SHA-256 に畳んだ値で、解決した alice-det-math の世代ごとの表 `ML_SEMANTICS_BY_DET_MATH` から引く (行の無い世代は compile が止まる) crate の version は畳まない `tests/ml_semantics_id.rs` が golden の source から digest を読み、表の全行を再計算する
- ALICE-Law G2 の tie 規則: 最大・最小・clamp・選択の各 site を比較式に書き換えた変異 (両方の operand 順) が試験で落ちることを確かめる `scripts/law_tie_mutants.py` (自身の試験 `scripts/test_law_tie_mutants.py`) と、公開 API を通して `+0` / `-0` の両順・NaN と数の両順・等しい値を bit で確かめる `tests/law_ties.rs` を足した IEEE 上 Law と同じ結果になる書き換え (定数 c > 0 との `x > c ? x : c` など) と、試験の command で build されない site は理由付きで `scripts/law_tie_equivalents.tsv` に列挙し、列挙外の生存・列挙した変異の死亡・消えた列挙・0 件の実行で失敗する AVX2 の分岐の 20 変異は `.github/workflows/law-tie-mutants.yml` (x86_64、対象 file の変更時と週 1 回) で走る
- ALICE-Law G1: CI の clippy job に `cargo clippy --lib --features pyo3 -- -D warnings` の step (`Clippy (pyo3, lib)`) を足し、preflight にも同じ command を入れた 全 native feature の組は pyo3 を含まないので、`src/python.rs` は今までどの clippy の step でも lint されていなかった
- ALICE-Law G2: 浮動小数点の最大・最小・clamp をすべて `alice_ml::law_math` に通した (alice-det-math 0.5.1 の IEEE maximumNumber / minimumNumber、`-0 < +0`、正準 NaN、NaN を保つ clamp、AVX2 は比較と blend の lane 演算) 変更前は `tensor_max(+0, -0)` が x86_64 で `-0`、aarch64 で `+0` で、同じ入力が target で違う bit を返していた `clippy.toml` で `f32` / `f64` の `max` / `min` / `clamp` を禁止 (canary 66 行) 出力の bit を固定する `tests/determinism_golden.rs` (特殊値 24 個の格子、8 本) を足し、7 構成で 1 通りの digest になる AVX2 の lane 試験は CI の x86_64 で AVX2 が無ければ失敗する alice-det-math の要求を 0.5.1 に上げた
- CI (`clippy.toml`、G10): `std::time::Instant`/`SystemTime` の `now`/`elapsed`、`std::collections::hash_map::RandomState` (`::new` を disallowed-methods、型そのものを disallowed-types で禁止、`::default()` も含めて検査する) を追加 以前はどちらも未検査だった `examples/elyza_ternary.rs` / `examples/qwen_qat_test.rs` の進捗表示用の計測 (9 箇所: elyza_ternary 2、qwen_qat_test 7) は scoped `#[allow(clippy::disallowed_methods, reason = "...")]` で許可、production (`src/`) に該当箇所は無し `scripts/gen_clippy_canary_calls.py` で `scripts/clippy_canary_calls.tsv` を再生成 (60 planted、0 skipped、clippy が全件検出することを確認済み) `alice-law-conformance.toml` の G10 を fail-known から pass にした (`src/` に足した `Instant::now()` / `SystemTime::now()` / `Instant::elapsed()` / `RandomState` がそれぞれ clippy で失敗することを確認)
- CI (ALICE-Law G14): 浮動小数点の結果を変える build の flag を静的と動的の 2 段で検査し、round-to-nearest-even と subnormal が flush されないことを実行時に確かめる (ALICE-DB と同一の `scripts/build_config_guard.py` / `scripts/build_flags_check.py` / `tests/float_environment.rs` と test、`scripts/run_tests.sh` / `scripts/test_run_tests.py`) 新しい float-environment job (ubuntu / macOS / Windows) で test を debug と release で `scripts/run_tests.sh float_environment=2` を通して実行し (2 本とも通らなければ失敗)、静的な検査を走らせ、ubuntu / macOS では新しい target dir で cargo を `-v` で走らせて rustc に実際に渡った flag を読む (default / full native feature set / no_std rlib) 意図的な出現は `scripts/build_config_allowlist.toml` に理由付きで書く `alice-law-conformance.toml` の G14 を fail-known から pass にした

### Changed
- `alice-law-conformance.toml` の G4 / G5 の note に、x86_64 と aarch64 の両方の leg を残す理由を書いた (NaN の入力に 0/0 を返す clamp の変異は aarch64 では 0x7fc0_0000 で通り、x86_64 では 0xffc0_0000 で落ちる)
- `law_math::clamp` / `clamp64` は NaN の入力に canonical NaN を `f32::from_bits(0x7fc0_0000)` / `f64::from_bits(0x7ff8_0000_0000_0000)` と明示して返す (`std` は `f32::NAN` の bit を保証しないため) 出力は変わらない (今の build の `f32::NAN` も同じ bit) `tests/float_edge_golden.rs` に符号の違う NaN と payload 付きの signalling NaN の 4 件を足して bit を pin した (40 件)
- **破壊的変更**: `SafetensorsFile::tensors` は全ての build で `BTreeMap` になり (std では `HashMap` だった)、`SafetensorsFile::tensor_names` は名前の順で返す `LlamaTernary::from_safetensors_with_scales` の `scales` は `&BTreeMap<String, (f32, f32)>` を取る
- **破壊的変更**: `ModelArchive::serialize` は ATML version 2 (識別子付き) を書く 0.3 以前の reader は version 2 を読めない `ModelArchive::deserialize` を削除した (読み込みは `ModelArchive::read(data, LegacyPolicy)` で、`Provenance` か `ArchiveError` を返す) `LayerStreamer::from_bytes` は `LegacyPolicy` を取り、`Result<(LayerStreamer, Provenance), ArchiveError>` を返す (以前は `Option`、識別子の不一致も `None` だった)
- 最大・最小の畳み込みと ReLU を `law_math::max_of` / `min_of` / `relu` に、argmax を `law_math::argmax` (最初の最大値、`+0` は前の `-0` に勝ち、NaN は選ばない) に、top-k の並べ替えを `law_math::order` (NaN は全ての数の下、`-0 < +0`) の安定 sort にまとめた clippy が見ない手書きの比較が 6 箇所あり (`moe` の softmax の最大値と top-k の `partial_cmp`、`speculative` の argmax、`calibration` の argmax・softmax の最大値・最大較正誤差)、Law に通した `speculative::SpeculativeDecoder::acceptance_mask` は `[-0, +0]` と NaN を含む logits で、`MoeRouter::route` は NaN の score と等しい score で、`ReliabilityBins::observe_distribution` は `[-0, +0]` で結果が変わる (Law の規則に従う)
- CI: clippy の canary の表の生成器 `scripts/gen_clippy_canary_calls.py` が generic 関数 (`std::thread::spawn` / `scope` / `Builder::spawn` / `spawn_scoped`) を turbofish 付きで植えるようにした (素の path では型が決まらず compile できないため) 他の行は byte 一致で、この repo の表は変わらない 共有 file の協調更新 (ALICE-Analytics などと同じ中身)
- `db` feature: `alice-db` の要求を `0.4` から `0.5` に変更 (`fs` feature は従来どおり有効) `src/db_bridge.rs` の変更は無い rkyv 0.7 (RUSTSEC-2026-0235) には alice-db 経由でのみ到達し、その ignore の前提を `scripts/no_direct_rkyv.py` が CI で検査する
- `deny.toml`: pyo3 0.29 で解消済の RUSTSEC-2026-0176 / RUSTSEC-2026-0177 の ignore を削除
- CI: 公開 API の互換性の検査を `.github/workflows/semver-checks.yml` (`scripts/semver_gate.py`、ALICE-Law / ALICE-DB ほかと同一) に置き換えた crates.io の最新の alice-ml を baseline にし、既定・既定なし・全 feature の 3 組で cargo semver-checks を走らせ、互換でない変更を失敗させる 公開済みは 0.2.0 なので 0.3.0 は major の段で比較しない 0.3.0 の公開後は 0.3.1 として検査する 比べるのは CI の host の architecture (x86_64) の item だけで、`neon` (aarch64) の item は見ない 旧 job (security-audit.yml) は直前の commit (`HEAD~1`) と比べ、結果を問わない informational だった
- CI: clippy の禁止一覧の canary (`scripts/clippy_canary.py`) を追加した 植える呼び出しの表 `scripts/clippy_canary_calls.tsv` は `scripts/gen_clippy_canary_calls.py` が `clippy.toml` から 1 entry 1 行で生成し、CI が再生成して byte で比べる この repo は 54 件 (`disallowed-methods` の f32 / f64 の method、すべて植える) `scripts/clippy_canary.py` / `scripts/test_clippy_canary.py` / `scripts/gen_clippy_canary_calls.py` / `scripts/test_gen_clippy_canary_calls.py` は ALICE-DB と同じ中身
- 公開する文面の検査の指摘を直した (tree の指摘は 0 件) 式と公開 API は変えていない
- **Breaking:** `CacheBudget` の L2 の 2 つの preset の名前を CPU 名の `CacheBudget::l2_cortex_a76` / `CacheBudget::l2_cortex_a76_dual` にし、`name` も `"L2-CortexA76"` / `"L2x2-CortexA76"` にした (値の 512 KB / 1 MB は同じ) 旧名の別名は置かない
- CI: 公開する文面の検査の pin を ALICE-Law 779fa44 (checker sha256 e439fdd7…) に進めた `.public-text.toml` で CHANGELOG.md の 0.2.x の 1 行 (公開済みの API 名) を件数 1 で許し、doc comment と CHANGELOG の対象 CPU の書き方を Cortex-A76 に揃えた

### Added
- CI: `no-direct-rkyv` guard (この crate が rkyv に直接依存せず archive もしないことを検査)
- **`alice-law-conformance.toml` を追加**: ALICE-Law の 14 gate (G1 から G14) それぞれについて、この repository を実測した状態を宣言する manifest 現時点で `pass` は 0 件 (`fail-known` 10 / `unverified` 4)、`pass` に届かない理由は G1 が pyo3 feature を CI の clippy で lint していないこと (src/python.rs に追加した禁止 method は既存の CI 全 job で通過し、`--features pyo3` の clippy のみが検出) と、canary / release leg / 固定 bit pattern の golden / semantics identifier が未整備なこと
- CI: 公開する文面の検査 (`.github/workflows/public-text.yml`、`scripts/public_text.sh`、`.github/public-text-pin`、`scripts/test_public_text.py`、ALICE-Edge と同じ中身) を追加した ALICE-Law の `scripts/public_text_check.py` を pin した revision と SHA-256 で取り、HEAD の tree と base 以降の commit を読み、何も比べなければ失敗する

### Fixed
- **`rust-version` を `1.98.1` から `1.87` に訂正**: 0.3.0 は開発用の toolchain pin の値をそのまま MSRV として宣言していたが、実測の下限は 1.87 (自 crate の `Vec::len` / `is_empty` の const 利用が 1.87 で安定化、`alice-db` / `alice-zip` の `rust-version` も 1.87、`alice-det-math` は 1.85) 1.92.0 / 1.90 / 1.88 / 1.87 で `cargo check --all-targets --all-features` が通り、1.86 / 1.85 は const fn の不安定機能で落ちる `rust-version` が過大だったため 1.98.1 未満の toolchain を使う下流 (path 依存の crate など) が `requires rustc 1.98.1` で build できなかった
- **CI に `msrv` job を追加し、宣言した MSRV を実際に build して確かめる**: `rust-toolchain.toml` の pin を `RUSTUP_TOOLCHAIN` で上書きして `rust-version` と同じ版で default / native feature 全部 / `no_std` rlib を build する `scripts/check_msrv.py` (+ `scripts/test_check_msrv.py`) が `Cargo.toml` の `rust-version` と job の `toolchain` / `RUSTUP_TOOLCHAIN` が同じ値か、`--rustc` で実際の rustc の版が一致するかを照合する 比較 0 件は fail `rust-version` だけを上げる変異で red を確認 `rust-toolchain.toml` に「MSRV ではない開発用 pin」と明記

## [0.3.0] - 2026-10-14

### Changed
- **`alice-det-math` `"0.4"` → `"0.5"`、`alice-db` `"0.2.0-beta.2"` → `"0.4"` (`features = ["fs"]` を追加、`AliceDB::open` は `fs` feature 配下で `default-features = false` のままでは未解決だった)**
- **超越関数 (production code、test を除く) を `alice_det_math` 経由に統一**: `training.rs` (softmax の `exp`/`ln`、Adam の bias correction `powf`) / `llama3_ternary.rs` (RoPE の `powf`/`sin_cos`、SiLU の `exp`) / `ffi.rs` / `tensor.rs` (softmax・GELU・sigmoid の `exp`) / `python.rs` (softmax の `exp`) / `quantize.rs` (entropy の `log2`、SNR の `log10`、二乗誤差の `powi`) / `error_analysis.rs` (geometric mean の `powf`) 実測 19 箇所 (非 test、`sqrt` 除く、comment/string を除いて計測) `clippy.toml` の `disallowed-methods` で `f32`/`f64` の `exp`/`ln`/`powf`/`powi`/`tanh`/`sin`/`cos`/`sin_cos`/`log2`/`log10` 等への新規の直呼びを禁止 (`--lib` 限定、test/example/bench は std のまま許容、理由付き `#[allow]`) `powi` は他の ALICE-* crate の一部と異なりこの禁止対象に含めた — `alice_det_math::powi` 自身の doc comment が「std の `powi` は compiler-rt routine で multiplication order が platform 依存」と明記しているため
- **`src/math.rs` の `no_std` `FloatExt` shim から `exp`/`ln`/`log2`/`log10`/`powf`/`powi` を削除**: 上記の routing で `std` / `no_std` を問わず全呼び出しが `alice_det_math` (こちらも `no_std` 対応) を直接呼ぶようになったため、`libm` 経由の代替実装が死んだ (旧 doc comment が明記していた「activation の値は std build と no_std build で完全一致を保証しない」という制約がこの 6 関数については解消された) `sqrt`/`round`/`hypot`/`mul_add` は残す (IEEE 754 の正しい丸め、または `alice_det_math` の対象外)
- **`Cargo.toml` に `rust-version = "1.98.1"`** (`rust-toolchain.toml` の channel と同じ、CI が実際に build・test する toolchain より古い版での検証は無い)
- **README.md / README_JP.md を全面改稿** (590 行 → 約 165 行、社内の README 執筆規約に準拠): slogan prose・自己採点・Use Cases・Roadmap チェックボックス・`path = "../ALICE-Physics"` snippet・英語版内の日本語 benchmark 節を削除 feature 表は `[features]` と 1 対 1 (`<!-- readme-sync: features -->`)、最初の \`\`\`rust block は `src/lib.rs` の crate doctest と同一、FFI 関数数のような非 gate の件数は書かない (旧版は同一 file 内で 51/65/67 の 3 通りが食い違っていた) 「何が決定論的か」を Law の適合 claim としてではなく、実装の事実として誠実に書く (ternary matvec kernel は超越関数を一度も呼ばないので platform を跨いで再現可能、`<!-- claim-test: every_matvec_kernel_equals_the_closed_form_for_all_lengths -->`、kernel の外側は上記の `alice_det_math` routing を経由) `docs/MODULES.md` を新設 (`pub mod` 全件 + 1 行説明 + 必要 feature) `<!-- law-integration:begin/end -->` の空 marker を License 節の前に設置 (ALICE-Law 側の生成器が後で埋める、Law-dependency=false のためここでは手書きの conformance claim を書かない)
- **`scripts/readme_sync.py` + `scripts/test_readme_sync.py` を追加** (ALICE-Physics の雛形を移植、crate 名のみ `alice-ml` に変更): feature 表 / MSRV / 最初の rust block = doctest / `docs/MODULES.md` の `pub mod` 過不足 / 相対 link 実在 / 英日の `##` 節数、を Cargo.toml・src/lib.rs と突合 比較件数 0 は fail CI (`readme-sync` job) + `scripts/preflight.sh` に配線、本物の tree のコピーで feature 1 つ・MSRV を変異させ red を確認してから revert
- **`mul_add` を production code (test を除く) から全廃、`clippy.toml` に `f32::mul_add` / `f64::mul_add` / `f64::hypot` を追加**: README が「`hypot`/`mul_add` は `alice_det_math` が直す drift の原因ではない」と書いていたのは `hypot` について誤りだった (`alice_det_math` 自身の README が `hypot` を bit-exact 対象に挙げている、platform libm 任せで `exp`/`ln` と同じ問題) `f64` の `hypot` 呼び出し (`error_analysis.rs` の残差合成、1 箇所) は `alice_det_math` に 64-bit 版が無いため `alice_det_math::sqrt64(a*a + b*b)` (教科書通りの等価式、`mul_add` 不使用) に書き換え、オーバーフロー保護を失う代わりに決定論を得るという判断をコードコメントに明記 `FloatExt` から `hypot` も削除 (呼び出し元が無くなり dead code) ⚠️ `mul_add` 自体は drift の原因ではない (Rust は IEEE 754 `fusedMultiplyAdd` として規定しており、target に FMA 命令が有るか無いかに関わらず 1 回丸め・platform を跨いで bit 一致する、無い target は正しく丸められた software 実装に fallback する) それでも `training.rs` / `llama3_ternary.rs` (RoPE・attention) / `moe.rs` (`fast_exp`) / `quantize.rs` / `tensor.rs` (`fast_exp`・GELU・RMS norm・layer norm) / `ffi.rs` / `layer.rs` の計 26 箇所を `a * b + c` に書き換え、`clippy.toml` に `f32::mul_add` / `f64::mul_add` を追加した理由は単純: `a * b + c` (2 回丸め) と `a.mul_add(b, c)` (1 回丸め) は互いに丸め方が違うので、どちらか 1 つの書き方に揃えておけば、どの結果も source だけから再現できる (`alice_det_math` 自身の consumer policy も同じ選択、本 crate の `src/calibration.rs` も同じ理由で既に `a * b + c` を選んでいた) `src/math.rs` の `no_std` `FloatExt` shim から `mul_add` を削除 (呼び出し元が無くなったため dead code) README.md / README_JP.md の該当節を訂正し、layer/RMS norm は `sqrt` しか呼ばない (超越関数ではない) ことも合わせて修正
- **`clippy.toml` の `disallowed-methods` を ALICE-Law G1 ("no-libm-direct") gate の banned 一覧 (25 method) と f32/f64 両方で完全一致させる**: 既存の一覧は `f64::powi` / `f64::tanh` / `exp2` / `exp_m1` / `ln_1p` / `sinh` / `cosh` / `cbrt` (一部 width) 等 複数が未掲載のまま README が「routing 済」と主張していた `tests/g1_disallowed_methods_complete.rs` を追加 (G1 banned 一覧のスナップショットを埋め込み、`clippy.toml` が両 width 全部を含むかを突合、比較 0 件は fail) `powi` / `mul_add` は G1 では pending / allowed だが本 crate は引き続き独自に禁止 (既存の理由を維持) README の method 列挙をこの完全な一覧に合わせて書き直し、`error_analysis.rs` の test 2 件が使っていた `f64::powi` を `a * b` の繰り返しに置き換え (test 専用の期待値計算で `alice_det_math` に f64 版が無いため)
  - **`tests/g1_disallowed_methods_complete.rs` を文字列検索から実 TOML parse に書き直し**: コメントアウトした entry が still green を返していた (`clippy.toml` が ban しなくなっても検査が気付かない) を修正 配列の形自体を見る試験も追加
  - **CI job `law-g1-banned-list` + `scripts/check_g1_banned_list.py` を新設**: `.github/law-g1-pin` が指す ALICE-Law の commit から `conformance/gates.toml` を fetch (SHA-256 で内容を確認、一致しなければ fail) し、G1 の *live* な banned 一覧と `clippy.toml` を直接突合 (`tests/g1_disallowed_methods_complete.rs` のスナップショットには依存しない) pin の更新は人の手で行う commit のみが ALICE-Law 側の変化を反映する (ALICE-Law 側だけの変更が本 repo の結果を黙って変えることはない) preflight にも配線

### Known issue (not fixed in this release)
- **`cargo semver-checks check-release` が `am_ml_arena_reset` / `am_ml_tensor_copy` の `const` 喪失を major 相当として検出する**: 2026-09-16 の `catch_unwind` 導入 commit (`2c958fd`) で全 `extern "C"` 関数が runtime の closure 呼出しになり、`const fn` ではいられなくなったことが原因 本 release の変更とは無関係 (det-math/db の bump・routing は ffi.rs のこの 2 関数に触れていない) CI の `semver-checks` job は `continue-on-error: true` (informational) のため、この break は 0.2.0 以降どの release でも bump の根拠として扱われてこなかった 対応は crate 所有者判断 (別途課題管理に記録)

### Changed
- **License: `AGPL-3.0-or-later` → `AGPL-3.0-or-later OR LicenseRef-Commercial` (dual-licensed、2026-09-27)** AGPL 側の条件は変更なし (既存 AGPL 利用者への影響ゼロ)、商用という選択肢が追加されただけ SPDX が AGPL 単独だと cargo-deny / FOSSA / SBOM に「商用オプションなし」と見えるため宣言を dual に 変更点: SPDX / `LICENSE` → `LICENSE-AGPL` rename / `LICENSE-COMMERCIAL.md` (商用トリガー 6 条件 = クローズド製品・商用 SaaS・エッジ・ファームウェア配布・plugin 再配布・プラットフォーム NDA・保証、社内利用は AGPL 側で無償と明記) / README の選択肢表 商用窓口は法人 `contact@extoria.co.jp`

### Added
- **`calibration` module** — ALICE-* で初めての確率校正 これまで ALICE-* 全体に
  Brier / ECE / isotonic の実装は 1 行も無く、`qat::CalibrationStats` は
  量子化 calibration (activation range / weight sensitivity) で別物だった
  - 測る側: `brier_binary` / `brier_multiclass` (proper scoring rule) /
    `negative_log_likelihood` / `ReliabilityBins` (ECE と MCE、reliability diagram)
  - 直す側: `TemperatureScaling` (1 param、黄金分割探索で NLL 最小化、**順位を
    変えない**) / `VectorScaling` (2K param、解析勾配 `(p_k - y_k)` による勾配降下、
    順位を変えうる) / `IsotonicRegression` (pool adjacent violators、binary)
  - `tests/calibration_oracle.rs` 30 本 一様分布の Brier は `1 - 1/k`、NLL は
    `ln k`、完全校正集合の ECE は厳密に 0、PAV は違反対をその平均に畳む —
    いずれも定義から導いた値で、実行結果を写したものではない
    温度の適合は「ラベル頻度が `softmax(z)` と厳密に一致する集合」を作って
    `s` 倍すると最適温度が解析的に `s` になることを使い、許容 0.01 で検証
  - `no_std` (`alloc`) 対応 `f32` 入出力に対し内部は `f64` 累算
- 依存に `alice-det-math` 0.2 を追加 (crates.io 版、path dep ではないので
  sibling checkout が要らない) calibration の `exp` / `ln` を platform libm に
  任せると target ごとに最終 ulp が変わり、同じ予測から違う校正値が出るため
  `mul_add` も使わない (FMA の有無で 1 回丸め / 2 回丸めが変わる)

### Fixed
- **`SafetensorsFile::tensor_to_f32` / `tensor_bytes` panicked on a header that disagrees with the data**: a `shape` larger than the byte range (`shape: [1000]`, `data_offsets: [0, 4]`) indexed past the slice, inverted `data_offsets` sliced `start > end`, and a hostile `shape` overflowed `n_elements` Both return `None` now, `n_elements` saturates, an unsupported dtype returns `None` before touching the bytes (`tests/analytic_oracle.rs` § safetensors_format)
- **`ternary_matvec_simd_dispatch` (safe, crate root export) と arch 別 `ternary_matvec_dispatch` は入力長を検証していなかった** AVX2 / NEON kernel は `input` を raw pointer で読むため、`input.len() < in_features` で out-of-bounds read (UB) `assert_eq!` で長さを確認して panic に変更 (`# Panics` doc、`simd_dispatch_rejects_short_input` test) scalar kernel は slice index で panic していた
- **`no_std` build が一度も通っていなかった** (`--no-default-features` で 65 error: `mul_add` / `sqrt` / `exp` / `vec!` 等) — `src/math.rs` の `FloatExt` trait (`libm` 委譲、`std` 時は不使用) と `alloc::vec` import で修正、`parallel` feature は `std` を要求するよう明示 host rlib + bare-metal `thumbv7em-none-eabihf` で build を確認 (`crate-type` に cdylib / staticlib を含むため `cargo check` では panic_handler / allocator 要求で落ちる、検証は `cargo rustc --crate-type rlib`)
- feature-gated module (`ffi` / `safetensors` / `llama3_ternary`) の clippy pedantic / nursery 38 件と rustdoc 未解決 link 2 件 (CI が `--features simd` のみで未 lint だった)
- `elyza_ternary` / `qwen_qat_test` example に `required-features = ["safetensors"]` (default build で unresolved import)

### Added
- `ffi`: 全 65 `extern "C"` 関数の本体を `catch_unwind` (`guarded`) で囲み、Rust panic を host process の abort ではなく sentinel 戻り値 (null / 0 / 0.0 / -1) にする `am_ml_last_error()` で thread-local の message (NUL 終端) を取得、`am_ml_clear_last_error()` で消去 UE5 / Unity / C から呼ぶ時に一つの bad input で editor が落ちない

- `#![deny(clippy::undocumented_unsafe_blocks)]`: 手書き `unsafe` block 212 個 (ffi 195 / neon 6 / tensor 8 / arena 2 / ops 1) 全てに `// SAFETY:` で不変条件を記載

### Changed
- `alice-db` は crates.io の `0.2.0-beta.2` (path dep 廃止) CI の "dependency stubs" (空 crate、`db` feature を何も compile しない偽 gate) と `alice-stubs` action を撤去、`db` を native feature set と feature-powerset に含めて実体で check
- `ternary_matvec` / `ternary_matvec_kernel` / `ternary_matvec_kernel_quantized` の長さ検証を `debug_assert_eq` → `assert_eq` (release でも SIMD dispatch と同じ `# Panics` 契約、scalar kernel は slice index で panic していたので UB は無かった)
- release profile の `panic = "abort"` を除去 (`catch_unwind` を無効化するため、理由は `Cargo.toml` の comment)
- `libm = "0.2"` を依存に追加 (`no_std` build のみ使用) `no_std` 時の `exp` / `sqrt` 等は platform libm と最終 ulp で異なりうる (ternary matvec の bit-exact 性は不変、`src/math.rs` doc 参照)
- `tests/analytic_oracle.rs`: 閉形式 oracle 32 本 — ternary matvec は packed / bit-parallel / SIMD dispatch / `from_packed_weight` / batch / INT8 入力の全 kernel を `y = γ·(Σ₊x − Σ₋x)` と `in_features = 1..=70` (NEON 4 lane / AVX2 8 lane / 32-bit word の端数 path 全部) で `==` 突合、quantize は `W ∈ {−a, 0, +a}` の自己再現 + γ = mean|W| + 正のスケール同変 + 丸め閾値 γ/2 (温度 τ で τγ/2)、圧縮率の bit 幅法則、softmax (Σ = 1 / shift 不変 / 2 要素閉形式 / fast 版は Schraudolph 6 % 内 + 順位保存)、layer norm (mean 0 / var 1 / affine 不変 / eps 独立)、rms norm (rms 1 / x/rms(x) 一致 / scale 不変) 初回 mutants (64.9 %) 後に 9 本追加: element-wise / reduction op を長さ 1..=40 で整数厳密 (AVX2 8 lane の端数 lane に極値を置く)、Tensor / OwnedTensor / QuantizedTensor accessor、layer / rms norm の weight・bias・eps 経路、softmax の行独立性と全 −inf 行の uniform fallback、`QuantStats` (count / range / mae / sparsity / Shannon entropy) と `compute_quantization_error` (mae / mse / rmse / max / SNR、完全一致で ∞)、sparse 量子化の閾値 = threshold·γ、QAT の stats と温度、bit-plane accessor と packed byte mask helper (全 256 byte)、`ternary_matmul_alloc` の 1D / [1, n] / [b, n] shape 2 回目 (74.4 %) 後に ternary Llama-3 forward 5 本: 4 次元 toy model (`from_parts`) で projection を zero / identity にして logits を閉形式 (zero → rms_norm(embedding)、identity v / o → hidden + [n0, n1, n0, n1]、identity FFN → hidden + silu(n)⊙n、KV cache 2 position と clear_cache で不変)、`memory_bytes` の総和、`save_atml` → `ModelArchive` の 8 layer、7 projection 未満で panic safetensors 3 本: format 定義から合成した file (F32 / F16 / BF16 / 未対応 dtype / nested `__metadata__`) の読み戻し、fp16 の IEEE 754 点 (max finite 65504 / min normal / max subnormal / ±inf / NaN) と bf16 = f32 上位 16 bit、malformed header 10 種が panic せず None training 3 本: MSE / MAE / cross-entropy の値と勾配 (one-hot で −log softmax、勾配和 0、1e-8 床で loss は 18.42 に cap)、SGD momentum の漸化式、Adam の初回 step = −lr·sign(g) と 20 step の漸化式突合 matvec 3 kernel の長さ契約 (catch_unwind 5 経路)
- `quality-deep.yml` (週次 / 手動 / PR 差分): cargo-mutants 16 shard、feature set = native (ffi / simd / parallel / safetensors / db)、test set = lib + `analytic_oracle`、`src/python.rs` / `src/math.rs` (no_std 専用 shim) / `src/neon.rs` (aarch64 専用) は除外 (`.cargo/mutants.toml`、理由は file 内)
- CI `neon` job (macos-latest = aarch64): NEON kernel を同じ oracle で clippy + test (今まで NEON path は CI で一度も走っていなかった)
- CI: clippy を `--all-targets` + full native feature set (`ffi,simd,parallel,safetensors`) で `-D warnings`、`no_std` job (host + thumbv7em + clippy-driver wrapper)、`feature-powerset` job (cargo-hack、std 固定 depth 2)、test / doc も full native feature set を追加、rust-cache

## [0.2.0] - 2026-03-06

### Added
- `micro_model` — L2 Cache-Resident Micro Model (`MicroModel`, `MicroModelBuilder`, `CacheBudget`)
  - Cortex-A76 L2 (512KB) に全重みを常駐、~100 GB/s で推論
  - `CacheBudget` プリセット (l2_rpi5, l2_rpi5_dual, l2_256k, custom)
  - `build_random()` で決定的テストビルド (LCG64)
- `speculative` — `CacheResidentDecoder` (L2 ドラフト + DRAM 検証)
- `speculative` — `SpeculativeDecoder` (ドラフトモデル先読み + バッチ検証)
- `streaming` — `LayerStreamer` (オンデマンドレイヤーロード + LRU エビクション)
- FFI: 14 新関数 (`am_ml_micro_model_*` 8個, `am_ml_cache_decoder_*` 6個) → 合計 65 関数
- Unity C#: 14 新 DllImport + 2 RAII handles (`MicroModelHandle`, `CacheDecoderHandle`) → 合計 65 DllImport + 9 handles
- UE5 C++: 14 新 extern C + 2 RAII handles (`MicroModelPtr`, `CacheDecoderPtr`) → 合計 65 extern C + 9 handles
- 235 tests (205 core + 24 FFI + 6 doc-tests)

### Fixed
- `cargo clippy --all-features -- -W clippy::pedantic -W clippy::nursery` → 0 warnings (53 warnings 修正)
- `cargo fmt` → 0 diff
- `cargo doc --no-deps` → 0 warnings

## [0.1.1] - 2026-03-04

### Added
- `ffi` — C-ABI FFI 51 `extern "C"` functions (Arena/Weight/Kernel/Matvec/Tensor/BitLinear/Quantize/Version)
- Unity C# bindings — 51 DllImport + 7 RAII IDisposable handles (`bindings/unity/AliceMl.cs`)
- UE5 C++ bindings — 51 extern C + 7 RAII unique_ptr handles (`bindings/ue5/AliceMl.h`)
- FFI prefix: `am_ml_*`
- 115 tests (93 core + 20 FFI + 2 doc-tests)

### Fixed
- `cargo fmt` trailing spaces修正
- `tensor.rs` unused_mut修正

## [0.1.0] - 2026-02-23

### Added
- `arena` — Bump allocator for zero-allocation inference
- `tensor` — `Tensor` (borrowed), `OwnedTensor`, `QuantizedTensor` (INT8); DPS operations (relu, softmax, layer_norm, gelu, silu, rms_norm, etc.)
- `ops` — `TernaryWeight` (packed 2-bit), `TernaryWeightKernel` (bit-parallel SIMD); DPS kernels (matvec, matmul_batch, quantized variants)
- `layer` — `BitLinear` layer abstraction
- `quantize` — INT8 quantization/dequantization with absmax scaling
- `error_analysis` — Quantization error analysis and error budget tracking
- `neon` — ARM NEON SIMD kernels (feature-gated: `neon`)
- `python` — PyO3 + NumPy zero-copy bindings (feature-gated: `pyo3`)
- `db_bridge` — ALICE-DB training metrics persistence (feature-gated: `db`)
- Feature flags: `std`, `simd`, `neon`, `parallel`, `pyo3`, `db`
- `no_std` + `alloc` support
- 93 unit tests (100 with all features) + 2 doc-tests
- CI/CD (GitHub Actions: test, clippy pedantic, fmt, doc)
- `#[must_use]` on all public query functions
- `# Errors` doc sections on all `Result`-returning functions
