# Changelog

All notable changes to ALICE-ML will be documented in this file.

## [Unreleased]

### Changed
- `db` feature: `alice-db` の要求を `0.4` から `0.5` に変更 (`fs` feature は従来どおり有効) `src/db_bridge.rs` の変更は無い rkyv 0.7 (RUSTSEC-2026-0235) には alice-db 経由でのみ到達し、その ignore の前提を `scripts/no_direct_rkyv.py` が CI で検査する
- `deny.toml`: pyo3 0.29 で解消済の RUSTSEC-2026-0176 / RUSTSEC-2026-0177 の ignore を削除
- CI: 公開 API の互換性の検査を `.github/workflows/semver-checks.yml` (`scripts/semver_gate.py`、ALICE-Law / ALICE-DB ほかと同一) に置き換えた crates.io の最新の alice-ml を baseline にし、既定・既定なし・全 feature の 3 組で cargo semver-checks を走らせ、互換でない変更を失敗させる 公開済みは 0.2.0 なので 0.3.0 は major の段で比較しない 0.3.0 の公開後は 0.3.1 として検査する 比べるのは CI の host の architecture (x86_64) の item だけで、`neon` (aarch64) の item は見ない 旧 job (security-audit.yml) は直前の commit (`HEAD~1`) と比べ、結果を問わない informational だった
- CI: clippy の禁止一覧の canary (`scripts/clippy_canary.py`) を追加した 植える呼び出しの表 `scripts/clippy_canary_calls.tsv` は `scripts/gen_clippy_canary_calls.py` が `clippy.toml` から 1 entry 1 行で生成し、CI が再生成して byte で比べる この repo は 54 件 (`disallowed-methods` の f32 / f64 の method、すべて植える) `scripts/clippy_canary.py` / `scripts/test_clippy_canary.py` / `scripts/gen_clippy_canary_calls.py` / `scripts/test_gen_clippy_canary_calls.py` は ALICE-DB と同じ中身
- 公開する文面の検査の指摘を直した (tree の指摘は 0 件) 式と公開 API は変えていない

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
  - RasPi 5 L2 (512KB) に全重みを常駐、~100 GB/s で推論
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
