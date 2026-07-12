# Code Quality & Engineering Assessment — `mlp` workspace

**Scope:** the Rust workspace (`network`, `data`, `cli`, `io`, `archivist`, `display/src-tauri`), the React/Tauri frontend, and the teaching document in `tex/`.

**Context taken into account:** this is a DIY workshop project in active development — an MLP built from scratch, deliberately without linear-algebra libraries, following a self-written teaching document. It is judged as such: work-in-progress scaffolding, parked code, and placeholder values are treated as what they are (the README's todo list already names most of them), and the assessment focuses on what actually matters — *does the engineering reasoning hold up, and what does it reveal about the author's level?*

---

## 1. Architecture at a glance

| Crate              | Role                                                                    |
|--------------------|-------------------------------------------------------------------------|
| `network`          | The MLP core: propagator, flattened parameters, rayon-parallel backprop |
| `data`             | Owned datasets, batch providers, evaluation/export                      |
| `cli`              | Batch training driver                                                   |
| `io` / `archivist` | Persistence layer (planned)                                             |
| `display`          | Tauri + React training playground                                       |

The crate split is sound, and it is the quiet best decision in the repo: the `BatchProvider` / `Evaluable` traits in `network/src/lib.rs` are exactly the right seams — `network` knows nothing about CSV files or Tauri, `data` knows nothing about threads. That separation is what made the Tauri frontend possible without touching the math core, and it was clearly deliberate rather than accidental.

---

## 2. The teaching document (`tex/`)

### 2.1 `mlp.tex` — the strongest artifact in the repo

This is genuinely good teaching material, and it is *better than most tutorials* in one specific way: it refuses to stop at matrix notation. The backpropagation derivation is done at the **index level with Kronecker deltas** (eq. `chainrule-error-signal` onward), which is exactly the level of rigor needed to write the flattened-array implementation — and the document says so explicitly: *"Hardware does not know about matrices, only about linear addresses and offsets."* The bias-as-ghost-neuron trick, the transition-offset formula, and the "From Coordinates to Contiguous Memory" section map 1:1 onto `net_properties.rs` and `net_propagator.rs`. Doc and code genuinely co-evolved; this is a doc-first project and it shows.

Two substantive gaps, both cheap to close:

1. **The factor 2 of the squared loss is silently dropped.** The loss is defined as $L = \sum e^2$ (eq. `eq:loss`), so $\partial L/\partial a = 2e$, but eq. `eq:first-signal` states $\delta^L = e \cdot g$ and the code follows (`delta_l = z - y`). Absorbing the 2 into the learning rate is standard practice — but a document whose selling point is index-level rigor should say so in one sentence. This is the kind of gap that costs a workshop participant an afternoon.
2. **The classifier path is undocumented.** The code has a softmax output and a cross-entropy-shaped first error signal; the document derives only the regression/MSE case. The fact that softmax+CE yields the *same* first error signal $z - y$ as identity+MSE is a beautiful result and exactly the kind of thing this document exists to explain — it deserves its own subsection when the classifier work settles.

### 2.2 `gd.tex` — ambitious and mostly sound

Deriving Adam ("Adamprop") from Newton–Raphson → Fisher information → diagonal approximation → low-pass-filtered moment estimates is an unusual and genuinely illuminating route — most texts present Adam as a bag of tricks; this derivation gives every term a *reason*. The "noisy integrator" and "cold start / transient response" framing gives away a signal-processing/controls background, and it is a productive lens: bias correction falls out of the geometric-series algebra instead of being asserted. The "dimensional sanity" argument for $\sqrt{\hat v}$ over $\hat v$ and the reading of $\epsilon$ as a damping floor rather than a numerical hack are both the marks of someone who *thinks in systems*, not recipes.

One mathematical note worth a paragraph in a future pass: the FIM step is the hand-waviest link in the chain — the score-function expectation identity requires the expectation to be taken *under the model distribution* (plus regularity conditions for swapping $\nabla^2$ and $\int$), while in practice the expectation is over the *empirical* distribution. That distinction is exactly why Adam's "variance" is only loosely the Fisher information, and acknowledging it would make the strongest chapter stronger. (Cosmetic: the heading *"Numerical Tweaks for Sanitization"* appears twice in a row.)

### 2.3 `implementation.tex`

Parked (commented out of `main.tex`) while the code moves — appropriate for a doc-first project mid-refactor. One observation that counts *in the author's favor*: the chapter's carefully argued case for a `training: bool` flag over `Option` was made obsolete by the author's own later typestate refactor, which beats both alternatives. That's the doc capturing a genuine progression in Rust maturity; when the chapter is revived, that section can be rewritten as "why typestate wins" and it will be better teaching material for having gone through all three stages.

Notably, this chapter's performance reasoning is **correctly calibrated** — see §5.

---

## 3. Code quality — what is genuinely good

These are things that are *right for real reasons*, not checkbox best practices:

- **The flattened-everything memory model.** One contiguous `Arc<[f32]>` for parameters, one `Signals` arena for activations, a two-buffer `SwapBuffer` for layer transitions. Zero heap traffic inside the forward/backward hot loops. This is the correct shape for the problem, and it was *derived* (in the tex) rather than cargo-culted.
- **The snapshot concurrency pattern** (`RwLock<Arc<[f32]>>`): read-lock once per batch, clone the `Arc`, fan out to rayon workers against an immutable snapshot, write-lock once to swap in the new pointer. This is the textbook-correct pattern for coarse-grained epoch parallelism, and the document's justification of it (no double-checked locking needed because updates are sequential by construction) is accurate.
- **Per-chunk propagator reuse in `backprop`** (`neural_net.rs:131-158`): manual `par_chunks` so each rayon worker allocates *one* propagator and keeps its buffers cache-hot across its whole chunk, instead of per-sample allocation under a naive `fold`. This is the kind of decision that separates people who have profiled parallel code from people who have read about it.
- **`enable_fast_math` (`network/src/lib.rs`)**: FTZ/DAZ via inline asm on both x86_64 (MXCSR) and aarch64 (FPCR), with a correct explanation of *why* (denormal stalls as weights/gradients approach zero — a failure mode most people learn about the hard way). The comment documenting the author's own confusion about `stmxcsr`/`ldmxcsr` direction is exactly what a workshop artifact should contain.
- **Numerically-aware details:** He initialization (`sqrt(2/n_in)` — correct for ReLU), max-subtraction in softmax, derivative-in-terms-of-activation convention (sigmoid `a(1-a)`) documented at the field definition (`activations.rs:7`).
- **Typestate `Batch<Training>/Batch<Inference>`**: a right-sized use — it removed a runtime flag and made `output_sample` (training-only) and `set_output` (inference-only) unrepresentable to misuse, at zero runtime cost. Not showing off; earning its keep.
- **Minimal visibility discipline**: `data_structures` internals are private, the crate root re-exports exactly the public surface (`lib.rs:3-5`). `NetPropagator`, `Signals`, `SwapBuffer` are implementation details and stay that way.
- **Insta snapshot tests** are a pragmatic fit for a from-scratch numeric project while it's moving: they pin behavior against refactors cheaply. The gap they leave is covered in §4.3.

## 4. Substantive findings

Filtered for the WIP context: only things that affect the math the project is *currently* relying on, or that will shape upcoming design decisions.

### 4.1 Loss accounting disagrees with itself

`neural_net.rs:163`: `self.update_loss(total_loss / (num_params as f32))` — the summed squared error over the batch is divided by the **number of network parameters** rather than the sample count, and because `update_loss` overwrites per batch, the value read at epoch end (`neural_net.rs:121`) reflects only the *last* batch. The number the Tauri UI plots therefore trends correctly (which is why it looks plausible) but its magnitude is arbitrary and it gets noisier when the final batch is short. Worth fixing early because two roadmap items — convergence-based termination (the closing step of `gd.tex`'s own algorithm) and Adamprop tuning — will both consume this number.

### 4.2 Gradient and reported metric are a mismatched pair (known)

`net_propagator.rs:46-55`: after a softmax output, `delta_l = z - y` *is* the exact softmax+cross-entropy gradient — the training direction is correct. The reported loss, however, is `Σ(z-y)²` (Brier score), and the inline comment says so (`// todo: ce loss later`). Flagged here only because it compounds with §4.1: anyone debugging classifier convergence against the emitted loss is watching a number the optimizer is not optimizing. The acknowledged todo is the right instinct; the improvements doc suggests making the (output activation ↔ loss ↔ first delta) triple one typed unit so it can't drift apart again.

### 4.3 The one missing test is the one snapshots can't provide

The snapshot suite pins *whatever the code did when the snapshot was accepted* — it cannot detect that a pinned gradient is mathematically wrong, and it will demand re-blessing on every legitimate numeric change. A **finite-difference gradient check** (`(L(θ+ε) − L(θ−ε))/2ε ≈ ∇L` on a tiny fixed-seed net) is ~30 lines, catches sign/index/off-by-one errors in backprop with near-certainty, and is the standard oracle of the trade. With Adamprop and loss changes on the roadmap, this is the highest-leverage test the project can add, and it's worth adding *before* those changes.

### 4.4 The `emission` feature forks the public API

`train()` has two incompatible signatures depending on the `emission` feature. Feature flags are additive by design; using one to change a signature means dependents get whichever API the workspace's feature-unification happens to produce — `cargo build -p cli` and `cargo build --workspace` currently see different crates. This is a small thing today and a compounding thing later (any new consumer — a bench harness, a WASM target — re-rolls the dice). A single `train` taking `impl FnMut(usize, f32)` costs nothing after monomorphization; the improvements doc has the details.

---

## 5. Real bottlenecks vs. imagined ones — a calibration check

This project explicitly asks to be judged on performance *reasoning*, so: is the author worrying about the right things?

**Correctly *not* worried about (and says so, in `implementation.tex`):**

- `Arc` pointer indirection on `NetProperties` — correct; it's cold data read through slices, dwarfed by the MAC loops. The doc's line *"If a single pointer dereference is slowing you down, the computational intensity of your linear algebra is likely the least of your problems"* is exactly right.
- The `RwLock` — locked twice per *batch*, contended never. Correct to leave alone.
- Avoiding lifetime-parameterized views in favor of `Arc` snapshots — correct tradeoff at this scale. Borrowed views into the dataset would save one memcpy per batch and cost an infectious `'a` on every type in the training stack. The copy is not the bottleneck; the ergonomics debt would have been permanent.
- `fn(f32) -> f32` pointers for activations instead of generics/closures — the indirect call is real but sits next to a dot product over an entire layer; fine, and it keeps `NetPropagator` non-generic over the activation.

**The actual hot-path costs — mostly already on the README's own todo list, which counts in the author's favor:**

1. **Per-batch allocation traffic**: every `backprop` call allocates a `Vec<usize>` index list, one `vec![0.0; num_params]` per rayon chunk, another per `reduce` identity, a fresh `Vec` for `delta_l` **per sample** (`net_propagator.rs:46-50` — the only allocation inside the innermost loop), and a new `Arc<[f32]>` per parameter update. At 10 000 epochs × batches this is the dominant allocator churn. None of it needs lifetimes to fix — chunk-local scratch buffers reused across batches suffice.
2. **Bounds checks in the inner dot products**: `self.parameters[start_idx + j]` indexes element-wise (`net_propagator.rs:80-81, 115`); rewriting over subslices/`zip` would let LLVM elide the checks and autovectorize. This is the single cheapest real speedup available.
3. **Batch data copies** — every batch `Vec::from`s its data (`batch.rs:31`), random batching gathers into a fresh `Vec` per draw. Acknowledged in the README ("use data views instead of owned data").

**The `const N` verdict** (this is the "proc-macro skill-signaling" test case): `NeuralNet<const N: usize>` puts `[usize; N]` — cold configuration data — on the stack, while everything hot (`parameters`, `Signals`, buffers) stays heap-allocated and is accessed through `&[usize]` slices anyway (`layout()` returns `&[usize]`, erasing `N` at every use site). Runtime gain: ~zero. Cost: the type parameter propagates through `NetProperties`, `NetPropagator`, and the Tauri `static NEURAL: OnceLock<NeuralNet<5>>`, and — decisively — it **structurally blocks the README's own todo "Make the net and training configurable"**, because a runtime-chosen layout cannot produce a compile-time `N` without dynamic-dispatch gymnastics. The commit history shows it was a deliberate learning exercise, which is legitimate workshop behavior — but the honest engineering assessment is: this is the one place the project optimized for *exercising* a feature over needing it. Knowing when *not* to deploy a fancy feature is the senior skill; the typestate `Batch` passes that test, `const N` fails it — and unlike the todo-list items, it gets more expensive to unwind with every new call site.

---

## 6. The frontend's Rust side

The event architecture in `display/src-tauri` is the right shape: training runs on its own thread, emits epoch/loss events on a decimated schedule (every 10 epochs), and inference-on-demand is a separate command — that decoupling is precisely what makes a live dashboard possible without the training loop ever waiting on the UI. The pieces still to grow (session lifecycle, re-initialization, passing a config instead of the in-file layout constants) are natural next steps once the config work from the improvements doc lands, and the current design doesn't have to be torn up to get there.

---

## 7. Level of the engineer

**Rust skill: solid intermediate-plus, on a steep and visible upward trajectory.**
Evidence: correct and *documented* reasoning about `Arc`/`RwLock` snapshot semantics; a genuinely appropriate typestate refactor (the git history shows booleans → typestate as a deliberate upgrade, and the old tex chapter preserves the before-state); comfortable with workspaces, rayon, feature flags, inline asm, serde, and Tauri glue; real module/visibility discipline. The remaining growth edge is API design under change — the signature-forking feature flag (§4.4) and `const N` (§5) are both cases where a mechanism was chosen before the future call sites were imagined. That's the specific skill that separates intermediate-plus from senior, and this project is exactly the kind of vehicle that builds it.

**Domain skill (ML/numerics): strong classical foundations.**
Derives backprop at index level and Adam from Newton–Raphson/FIM — that is *understanding*, not tutorial-following. Knows about denormal stalls, He initialization, softmax stability, derivative-via-activation. The signal-processing lens (low-pass filters, transient response, SNR, damping) is a real asset that most ML-first practitioners lack. The open items are bookkeeping rather than concept: the loss-normalization slip (§4.1) is the kind of thing a validation-curve habit surfaces immediately, and the missing gradient oracle (§4.3) is the one piece of standard numeric-code practice not yet in the toolbox.

**Composite:** this reads as a **strong systems/controls engineer self-teaching ML internals and modern Rust simultaneously — and doing it the right way**: doc-first, deriving before implementing, profiling intuitions calibrated against hardware reality (the `Arc` paragraph in the tex could be framed on a wall), and a todo list that accurately names the project's own open edges. Judged as work-in-progress, the substantive debts reduce to three: `const N` (a design decision that resists the project's own roadmap and should be unwound early), the loss accounting (ten lines, but it gates everything that consumes the metric), and the absent gradient check (the cheapest insurance available before the optimizer work begins). Everything else is the ordinary and healthy surface of a project that is being *built in the open*.
