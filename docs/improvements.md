# Real Improvements — structural decisions that will hinder future design

Companion to `code-quality-assessment.md`. Scope is deliberately narrow:

- **No style, naming, or cleanup items** — WIP surface is WIP surface.
- **Nothing already on the roadmap.** The README todos and `gd.tex` already name the planned work (Adamprop, data normalization, batch views, configurability, own training data); those need no restating. What's listed here is the opposite category: **current design decisions that will actively resist that roadmap** — the things that get more expensive with every commit that builds on them.

Ordered by how expensive each becomes the longer it stays.

---

## 1. Remove `const N` from the network core

**What's wrong:** `NeuralNet<const N: usize>` / `NetProperties<const N: usize>` bake the layer *count* into the type. The layout array `[usize; N]` is cold configuration data; every hot structure (parameters, signals, buffers) is heap-allocated regardless, and every consumer reads the layout through `&[usize]` anyway (`NetProperties::layout()` erases `N` on the spot).

**Why it hinders:** the roadmap says *"Make the net and training configurable"* — a user-chosen layout is a runtime value, and a runtime value cannot become a `const N` without a match-arm ladder or `dyn` gymnastics. It has already forced `static NEURAL: OnceLock<NeuralNet<5>>` in the Tauri crate. Every configurability feature (layout picker in the UI, model loading via the archivist) hits this wall first.

**Fix:** revert `layout` to `Vec<usize>`/`Box<[usize]>` inside `NetProperties` and delete the type parameter everywhere. Nothing measurable is lost. Half-day change today; it grows with every new call site.

## 2. Configuration has no single source of truth

**What's wrong:** `[2, 24, 24, 24, 2]`, batch size, epoch count, and `is_classifier` exist as independent literals in `cli/src/lib.rs` and *three separate times* in `display/src-tauri/src/lib.rs` (`get_net`, `train`, `calculate`). Nothing enforces that they agree — the net's identity is smeared across call sites.

**Why it hinders:** three roadmap items (*configurable net, own training data, striding selection*) all reduce to "pass a config in," and each will otherwise become another literal-hunt. It also blocks the persistence layer before it starts: a model can't be saved/reloaded without its topology traveling with the weights.

**Fix:** one serde-derived `NetConfig { layout, activation, is_classifier }` + `TrainConfig { batch_size, epochs, batch_mode, … }` in `network`. The Tauri `train` command takes it as its argument (Tauri serializes it for free), the CLI parses it, the archivist stores it beside the parameters. One struct, three roadmap items unlocked. (Requires §1 — a runtime layout can't feed `const N`.)

## 3. The planned optimizer has nowhere to live

The GD→Adamprop upgrade itself is roadmap — this item is only about the *structure it needs*. Adamprop carries **per-parameter persistent state** (`m`, `v`, step counter `k` for bias correction), and today parameter updating is a private method on `NeuralNet` (`neural_net.rs:52-61`). Implementing Adamprop in place forces that state *into* the net, permanently entangling "what the network is" with "how it is currently being trained" — and learning-rate schedules and gradient clipping will pile into the same spot later.

**Fix, before the Adamprop work starts:**

```rust
pub trait Optimizer {
    fn step(&mut self, params: &mut [f32], gradient: &[f32]);
}
// Sgd { lr }   ·   Adamprop { lr, alpha, beta, m: Vec<f32>, v: Vec<f32>, k: u64 }
```

owned by `NeuralNet` or, cleaner, by a `Trainer` that borrows the net. One evening now; a multi-day untangling if done after Adamprop lands.

## 4. The `emission` feature forks the public API

**What's wrong:** `#[cfg(feature = "emission")]` swaps between a 2-arg and a 3-arg `train()`. Feature flags are additive by design; using one to change a *signature* means the crate has two incompatible APIs, and every dependent silently gets whichever one the workspace's feature-unification produces — `cargo build -p <crate>` and `cargo build --workspace` can already disagree today.

**Why it hinders:** every future consumer of `network` (a bench harness, a WASM target, a second frontend) re-rolls this dice, and the fork will deepen as training gains more observable state (per-epoch metrics, early-stopping signals).

**Fix:** one `train` that always takes `impl FnMut(usize, f32)` — a no-op closure costs nothing after monomorphization — or an observer trait with a default no-op impl. Delete the feature.

## 5. Loss has no accumulation design

**What's wrong:** loss is a single overwritten cell — `update_loss` stores each batch's value over the last (`neural_net.rs:63-66`), and the stored value is normalized by **parameter count** rather than sample count (`neural_net.rs:163`). What the epoch-end read reports is therefore the last batch only, at an arbitrary scale.

**Why it hinders:** everything downstream that the roadmap points at *consumes this number* — convergence-based termination (`gd.tex`'s algorithm literally ends with "Check Termination: evaluate loss"), Adamprop tuning, and the UI's loss chart. Building any of them on the current cell means building on fiction, then re-fixing them all when the accounting changes.

**Fix:** accumulate `(sum_loss, sample_count)` across the epoch, report `sum / count`, reset per epoch. Ten lines, but it's a *shape* decision (per-epoch aggregation vs. per-batch cell), which is why it's here and not in a bug list.

## 6. The objective is an implicit convention spread across two files

**What's wrong:** `is_classifier: bool` selects softmax in the forward pass (`net_propagator.rs:125`), while the first error signal (`delta = z − y`) and the reported loss (`Σ(z−y)²`) are hardcoded in `backprop` — the three parts of one mathematical unit (output activation ↔ loss ↔ first delta) are held together only by the author remembering they belong together. For the classifier this already produces a silent split: correct CE gradient, Brier-score reporting (the inline `todo` knows).

**Why it hinders:** each future objective — proper CE reporting, MSE with the factor 2 made explicit, label smoothing, anything from the doc's "other loss functions are possible" — has to be threaded as another flag check through the hot path, and nothing stops gradient and metric drifting apart again.

**Fix:** an `enum Objective { Regression, Classification }` (or a small trait) owning `final_activation`, `loss(z, y)`, and `first_delta(z, y)` in one place. Bonus: the missing "softmax+CE yields the same δ as identity+MSE" section of `mlp.tex` writes itself from this type.

## 7. The Tauri training session has no lifecycle

**What's wrong:** the net lives in a write-once global (`OnceLock`) and `train` spawns a detached thread per invocation with no guard (`display/src-tauri/src/lib.rs:30`). Two clicks on Train → two concurrent trainers interleaving read-snapshot/write-swap on the same parameters — memory-safe (the snapshot pattern holds) but the training trajectory is silently corrupted. There is no completion event, no way to stop a run, and no way to *re-initialize* — one net per process, forever.

**Why it hinders:** every interactive feature the playground naturally grows toward — restart with a new layout (needs re-init, blocked twice over by `OnceLock` + `const N`), a stop button, changing hyperparameters between runs — is impossible against a fire-and-forget thread and a write-once global.

**Fix:** Tauri managed state: `Mutex<Session>` with `Session { net: Option<NeuralNet>, status: Idle | Training { cancel: Arc<AtomicBool> } }`. `train` returns `Err` if a run is active; the worker checks `cancel` each epoch; explicit `training-started` / `training-finished` events replace frontend guessing.

## 8. No gradient oracle before gradient-touching work begins

**What's wrong:** the test suite is snapshot-based, and snapshots pin *whatever the code did when they were blessed* — they cannot detect that the pinned gradient is mathematically wrong, and they will demand re-blessing on every legitimate numeric change (the loss fix in §5 alone invalidates them).

**Why it hinders:** §§3, 5, 6 and the planned Adamprop all modify gradient-adjacent code; without an oracle, each change is verified by eyeball and a training curve that "looks right."

**Fix:** one finite-difference test in `network`: tiny fixed-seed net (e.g. `[3, 4, 2]`), compare `(L(θ+ε) − L(θ−ε))/2ε` against the analytic gradient elementwise within tolerance. ~30 lines, near-certain detection of sign/index/off-by-one errors, standard practice for hand-rolled backprop. Add it *first*.

---

### Suggested order of attack

**8 → 5 → 4 → 1 → 2 → 3 → 6 → 7**: the gradient oracle first so every later change is verifiable; loss accounting and the API un-forking are an afternoon together and make the emitted numbers honest; `const N` before the config type (a runtime layout can't feed it); the config type before the optimizer and the Tauri lifecycle (both consume it). After that, the roadmap items — Adamprop, normalization, batch views — land on structure that won't fight them.
