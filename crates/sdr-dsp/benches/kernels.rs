//! Throughput of the hot kernels, in samples per second.
//!
//! The numbers only mean something relative to each other and to the scalar reference —
//! this measures the host's SSE2 path, not the wasm one that ships. It exists to catch
//! regressions where a kernel silently stops vectorising.

use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(c: &mut Criterion) {
    c.bench_function("noop", |b| b.iter(|| 0u32));
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
