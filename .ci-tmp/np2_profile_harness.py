"""THROWAWAY x86 profiling harness for the numpy-2 errgen-composition slowdown.

Run on a GitHub x86_64 runner under several numpy/thread configs to (a) reproduce
the slowdown, (b) test the OpenBLAS-thread hypothesis, (c) get an x86 cProfile, and
(d) measure which numpy primitive (tiny matmul / norm / scalar-array) is the locus.

Usage: python np2_profile_harness.py [N_pairs] [--profile]
"""
import sys, os, time, cProfile, pstats, io
import numpy as np
from itertools import product

N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6000
do_profile = '--profile' in sys.argv

print("=" * 70)
print(f"numpy {np.__version__}")
for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NPY_NUM_THREADS"):
    print(f"  env {v}={os.environ.get(v, '<unset>')}")
try:
    import threadpoolctl
    for p in threadpoolctl.threadpool_info():
        print(f"  threadpool: {p.get('user_api')}/{p.get('internal_api')} "
              f"num_threads={p.get('num_threads')} ({p.get('prefix', '')})")
except Exception as e:
    print(f"  threadpoolctl unavailable: {e}")
print(f"  os.cpu_count()={os.cpu_count()}")
print("=" * 70)

# ---- primitive microbench (the exact ops the per-pair loop performs) ----
rng = np.random.default_rng(0)
def bench(name, fn, n):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    print(f"  {name:30s} {dt/n*1e6:9.3f} us/call ({n} calls, {dt:.3f}s)")

print("PRIMITIVE MICROBENCH")
for dim, n in [(16, 100_000), (64, 30_000)]:
    A = (rng.standard_normal((dim, dim)) + 1j*rng.standard_normal((dim, dim)))
    B = (rng.standard_normal((dim, dim)) + 1j*rng.standard_normal((dim, dim)))
    rate = 0.37 + 0.12j
    print(f" --- {dim}x{dim} complex128 ---")
    bench("matmul A@B", lambda: A @ B, n)
    bench("scalar*array rate*A", lambda: rate * A, n)
    bench("inplace M+=rate*A", (lambda M=np.zeros_like(A): M.__iadd__(rate*A)), n)
    bench("subtract A-B", lambda: A - B, n)
    bench("linalg.norm(A-B)", lambda: np.linalg.norm(A - B), n)

# ---- composition-loop reproduction (mirrors test_errorgen_composition 2Q inner loop) ----
from pygsti.baseobjs import QubitSpace
from pygsti.baseobjs.errorgenbasis import CompleteElementaryErrorgenBasis
from pygsti.errorgenpropagation.localstimerrorgen import LocalStimErrorgenLabel as _LSE
from pygsti.tools import errgenproptools as _eprop

basis2 = CompleteElementaryErrorgenBasis('PP', QubitSpace(2), default_label_type='local')
lbls = list(basis2.labels)
dct = {lbl: mat for lbl, mat in zip(lbls, basis2.elemgen_matrices)}
stim_lbls = [_LSE.cast(l) for l in lbls]
srng = np.random.default_rng(20260614)
ii = srng.integers(0, len(lbls), N); jj = srng.integers(0, len(lbls), N)
pairs = [(lbls[i], lbls[j]) for i, j in zip(ii, jj)]
stim_pairs = [(stim_lbls[i], stim_lbls[j]) for i, j in zip(ii, jj)]

def work():
    for (p1, p2), (s1, s2) in zip(pairs, stim_pairs):
        numeric = _eprop.error_generator_composition_numerical(p1, p2, dct)
        analytic = _eprop.error_generator_composition(s1, s2)
        mat = _eprop.errorgen_layer_to_matrix(analytic, 2, errorgen_matrix_dict=dct)
        _ = np.linalg.norm(numeric - mat)

print("COMPOSITION LOOP")
work()
t0 = time.perf_counter(); work(); dt = time.perf_counter() - t0
print(f"  {N} pairs in {dt:.3f}s  ({dt/N*1000:.4f} ms/pair)  "
      f"=> extrapolated 57600-pair test loop ~= {dt/N*57600:.1f}s")

if do_profile:
    pr = cProfile.Profile()
    pr.enable(); work(); pr.disable()
    for key in ('tottime', 'cumulative'):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(key).print_stats(20)
        print(f"\n---- cProfile sorted by {key} ----\n" + s.getvalue())
