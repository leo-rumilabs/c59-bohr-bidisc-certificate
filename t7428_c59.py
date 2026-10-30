#!/usr/bin/env python3
"""T7428: search + exact certificate for C59 of teorth/optimizationproblems.

C59 = K_2, the Bohr radius of the bidisc.  Record upper bound 0.302825279492
(P2026, Zenodo 10.5281/zenodo.22341928): a Schur function f on D^2 whose
finite Bohr majorant sum_{j,k<=28} |f_jk| r^(j+k) exceeds 1 at r = R/E proves
K_2 < R/E (the majorant is continuous and increasing in r, so it already
exceeds 1 at a slightly smaller radius, and every f_r with a larger r has
l1 norm > 1).

Family.  f is the transfer function of a unitary colligation
    U = [[A, B], [C, D]]  ((1+n+m) x (1+n+m), U* U = I),
    f(z, w) = A + B Dl (I - D Dl)^-1 C,   Dl = diag(z I_n, w I_m).
With x = (I - D Dl)^-1 C, unitarity of U applied to (1, Dl x) gives
    1 - |f|^2 = (1 - |z|^2) |x_1..n|^2 + (1 - |w|^2) |x_n+1..n+m|^2 >= 0,
and I - D Dl is invertible on D^2 since ||D|| <= 1 > ||Dl||, so f is in the
Schur class whenever U is exactly unitary.  Every rational inner function of
bidegree (n, m) has such a realization (Agler / Kummert), so n = m = 1
contains P2026's f = (SQ - P)/(SQ + P).  A Gaussian-rational U = Ut / d keeps
everything exact: with s_a stacking the first n entries of x_{a-e1} and the
last m of x_{a-e2}, x_a = D s_a and f_a = B s_a (a != 0), x_0 = C, f_0 = A,
so f_a = v_a / d^(|a|+1) with Gaussian-integer v_a.

  --search  multi-start optimisation over colligations of several shapes
            (n, m) of the S -> oo limit functional of f = (a - phi)/(1 - a
            phi), a = (S-1)/(S+1): h = (1 + phi)/(1 - phi) has Re h > 0 and
            ||f_r||_1 - 1 = (2/S) (sum_{a != 0} |h_a| r^|a| - Re h_0) + O(1/S^2)
            (finite S is worse, measured).  The best phi is rounded to an
            exact unitary (Cayley transform of a rounded skew-Hermitian K),
            composed with the Moebius map at S = s^2 (sqrt(1 - a^2) rational),
            and written as <out_dir>/c59_cert.json with the smallest R/E the
            exact majorant certifies.
  --check   run the stdlib-only exact checker on that certificate: exit 0 only
            if it proves K_2 < R/E with R/E strictly below 0.302825279492.

The checker (check_cert / main_check) uses integers and fractions.Fraction
only: exact unitarity Ut* Ut = d^2 I, the Gaussian-integer recurrence, n_a =
isqrt(|v_a|^2) <= |v_a|, and the integer comparison
    A = sum_{j,k<=N} n_a R^|a| (E d)^(2N-|a|)  >  B = d (E d)^(2N),
i.e. sum |f_a| (R/E)^|a| >= A/B > 1.  Outputs never go into the repo:
--out_dir defaults to /data/logs/ttl=58d/t7428/c59; --log_file is accepted
and ignored (the caller redirects stdout).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from decimal import Decimal
from decimal import localcontext
from fractions import Fraction
from math import isqrt

DEFAULT_OUT = '/data/logs/ttl=58d/t7428/c59'
CERT_NAME = 'c59_cert.json'
RECORD = Fraction(302825279492, 10**12)
RECORD_TEXT = '0.302825279492'
RECORD_REF = 'P2026 (Patel), Zenodo 10.5281/zenodo.22341928'
SHAPES = ((1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 2), (2, 3), (3, 3))


# ============================================================ exact checker (stdlib only)
class Invalid(Exception):
    pass


def gmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def gdot(row, vec):
    re = im = 0
    for a, b in zip(row, vec):
        if a[0] or a[1]:
            re += a[0] * b[0] - a[1] * b[1]
            im += a[0] * b[1] + a[1] * b[0]
    return re, im


def _int(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, str)):
        raise Invalid(f'{what}: expected an integer or a decimal string, got {v!r}')
    try:
        return int(v)
    except ValueError:
        raise Invalid(f'{what}: not an integer: {v!r}') from None


def parse_cert(cert):
    try:
        n, m, N = (_int(cert[k], k) for k in ('n', 'm', 'N'))
        d = _int(cert['den'], 'den')
        R, E = (_int(v, 'r') for v in cert['r'])
        rows = cert['U']
    except (KeyError, TypeError, ValueError) as error:
        raise Invalid(f'missing or malformed field: {error!r}') from None
    if n < 0 or m < 0 or n + m < 1:
        raise Invalid(f'need n, m >= 0 and n + m >= 1, got n={n} m={m}')
    if not 1 <= N <= 400:
        raise Invalid(f'N must be in [1, 400], got {N}')
    if d <= 0 or not 0 < R < E:
        raise Invalid(f'need den > 0 and 0 < R < E, got den={d} R={R} E={E}')
    k = 1 + n + m
    if not isinstance(rows, list) or len(rows) != k or any(not isinstance(r, list) or len(r) != k for r in rows):
        raise Invalid(f'U must be a {k} x {k} matrix of [re, im] pairs')
    try:
        U = [[(_int(e[0], 'U'), _int(e[1], 'U')) for e in row] for row in rows]
    except (TypeError, IndexError):
        raise Invalid('U entries must be [re, im] integer pairs') from None
    return n, m, d, U, N, R, E


def unitary_defect(U, d):
    """Entries (i, j) where (Ut* Ut)_ij != d^2 [i == j]; empty iff U = Ut/d is exactly unitary."""
    k = len(U)
    bad = []
    for i in range(k):
        for j in range(k):
            re = im = 0
            for r in range(k):
                a, b = U[r][i], U[r][j]
                re += a[0] * b[0] + a[1] * b[1]       # conj(a) * b
                im += a[0] * b[1] - a[1] * b[0]
            if (re, im) != ((d * d if i == j else 0), 0):
                bad.append((i, j))
    return bad


def coefficients(n, m, U, N):
    """Yield (j, k, v) for 0 <= j, k <= N, where f_jk = v / d^(j+k+1) (v a Gaussian integer)."""
    A = U[0][0]
    B = U[0][1:]
    C = [U[i][0] for i in range(1, 1 + n + m)]
    D = [U[i][1:] for i in range(1, 1 + n + m)]
    zero = (0, 0)
    prev = None
    for j in range(N + 1):
        cur = []
        for k in range(N + 1):
            if j == 0 and k == 0:
                x, v = list(C), A
            else:
                s = ([prev[k][i] for i in range(n)] if j else [zero] * n) + \
                    ([cur[k - 1][n + i] for i in range(m)] if k else [zero] * m)
                x = [gdot(row, s) for row in D]
                v = gdot(B, s)
            cur.append(x)
            yield j, k, v
        prev = cur


def check_cert(cert):
    """Decide the certificate exactly.  Returns a dict; raises Invalid on a malformed or non-unitary one."""
    n, m, d, U, N, R, E = parse_cert(cert)
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)          # coefficients have ~10^4 digits; they are hashed as decimal text
    bad = unitary_defect(U, d)
    if bad:
        raise Invalid(f'U is not exactly unitary: (U* U) differs from I at {bad[:6]}')
    ed = E * d
    pow_r = [1] * (2 * N + 1)
    pow_ed = [1] * (2 * N + 1)
    for t in range(1, 2 * N + 1):
        pow_r[t] = pow_r[t - 1] * R
        pow_ed[t] = pow_ed[t - 1] * ed
    total = 0
    h_coef = hashlib.sha256()
    h_floor = hashlib.sha256()
    for j, k, v in coefficients(n, m, U, N):
        sq = v[0] * v[0] + v[1] * v[1]
        fl = isqrt(sq)
        if not fl * fl <= sq < (fl + 1) * (fl + 1):
            raise Invalid('isqrt self-check failed')
        t = j + k
        total += fl * pow_r[t] * pow_ed[2 * N - t]
        h_coef.update(f'{j},{k},{v[0]},{v[1]}\n'.encode())
        h_floor.update(f'{j},{k},{fl}\n'.encode())
    bound_b = d * pow_ed[2 * N]
    r = Fraction(R, E)
    with localcontext() as ctx:
        ctx.prec = 40
        excess = Decimal(total - bound_b) / Decimal(bound_b)
        r_dec = Decimal(R) / Decimal(E)
        gain = Decimal(RECORD.numerator) / Decimal(RECORD.denominator) - r_dec
    return {'n': n, 'm': m, 'N': N, 'coefficients': (N + 1) ** 2, 'den_digits': len(str(d)),
            'r': f'{R}/{E}', 'r_decimal': str(r_dec), 'majorant_gt_1': total > bound_b,
            'beats_record': total > bound_b and r < RECORD, 'excess': str(excess), 'gain': str(gain),
            'sha256_A': hashlib.sha256(str(total).encode()).hexdigest(),
            'sha256_B': hashlib.sha256(str(bound_b).encode()).hexdigest(),
            'sha256_coefficients': h_coef.hexdigest(), 'sha256_floors': h_floor.hexdigest()}


def main_check(path):
    """Print the verdict; 0 iff the certificate proves K_2 < R/E < 0.302825279492."""
    try:
        with open(path, 'rb') as fh:
            raw = fh.read()
    except OSError as error:
        print(f'check: FAIL: cannot read certificate {path}: {error}', flush=True)
        return 1
    sha = hashlib.sha256(raw).hexdigest()
    print(f'check: certificate {path} sha256 {sha}', flush=True)
    t0 = time.time()
    try:
        res = check_cert(json.loads(raw))
    except (Invalid, ValueError) as error:
        print(f'check: FAIL: invalid certificate: {error}', flush=True)
        return 1
    print(f'check: colligation n={res["n"]} m={res["m"]}, U exactly unitary (den {res["den_digits"]} digits); '
          f'{res["coefficients"]} exact coefficients 0 <= j,k <= {res["N"]} ({time.time() - t0:.1f}s)', flush=True)
    print(f'check: A/B - 1 = {res["excess"]}  (A > B: {res["majorant_gt_1"]})', flush=True)
    for key in ('sha256_A', 'sha256_B', 'sha256_coefficients', 'sha256_floors'):
        print(f'check: {key} {res[key]}', flush=True)
    print(f'check: certified upper bound K_2 < {res["r"]} = {res["r_decimal"]}', flush=True)
    print(f'check: previous best upper bound   {RECORD_TEXT} ({RECORD_REF})', flush=True)
    if not res['majorant_gt_1']:
        print('check: FAIL: the finite majorant does not exceed 1 at r', flush=True)
        return 1
    if not res['beats_record']:
        print(f'check: FAIL: certified r = {res["r_decimal"]} is not strictly below {RECORD_TEXT}', flush=True)
        return 1
    print(f'check: PASS: K_2 < {res["r_decimal"]}, strictly below {RECORD_TEXT} by {res["gain"]}', flush=True)
    return 0


# ============================================================ search (numpy / scipy, imported lazily)
GRID = 96
RHO = 0.6


def _np():
    import numpy as np
    return np


def unitary_from(x, k):
    np = _np()
    from scipy.linalg import expm
    H = np.zeros((k, k), complex)
    iu = np.triu_indices(k, 1)
    nu = len(iu[0])
    H[iu] = x[:nu] + 1j * x[nu:2 * nu]
    H = H - H.conj().T
    H[np.diag_indices(k)] = 1j * np.asarray(x[2 * nu:2 * nu + k])
    return expm(H)


class Evaluator:
    """|h_a| (and Re h_0) of h = (1 + phi)/(1 - phi) by a 2-D FFT of phi on the torus of radius RHO."""

    def __init__(self, n, m, grid=GRID, rho=RHO):
        np = _np()
        self.n, self.m, self.k = n, m, 1 + n + m
        th = np.exp(2j * np.pi * np.arange(grid) / grid) * rho
        Z, W = np.meshgrid(th, th, indexing='ij')
        self.dl = np.concatenate([np.repeat(Z.ravel()[:, None], n, 1), np.repeat(W.ravel()[:, None], m, 1)], 1)
        self.grid, self.rho = grid, rho
        self.deg = np.add.outer(np.arange(grid), np.arange(grid))

    def coefs(self, U):
        np = _np()
        A, B, C, D = U[0, 0], U[0, 1:], U[1:, 0], U[1:, 1:]
        mats = np.eye(self.k - 1)[None] - D[None] * self.dl[:, None, :]
        x = np.linalg.solve(mats, np.broadcast_to(C, (len(self.dl), self.k - 1))[..., None])[..., 0]
        phi = (A + (B[None] * self.dl * x).sum(1)).reshape(self.grid, self.grid)
        with np.errstate(all='ignore'):
            h = (1 + phi) / (1 - phi)
        c = np.fft.fft2(h) / self.grid ** 2
        return np.abs(c) / self.rho ** self.deg, c[0, 0].real

    def ratio(self, x, r):
        np = _np()
        ac, re0 = self.coefs(unitary_from(x, self.k))
        if not (re0 > 1e-9 and np.all(np.isfinite(ac))):
            return 0.0
        t = ac * r ** self.deg
        return float((t.sum() - t[0, 0]) / re0)

    def rstar(self, x, lo=0.2, hi=0.6):
        np = _np()
        ac, re0 = self.coefs(unitary_from(x, self.k))
        if not (re0 > 1e-9 and np.all(np.isfinite(ac))):
            return 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            t = ac * mid ** self.deg
            if t.sum() - t[0, 0] > re0:
                hi = mid
            else:
                lo = mid
        return hi


def search_job(job):
    """Multi-start local search on one shape until the job's deadline; returns its best (r*, x)."""
    np = _np()
    from scipy.optimize import minimize
    n, m, seed, until = job['n'], job['m'], job['seed'], job['until']
    ev = Evaluator(n, m)
    k = 1 + n + m
    rng = np.random.default_rng(seed)
    best = {'n': n, 'm': m, 'seed': seed, 'rstar': 1.0, 'x': None, 'starts': 0}
    while time.time() < until:
        seen = {'x': rng.normal(size=k * k), 'v': 0.0}

        def objective(x, r):
            if time.time() > until:
                raise _Deadline
            v = ev.ratio(x, r)
            if r > 0.3 and v > seen['v']:
                seen.update(x=np.array(x), v=v)
            return -v

        try:
            res = minimize(objective, seen['x'], args=(0.30,), method='Nelder-Mead',
                           options={'maxfev': 200 * k * k, 'adaptive': True})
            minimize(objective, res.x, args=(0.30283,), method='BFGS', options={'gtol': 1e-12})
        except _Deadline:
            pass
        r = ev.rstar(seen['x'])
        best['starts'] += 1
        if r < best['rstar']:
            best.update(rstar=r, x=[float(v) for v in seen['x']])
    return best


class _Deadline(Exception):
    pass


# ---------------------------------------------------------- exact Gaussian-rational matrices
def cq(z, den):
    """Round a complex float to a Gaussian rational with denominator den."""
    return Fraction(round(z.real * den), den), Fraction(round(z.imag * den), den)


def qadd(a, b):
    return a[0] + b[0], a[1] + b[1]


def qsub(a, b):
    return a[0] - b[0], a[1] - b[1]


def qmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def qdiv(a, b):
    nb = b[0] * b[0] + b[1] * b[1]
    return (a[0] * b[0] + a[1] * b[1]) / nb, (a[1] * b[0] - a[0] * b[1]) / nb


def qmatmul(X, Y):
    k = len(Y)
    return [[_qsum(qmul(X[i][r], Y[r][j]) for r in range(k)) for j in range(len(Y[0]))] for i in range(len(X))]


def _qsum(items):
    re = im = Fraction(0)
    for a in items:
        re += a[0]
        im += a[1]
    return re, im


def qinv(X):
    k = len(X)
    zero, one = (Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))
    M = [list(X[i]) + [one if i == j else zero for j in range(k)] for i in range(k)]
    for c in range(k):
        p = next(r for r in range(c, k) if M[r][c] != zero)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [qdiv(e, piv) for e in M[c]]
        for r in range(k):
            if r != c and M[r][c] != zero:
                f = M[r][c]
                M[r] = [qsub(e, qmul(f, g)) for e, g in zip(M[r], M[c])]
    return [row[k:] for row in M]


def exact_unitary(U, den_bits=64):
    """An exactly unitary Gaussian-rational matrix within ~2^-den_bits of the float unitary U:
    Q (I - K)(I + K)^-1 with K skew-Hermitian rounded, Q a +-1 diagonal keeping I + QU well conditioned."""
    np = _np()
    k = len(U)
    best = None
    for mask in range(2 ** k):
        q = np.array([-1.0 if mask >> i & 1 else 1.0 for i in range(k)])
        smin = np.linalg.svd(np.eye(k) + q[:, None] * U, compute_uv=False)[-1]
        if best is None or smin > best[0]:
            best = (smin, q)
    q = best[1]
    V = q[:, None] * U
    K = (np.eye(k) - V) @ np.linalg.inv(np.eye(k) + V)
    den = 2 ** den_bits
    Kq = [[None] * k for _ in range(k)]
    for i in range(k):
        Kq[i][i] = (Fraction(0), Fraction(round(K[i, i].imag * den), den))
        for j in range(i + 1, k):
            a = cq((K[i, j] - K[j, i].conjugate()) / 2, den)
            Kq[i][j], Kq[j][i] = a, (-a[0], a[1])
    one = (Fraction(1), Fraction(0))
    eye = [[one if i == j else (Fraction(0), Fraction(0)) for j in range(k)] for i in range(k)]
    im_k = [[qsub(eye[i][j], Kq[i][j]) for j in range(k)] for i in range(k)]
    ip_k = [[qadd(eye[i][j], Kq[i][j]) for j in range(k)] for i in range(k)]
    W = qmatmul(im_k, qinv(ip_k))
    return [[(W[i][j][0] * int(q[i]), W[i][j][1] * int(q[i])) for j in range(k)] for i in range(k)]


def moebius(U, s):
    """Colligation of f = (phi - a)/(1 - a phi), a = (s^2 - 1)/(s^2 + 1), sqrt(1 - a^2) = 2s/(s^2 + 1)."""
    a = Fraction(s * s - 1, s * s + 1)
    sig = Fraction(2 * s, s * s + 1)
    A, B = U[0][0], U[0][1:]
    C = [row[0] for row in U[1:]]
    D = [row[1:] for row in U[1:]]
    g = qdiv((Fraction(1), Fraction(0)), (1 - a * A[0], -a * A[1]))     # 1 / (1 - a A)
    A2 = qmul(qsub(A, (a, Fraction(0))), g)
    sg = (sig * g[0], sig * g[1])
    B2 = [qmul(b, sg) for b in B]
    C2 = [qmul(c, sg) for c in C]
    ag = (a * g[0], a * g[1])
    D2 = [[qadd(D[i][j], qmul(qmul(C[i], B[j]), ag)) for j in range(len(B))] for i in range(len(C))]
    return [[A2] + B2] + [[C2[i]] + D2[i] for i in range(len(C))]


def integer_form(Uq):
    """(den, Ut) with Uq = Ut / den, Ut Gaussian-integer."""
    den = 1
    for row in Uq:
        for re, im in row:
            for v in (re, im):
                den = den * v.denominator // _gcd(den, v.denominator)
    return den, [[(int(re * den), int(im * den)) for re, im in row] for row in Uq]


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def min_radius(n, m, U, den, N, prec=120):
    """Smallest r (Decimal) with sum_{j,k<=N} |f_jk| r^(j+k) >= 1, from the exact v (search side only)."""
    scale = 10 ** prec
    by_deg = [0] * (2 * N + 1)
    for j, k, v in coefficients(n, m, U, N):
        by_deg[j + k] += isqrt((v[0] * v[0] + v[1] * v[1]) * scale * scale)
    with localcontext() as ctx:
        ctx.prec = prec + 20
        c = [Decimal(by_deg[t]) / (Decimal(scale) * Decimal(den) ** (t + 1)) for t in range(2 * N + 1)]
        lo, hi = Decimal('0.25'), Decimal('0.34')

        def maj(r):
            acc = Decimal(0)
            for coef in reversed(c):
                acc = acc * r + coef
            return acc

        if maj(hi) <= 1:
            return None
        for _ in range(200):
            mid = (lo + hi) / 2
            if maj(mid) > 1:
                hi = mid
            else:
                lo = mid
        return hi


def certify(best, out_dir, N, s, e_digits):
    """Exact certificate for the best colligation; writes <out_dir>/c59_cert.json and returns its path."""
    np = _np()
    n, m = best['n'], best['m']
    U = unitary_from(np.array(best['x']), 1 + n + m)
    Uq = moebius(exact_unitary(U), s)
    den, Ut = integer_form(Uq)
    if unitary_defect(Ut, den):
        raise RuntimeError('rounded colligation is not exactly unitary (bug)')
    r_min = min_radius(n, m, Ut, den, N)
    E = 10 ** e_digits
    cert = {'constant': 'C59', 'quantity': 'K_2, Bohr radius of the bidisc',
            'claim': 'K_2 < R/E with [R, E] = r',
            'family': 'transfer function of the unitary colligation U = Ut/den, block sizes 1, n, m',
            'n': n, 'm': m, 'N': N, 'den': str(den),
            'U': [[[str(re), str(im)] for re, im in row] for row in Ut],
            'r': None,
            'previous_best': RECORD_TEXT, 'previous_ref': RECORD_REF,
            'provenance': {'search_rstar_limit': best['rstar'], 'S': f'{s}^2', 'seed': best['seed'],
                           'host': os.uname().nodename, 'script': 'ReFin scripts/t7428_c59.py',
                           'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}}
    if r_min is None:
        R = E - 1
    else:
        R = int(r_min * E) + 1
    print(f'certify: shape ({n},{m}) den {len(str(den))} digits; exact majorant reaches 1 at r ~ '
          f'{r_min}; trying R/E = {R}/10^{e_digits}', flush=True)
    for _ in range(6):
        cert['r'] = [str(R), str(E)]
        res = check_cert(cert)
        if res['majorant_gt_1']:
            break
        R += 1
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, CERT_NAME)
    with open(path, 'w') as fh:
        json.dump(cert, fh, indent=1)
        fh.write('\n')
    print(f'certify: wrote {path}: r = {R}/10^{e_digits}, majorant > 1: {res["majorant_gt_1"]}', flush=True)
    return path


def polish(best, seconds):
    """Nelder-Mead on r* itself from the best start, for at most `seconds`."""
    np = _np()
    from scipy.optimize import minimize
    ev = Evaluator(best['n'], best['m'])
    until = time.time() + seconds
    seen = {'x': np.array(best['x']), 'r': ev.rstar(np.array(best['x']))}

    def objective(x):
        if time.time() > until:
            raise _Deadline
        r = ev.rstar(x, 0.29, 0.32)
        if r < seen['r']:
            seen.update(x=np.array(x), r=r)
        return r

    try:
        minimize(objective, seen['x'], method='Nelder-Mead',
                 options={'maxfev': 4000, 'xatol': 1e-13, 'fatol': 1e-18, 'adaptive': True})
    except _Deadline:
        pass
    print(f'search: polish r* {best["rstar"]:.16f} -> {seen["r"]:.16f}', flush=True)
    return dict(best, rstar=float(seen['r']), x=[float(v) for v in seen['x']])


def run_search(args):
    from multiprocessing import Pool
    t0 = time.time()
    until = t0 + args.minutes * 60
    n_jobs = max(len(SHAPES), int(args.minutes * 60 * args.workers / args.per_job_s))
    jobs = [{'n': SHAPES[i % len(SHAPES)][0], 'm': SHAPES[i % len(SHAPES)][1], 'seed': 7428_000 + i,
             'until': min(until, t0 + (i // args.workers + 1) * args.per_job_s)} for i in range(n_jobs)]
    print(f'search: {n_jobs} jobs over shapes {list(SHAPES)}, {args.minutes} min on {args.workers} workers; '
          f'record {RECORD_TEXT}', flush=True)
    best = None
    per_shape = {}
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(search_job, jobs):
            key = f'{res["n"]},{res["m"]}'
            per_shape.setdefault(key, {'best': 1.0, 'starts': 0})
            per_shape[key]['starts'] += res['starts']
            per_shape[key]['best'] = min(per_shape[key]['best'], res['rstar'])
            if res['x'] is not None and (best is None or res['rstar'] < best['rstar'] - 1e-15):
                best = res
            print(f'search: shape ({res["n"]},{res["m"]}) seed {res["seed"]} starts {res["starts"]} '
                  f'r* {res["rstar"]:.13f} (best {best["rstar"] if best else 1.0:.13f}, '
                  f'{time.time() - t0:.0f}s)', flush=True)
    with open(os.path.join(args.out_dir, 'c59_search_summary.json'), 'w') as fh:
        json.dump({'record': RECORD_TEXT, 'best': best, 'per_shape': per_shape, 'host': os.uname().nodename,
                   'seconds': round(time.time() - t0)}, fh, indent=1)
    if best is None:
        raise RuntimeError('search produced no candidate')
    best = polish(best, min(600.0, 30.0 + args.minutes * 2))
    print(f'search: best limit r* {best["rstar"]:.16f} from shape ({best["n"]},{best["m"]}); per shape '
          + ', '.join(f'({k}) {v["best"]:.13f}/{v["starts"]} starts' for k, v in sorted(per_shape.items())),
          flush=True)
    return certify(best, args.out_dir, args.N, 10 ** args.s_exp, args.e_digits)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--search', action='store_true', help='search and write the certificate')
    ap.add_argument('--check', action='store_true', help='check the certificate exactly')
    ap.add_argument('--out_dir', default=DEFAULT_OUT)
    ap.add_argument('--cert', default='', help=f'certificate to check (default <out_dir>/{CERT_NAME})')
    ap.add_argument('--minutes', type=float, default=60.0, help='search wall-clock budget')
    ap.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument('--per_job_s', type=float, default=600.0, help='seconds per search job (one shape, one seed)')
    ap.add_argument('--N', type=int, default=40, help='certificate truncation 0 <= j,k <= N')
    ap.add_argument('--s_exp', type=int, default=12, help='S = (10^s_exp)^2 in a = (S-1)/(S+1)')
    ap.add_argument('--e_digits', type=int, default=16, help='r = R / 10^e_digits')
    ap.add_argument('--log_file', default='', help='accepted and ignored (the caller redirects stdout)')
    args = ap.parse_args(argv)
    if not (args.search or args.check):
        ap.error('pass --search and/or --check')
    os.makedirs(args.out_dir, exist_ok=True)
    path = args.cert or os.path.join(args.out_dir, CERT_NAME)
    if args.search:
        path = run_search(args)
    if args.check:
        rc = main_check(path)
        try:
            with open(path, 'rb') as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
            r = json.loads(open(path).read())['r']
            bound = str(Decimal(int(r[0])) / Decimal(int(r[1])))
        except (OSError, ValueError, KeyError, TypeError):
            sha, bound = 'absent', ''
        print('T7428_RESULT ' + json.dumps({'cert': os.path.abspath(path), 'sha256': sha, 'exit': rc,
                                            'certified_bound': bound if rc == 0 else '',
                                            'previous_best': RECORD_TEXT}), flush=True)
        print(f'check: exit {rc}', flush=True)
        return rc
    return 0


if __name__ == '__main__':
    sys.exit(main())
