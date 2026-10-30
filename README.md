# C59: certified upper bound K_2 < 0.302825279490988 for the Bohr radius of the bidisc

Certificate and exact checker for constant C59 of [teorth/optimizationproblems](https://github.com/teorth/optimizationproblems) ([constants/59a.md](https://teorth.github.io/optimizationproblems/constants/59a.html)).

**Claim.** K_2 < 3028252794909880/10^16 = 0.302825279490988. Previous best upper bound: 0.302825279492 (P2026, Patel, an exact bound with a Lean proof, Zenodo 10.5281/zenodo.22341928). The improvement is about 1.0e-12 (0.302825279492 - 0.302825279490988 = 1.012e-12).

## Files

| file | bytes | SHA-256 |
| ---- | ----- | ------- |
| `c59_cert.json` | 3583 | `6d2e39e8741fb5e73593d4db007c68831243d32fe4ae298bc4fe750033ebedab` |
| `t7428_c59.py` | 25131 | `f78b73aad627ec446af3af00252631c00f482b9eb827109d99a783dc0cac7db4` |
| `SHA256SUMS` | | `sha256sum` lines for the two files above |

## What the certificate proves

f(z, w) = A + B D(z,w) (I - D_U D(z,w))^-1 C with D(z,w) = diag(z, w) is the transfer function of the unitary colligation U = [[A, B], [C, D_U]] (3 x 3, Gaussian-rational entries over one common 149-digit denominator) stored in the certificate. U is exactly unitary, so f maps the bidisc into the unit disc (a Schur function). The checker recomputes the 1681 coefficients f_jk, 0 <= j,k <= 40, exactly (integers and fractions.Fraction only) and shows the finite Bohr majorant sum |f_jk| r^(j+k) exceeds 1 (its lower bound A/B satisfies A/B - 1 = 2.2e-42) at r = 3028252794909880/10^16. Since the majorant increases with r, K_2 < 0.302825279490988.

## Reproduce (Python 3.8 or later, standard library only)

Clone this repository at the commit linked from the constant page, then:

```sh
sha256sum -c SHA256SUMS
python3 -I t7428_c59.py --check --cert c59_cert.json --out_dir .
```

It exits 0 and prints, among other lines:

```
check: certified upper bound K_2 < 3028252794909880/10000000000000000 = 0.302825279490988
check: previous best upper bound   0.302825279492
check: PASS: K_2 < 0.302825279490988, strictly below 0.302825279492
```

`t7428_c59.py` also contains the search that produced the certificate (`--search`); only `--check` is needed to verify it, and `--out_dir .` keeps it from creating its default output directory.

## Provenance and AI use

The search, the certificate and the checker were produced by an AI agent run by Leo Tang. The checker has not been independently re-implemented or formally verified.
