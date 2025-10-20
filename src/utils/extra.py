#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List
from typing import Optional
from typing import Tuple

import torch
from ebtorch.typing import numlike
from ebtorch.typing import realnum
from ebtorch.typing import strdev
from torch import Tensor
from torch.nn import Parameter

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "mkada",
    "kron4chain",
    "stablediv",
    "ln_default",
    "mmul3chain",
    "ls_default",
    "dimmax",
    "drive_hamiltonian",
    "couple_hamiltonian",
    "energize_hamiltonian",
]
# ──────────────────────────────────────────────────────────────────────────────


def mkada(dsize: int, device: Optional[strdev] = None) -> Tuple[Tensor, Tensor]:
    ad: Tensor = torch.zeros((dsize, dsize), device=device)
    a: Tensor = torch.zeros((dsize, dsize), device=device)

    for i in range(1, dsize):
        value: realnum = i**0.5
        ad[i, i - 1] = value
        a[i - 1, i] = value

    return ad, a


# ──────────────────────────────────────────────────────────────────────────────


def kron4chain(m1: Tensor, m2: Tensor, m3: Tensor, m4: Tensor, order: str) -> Tensor:
    if order == "ABCD":
        return ((m1.kron(m2)).kron(m3)).kron(m4)
    elif order == "BCAD":
        return (m1.kron(m2.kron(m3))).kron(m4)
    elif order == "BCDA":
        return m1.kron(m2.kron(m3).kron(m4))
    else:
        raise ValueError("Unsupported order of Kronecker product.")


def mmul3chain(m1: Tensor, m2: Tensor, m3: Tensor) -> Tensor:
    return m1 @ m2 @ m3


# ──────────────────────────────────────────────────────────────────────────────


def stablediv(num: numlike, den: numlike, eps: numlike, stabilize_both: bool = False) -> numlike:
    """Numerically stable division of two numbers.

    Args:
        num (numlike): Numerator.
        den (numlike): Denominator.
        eps (numlike): Numerical stability factor.
        stabilize_both (bool, optional): Whether to stabilize both terms. Defaults to False.
    """
    return (num + eps * stabilize_both) / (den + eps)


# ──────────────────────────────────────────────────────────────────────────────
def ln_default(
    rho: Tensor,
    idr: Tensor,
    idq1: Tensor,
    idq2: Tensor,
    d: int,
    n: int,
    device: Optional[strdev] = None,
) -> Tensor:
    rhof: Tensor = torch.zeros((d, d), dtype=torch.complex64, device=device)

    for i in range(n):
        zeros: Tensor = torch.zeros((n, n), dtype=torch.complex64, device=device)
        zeros[i, i] = 1
        kch: Tensor = kron4chain(idr, idq1, zeros, idq2, order="ABCD")
        rhof: Tensor = rhof + mmul3chain(kch, rho, kch)

    return rhof - rho


# ──────────────────────────────────────────────────────────────────────────────
def ls_default(
    rho: Tensor,
    idr: Tensor,
    idq1: Tensor,
    zn: Tensor,
    nz: Tensor,
    eg2: Tensor,
    ge2: Tensor,
    gg2: Tensor,
    nn: Tensor,
):
    kcn: Tensor = kron4chain(idr, idq1, nn, gg2, order="ABCD")
    return mmul3chain(
        kron4chain(idr, idq1, zn, eg2, order="ABCD"),
        rho,
        kron4chain(idr, idq1, nz, ge2, order="ABCD"),
    ) - (1 / 2) * (kcn @ rho + rho @ kcn)


# ──────────────────────────────────────────────────────────────────────────────
def dimmax(nmax: int, nnet: int) -> int:
    return 4 * (nmax + 1) * nnet


# ──────────────────────────────────────────────────────────────────────────────


def drive_hamiltonian(
    an: Parameter,
    wn: Parameter,
    bn: Parameter,
    aq: Parameter,
    wq: Parameter,
    bq: Parameter,
    mup: Tensor,
    ee1: Tensor,
    gg1: Tensor,
    idr: Tensor,
    idn: Tensor,
    idq1: Tensor,
    idq2: Tensor,
    hr: Tensor,
    hrq: Tensor,
    hqn: Tensor,
    diagcorr: realnum,
    dt: numlike,
    t: int,
) -> Tensor:
    mup[-1, -1] = diagcorr + torch.sum(an * torch.sin(wn * t * dt + bn))
    hn: Tensor = kron4chain(idr, idq1, mup, idq2, order="BCAD")
    hq: Tensor = kron4chain(
        idr,
        ((1 + torch.sum(aq * torch.sin(wq * t * dt + bq))) * (ee1 - gg1)),
        idn,
        idq2,
        order="BCAD",
    )
    htot: Tensor = hn + hq + hr + hrq + hqn
    return htot


# ──────────────────────────────────────────────────────────────────────────────


def couple_hamiltonian(
    lrq: Parameter,
    lqn: Parameter,
    a: Tensor,
    zo: Tensor,
    eg1: Tensor,
    idn: Tensor,
    idr: Tensor,
    idq2: Tensor,
    hn: Tensor,
    hq: Tensor,
    hr: Tensor,
) -> Tensor:
    _hrq: Tensor = lrq * kron4chain(a, eg1, idn, idq2, order="ABCD")
    hrq: Tensor = _hrq + _hrq.conj().T
    _hqn: Tensor = lqn * kron4chain(idr, eg1, zo, idq2, order="BCAD")
    hqn: Tensor = _hqn + _hqn.conj().T
    htot: Tensor = hn + hq + hr + hrq + hqn
    return htot


# ──────────────────────────────────────────────────────────────────────────────


def energize_hamiltonian(
    e: Parameter,
    mup: Tensor,
    n: int,
    idr: Tensor,
    idq1: Tensor,
    idq2: Tensor,
    hq: Tensor,
    hr: Tensor,
    hrq: Tensor,
    hqn: Tensor,
    device: Optional[strdev] = None,
) -> Tensor:
    newm: Tensor = mup * (
        torch.ones(n, dtype=torch.complex64, device=device) - torch.eye(n, dtype=torch.complex64, device=device)
    ) + torch.diag(e)

    hn: Tensor = kron4chain(idr, idq1, newm, idq2, order="BCAD")
    htot: Tensor = hn + hq + hr + hrq + hqn
    return htot
