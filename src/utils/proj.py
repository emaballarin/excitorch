#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List
from typing import Optional
from typing import Tuple

import torch
from ebtorch.typing import realnum
from ebtorch.typing import strdev
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "mkee",
    "mkgg",
    "mkeg",
    "mkge",
    "mkzo",
    "mkoz",
    "mkzz",
    "mknn",
    "mkzn",
    "mknz",
    "mkuu",
    "mkdd",
    "mkeye",
    "mkallproj",
    "mkids",
    "mkallp_2q",
]


# ──────────────────────────────────────────────────────────────────────────────
def mkproj(
    size: int,
    idxl: int,
    idxr: int,
    value: realnum,
    device: Optional[strdev] = None,
) -> Tensor:
    """
    Build a 2D projector of given `size` with `value` at position `[idxl, idxr]`.
    """
    projector: Tensor = torch.zeros((size, size), dtype=torch.complex64, device=device)
    projector[idxl, idxr] = value
    return projector


# ──────────────────────────────────────────────────────────────────────────────

# Projectors for single-qubit states


def mkee(device: Optional[strdev] = None) -> Tensor:
    return mkproj(2, 0, 0, 1, device=device)


def mkgg(device: Optional[strdev] = None) -> Tensor:
    return mkproj(2, 1, 1, 1, device=device)


def mkeg(device: Optional[strdev] = None) -> Tensor:
    return mkproj(2, 0, 1, 1, device=device)


def mkge(device: Optional[strdev] = None) -> Tensor:
    return mkproj(2, 1, 0, 1, device=device)


# ──────────────────────────────────────────────────────────────────────────────

# Projectors for n-qubit states


def mkzo(n: int, device: Optional[strdev] = None) -> Tensor:
    return mkproj(n, 0, 1, 1, device=device)


def mkoz(n: int, device: Optional[strdev] = None) -> Tensor:
    return mkproj(n, 1, 0, 1, device=device)


def mkzz(n: int, device: Optional[strdev] = None) -> Tensor:
    return mkproj(n, 0, 0, 1, device=device)


def mknn(n: int, device: Optional[strdev] = None) -> Tensor:
    return mkproj(n, n - 1, n - 1, 1, device=device)


def mkzn(n: int, device: Optional[strdev] = None) -> Tensor:
    return mkproj(n, 0, n - 1, 1, device=device)


def mknz(n: int, device: Optional[strdev] = None) -> Tensor:
    return mkproj(n, n - 1, 0, 1, device=device)


def mkuu(n: int, device: Optional[strdev] = None) -> Tensor:
    return mkproj(n, 1, 1, 1, device=device)


def mkdd(n: int, device: Optional[strdev] = None) -> Tensor:
    return mkproj(n, 2, 2, 1, device=device)


# ──────────────────────────────────────────────────────────────────────────────


# Identity
def mkeye(n: int, device: Optional[strdev] = None) -> Tensor:
    """
    Build a complex 2D identity matrix of size `n`.
    """
    return torch.eye(n, dtype=torch.complex64, device=device)


# ──────────────────────────────────────────────────────────────────────────────
def mkallproj(n: int, device: Optional[strdev] = None) -> Tuple[Tensor, ...]:
    return (
        mkee(device=device),
        mkgg(device=device),
        mkeg(device=device),
        mkge(device=device),
        mkzo(n, device=device),
        mkoz(n, device=device),
        mkzz(n, device=device),
        mknn(n, device=device),
        mkzn(n, device=device),
        mknz(n, device=device),
        mkuu(n, device=device),
        mkdd(n, device=device),
        mkeye(n, device=device),
    )


def mkids(n: int, nmax: int, device: Optional[strdev] = None) -> Tuple[Tensor, ...]:
    return (
        mkeye(nmax + 1, device=device),
        mkeye(2, device=device),
        mkeye(2, device=device),
        mkeye(n, device=device),
    )


# ──────────────────────────────────────────────────────────────────────────────
def mkallp_2q(n: int, device: Optional[strdev] = None) -> Tuple[Tensor, ...]:
    ee, gg, eg, ge, zo, oz, zz, nn, zn, nz, uu, dd, eye = mkallproj(n, device=device)
    return (
        ee,
        ee.clone(),
        gg,
        gg.clone(),
        eg,
        eg.clone(),
        ge,
        ge.clone(),
        zo,
        oz,
        zz,
        nn,
        zn,
        nz,
        uu,
        dd,
        eye,
    )
