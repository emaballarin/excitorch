#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from collections.abc import Callable
from typing import List
from typing import Optional
from typing import Tuple

import torch
from ebtorch.typing import numlike
from ebtorch.typing import realnum
from torch import Tensor
from torch.nn import Parameter

from .extra import couple_hamiltonian
from .extra import drive_hamiltonian
from .extra import energize_hamiltonian

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "commut",
    "rk4step",
    "psink_evol",
    "driven_psink_evol",
    "coupled_psink_evol",
    "energized_psink_evol",
]


# ──────────────────────────────────────────────────────────────────────────────
rk4weights: Tuple[int, int, int, int] = (1, 2, 2, 1)


# ──────────────────────────────────────────────────────────────────────────────
def commut(a: Tensor, b: Tensor) -> Tensor:
    return a @ b - b @ a


# ──────────────────────────────────────────────────────────────────────────────
def evolve_by(rho: numlike, dt: numlike, ki: numlike) -> numlike:
    return rho + dt * ki


def khcomm(h: Tensor, rho: Tensor) -> Tensor:
    return -1j * commut(h, rho)


def ns_evolve(
    rhon: Tensor,
    rhos: Tensor,
    lamn: numlike,
    lams: numlike,
    lnfx: Callable[[Tensor], numlike],
    lsfx: Callable[[Tensor], numlike],
) -> numlike:
    return lamn * lnfx(rhon) + lams * lsfx(rhos)


# ──────────────────────────────────────────────────────────────────────────────
def rk4step(
    rho: Tensor,
    h: Tensor,
    lamn: numlike,
    lams: numlike,
    lnfx: Callable[[Tensor], numlike],
    lsfx: Callable[[Tensor], numlike],
    dt: numlike,
):
    kis: List[Tensor] = []
    ki: Tensor = khcomm(h, rho) + ns_evolve(rho, rho, lamn, lams, lnfx, lsfx)
    kis.append(ki * rk4weights[0])
    for i in range(len(rk4weights) - 1):
        rhoi: Tensor = evolve_by(rho, dt, ki / rk4weights[i + 1])
        ki: Tensor = khcomm(h, rhoi) + ns_evolve(rhoi, rho, lamn, lams, lnfx, lsfx)
        kis.append(ki * rk4weights[i + 1])

    return rho + (dt / 6) * torch.stack(kis, dim=0).sum(dim=0)


# ──────────────────────────────────────────────────────────────────────────────
def psink_evol(
    rho0: Tensor,
    pee2: Tensor,
    h: Tensor,
    lamn: numlike,
    lams: numlike,
    lnfx: Callable[[Tensor], numlike],
    lsfx: Callable[[Tensor], numlike],
    dt: numlike,
    time: int,
    zfactor: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    psink: Tensor = torch.empty((time * zfactor), device=device)
    rhot: Tensor = rho0
    for t in range(time * zfactor):
        rhot: Tensor = rk4step(rhot, h, lamn, lams, lnfx, lsfx, dt)
        psink[t] = torch.trace(rhot @ pee2).real
    return psink


def driven_psink_evol(
    rho0: Tensor,
    pee2: Tensor,
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
    lamn: numlike,
    lams: numlike,
    lnfx: Callable[[Tensor], numlike],
    lsfx: Callable[[Tensor], numlike],
    dt: numlike,
    time: int,
    zfactor: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    psink: Tensor = torch.empty((time * zfactor), device=device)
    rhot: Tensor = rho0
    for t in range(time * zfactor):
        # Driving
        htot: Tensor = drive_hamiltonian(
            an,
            wn,
            bn,
            aq,
            wq,
            bq,
            mup,
            ee1,
            gg1,
            idr,
            idn,
            idq1,
            idq2,
            hr,
            hrq,
            hqn,
            diagcorr,
            dt,
            t,
        )
        # Evolution as usual
        rhot: Tensor = rk4step(rhot, htot, lamn, lams, lnfx, lsfx, dt)
        psink[t] = torch.trace(rhot @ pee2).real
    return psink


def coupled_psink_evol(
    rho0: Tensor,
    pee2: Tensor,
    lrq: Parameter,
    lqn: Parameter,
    a: Tensor,
    zo: Tensor,
    eg1: Tensor,
    idr: Tensor,
    idn: Tensor,
    idq2: Tensor,
    hn: Tensor,
    hq: Tensor,
    hr: Tensor,
    lamn: numlike,
    lams: numlike,
    lnfx: Callable[[Tensor], numlike],
    lsfx: Callable[[Tensor], numlike],
    dt: numlike,
    time: int,
    zfactor: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    psink: Tensor = torch.empty((time * zfactor), device=device)
    rhot: Tensor = rho0
    htot: Tensor = couple_hamiltonian(lrq, lqn, a, zo, eg1, idn, idr, idq2, hn, hq, hr)
    for t in range(time * zfactor):
        rhot: Tensor = rk4step(rhot, htot, lamn, lams, lnfx, lsfx, dt)
        psink[t] = torch.trace(rhot @ pee2).real
    return psink


def energized_psink_evol(
    rho0: Tensor,
    pee2: Tensor,
    e: Parameter,
    n: int,
    mup: Tensor,
    idr: Tensor,
    idq1: Tensor,
    idq2: Tensor,
    hq: Tensor,
    hr: Tensor,
    hrq: Tensor,
    hqn: Tensor,
    lamn: numlike,
    lams: numlike,
    lnfx: Callable[[Tensor], numlike],
    lsfx: Callable[[Tensor], numlike],
    dt: numlike,
    time: int,
    zfactor: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    psink: Tensor = torch.empty((time * zfactor), device=device)
    rhot: Tensor = rho0
    htot: Tensor = energize_hamiltonian(e, mup, n, idr, idq1, idq2, hq, hr, hrq, hqn, device)
    for t in range(time * zfactor):
        rhot: Tensor = rk4step(rhot, htot, lamn, lams, lnfx, lsfx, dt)
        psink[t] = torch.trace(rhot @ pee2).real
    return psink
