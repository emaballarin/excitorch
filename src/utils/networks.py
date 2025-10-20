#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List
from typing import Optional

import torch
from ebtorch.typing import realnum
from ebtorch.typing import strdev
from safe_assert import safe_assert as sassert
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "mkmfmo",
    "mkmnneigh",
    "mkstar",
    "param_network_checks",
    "mknetwork",
]


# ──────────────────────────────────────────────────────────────────────────────
def mkmfmo(rescaling: realnum = 1 / 100, device: Optional[strdev] = None) -> Tensor:
    """
    Build the Fenna-Matthews-Olson (FMO) complex network interaction Hamiltonian.

    Args:
        rescaling (realnum): Scaling factor for the Hamiltonian matrix.
        device (strdev): Device to use for allocating the Hamiltonian matrix.

    Returns:
        Tensor: The FMO Hamiltonian matrix.
    """
    modiag: Tensor = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -104.1, 5.1, -4.3, 4.7, -15.1, -7.8],
            [0, -104.1, 0, 32.6, 7.1, 5.4, 8.3, 0.8],
            [0, 5.1, 32.6, 0, -46.8, 1.0, -8.1, 5.1],
            [0, -4.3, 7.1, -46.8, 0, -70.7, -14.7, -61.5],
            [0, 4.7, 5.4, 1.0, -70.7, 0, 89.7, -2.5],
            [0, -15.1, 8.3, -8.1, -14.7, 89.7, 0, 32.7],
            [0, -7.8, 0.8, 5.1, -61.5, -2.5, 32.7, 0],
        ],
        dtype=torch.complex64,
        device=device,
    )
    mddiag: Tensor = torch.diag(torch.tensor([0, 65.7, -11.1, -56.1, -36.2, -30.6, 55.7, 4.2], device=device))
    return rescaling * (modiag + mddiag)


# ──────────────────────────────────────────────────────────────────────────────
def mkddaux(n: int, interw: int = 1, device: Optional[strdev] = None) -> Tensor:
    mmat: Tensor = torch.zeros((n, n), dtype=torch.complex64, device=device)

    for i in range(1, interw + 1):
        mmat = mmat + torch.diag(torch.ones(n - i, device=device), i) + torch.diag(torch.ones(n - i, device=device), -i)

    return mmat


def mkmnneigh(
    n: int,
    dscale: realnum = 0.5,
    rescaling: realnum = 1,
    device: Optional[strdev] = None,
) -> Tensor:
    mmat: Tensor = mkddaux(n, 1, device=device)
    mmat[0, 1] = 0
    mmat[1, 0] = 0

    mdiag: Tensor = torch.diag(dscale * torch.ones(n, device=device, dtype=torch.complex64))

    mmm: Tensor = mmat + mdiag
    mmm[0][0] = 0

    return rescaling * mmm


def mkstar(n: int, centre: int, rescaling: realnum = 1, device: Optional[strdev] = None) -> Tensor:
    sassert(0 < centre < n - 1, "Coordinates of the central site must be within (0, n-1)")

    # Diagonal entries
    mdiag: Tensor = torch.diag(0.5 * torch.ones(n, device=device, dtype=torch.complex64))
    mdiag[0, 0] = 0

    # Off-diagonal entries
    modiag: Tensor = torch.zeros((n, n), dtype=torch.complex64, device=device)
    modiag[centre, 1:] = 1
    modiag[1:, centre] = 1
    modiag[centre, centre] = 0
    modiag: Tensor = modiag * (
        torch.tril(torch.ones((n, n), device=device)) + torch.triu(torch.ones((n, n), device=device), 1)
    )

    return rescaling * (mdiag + modiag)


# ──────────────────────────────────────────────────────────────────────────────


def param_network_checks(network: str, nsize: Optional[int]) -> None:
    sassert(
        network in ("FMO", "NN", "star"),
        "`network` must be either `FMO`, `NN`, or `star`",
    )
    sassert(network != "FMO" or nsize is None, "Network size if fixed if `network` is `FMO`")
    sassert(
        network not in ("NN", "star") or nsize is not None,
        "Network size must be provided if `network` is `NN` or `star`",
    )


def mknetwork(network: str, nsize: Optional[int], device: Optional[strdev]) -> Tensor:
    if network == "FMO":
        return mkmfmo(device=device)
    elif network == "NN":
        return mkmnneigh(nsize, device=device)
    else:  # network == "star"
        return mkstar(nsize, 2, device=device)
