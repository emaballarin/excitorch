#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from functools import partial as fpartial
from typing import List
from typing import Optional
from typing import Tuple

import torch
from ebtorch.typing import numlike
from ebtorch.typing import realnum
from ebtorch.typing import strdev
from torch import Tensor
from tqdm.auto import tqdm
from tqdm.auto import trange
from utils.config import OptimConfig
from utils.config import ProblemConfig
from utils.config import SystemConfig
from utils.extra import couple_hamiltonian
from utils.extra import dimmax
from utils.extra import drive_hamiltonian
from utils.extra import energize_hamiltonian
from utils.extra import kron4chain
from utils.extra import ln_default
from utils.extra import ls_default
from utils.extra import mkada
from utils.networks import mknetwork
from utils.networks import param_network_checks
from utils.proj import mkallp_2q
from utils.proj import mkids
from utils.proj import mknn
from utils.rkevol import coupled_psink_evol
from utils.rkevol import driven_psink_evol
from utils.rkevol import energized_psink_evol
from utils.rkevol import psink_evol
from utils.rkevol import rk4step

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = [
    "run_experiment_fmo_drv",
    "run_experiment_nn_drv",
    "run_experiment_star_drv",
    "run_experiment_fmo_cpl",
    "run_experiment_nn_cpl",
    "run_experiment_star_cpl",
    "run_experiment_fmo_erg",
    "run_experiment_nn_erg",
    "run_experiment_star_erg",
]


# ──────────────────────────────────────────────────────────────────────────────
def _run_experiment_driving(
    network: str,
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    nsines: int,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    device: strdev,
    nsize: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    param_network_checks(network, nsize)

    # RNG seeding
    _ = torch.manual_seed(seed)

    # Experiment configuration
    syscfg: SystemConfig = SystemConfig(
        nsize=(8 if network == "FMO" else nsize),
        lamn=lamn,
        lams=0.05,
        freq=freq,
        betacoeff=1,
    )
    prbcfg: ProblemConfig = ProblemConfig(timeafter=timeafter)
    optcfg: OptimConfig = OptimConfig(dt=0.01, timeo=timeo, lr=lr, niter=epochs, nsines=nsines)

    # Network matrix
    # noinspection DuplicatedCode
    m: Tensor = mknetwork(network, nsize, device)

    # Projectors
    ee1, ee2, gg1, gg2, eg1, eg2, _, ge2, zo, _, zz, nn, zn, nz, _, _, _ = mkallp_2q(syscfg.nsize, device=device)

    # Identity matrices
    idr, idq1, idq2, idn = mkids(syscfg.nsize, syscfg.nmax, device=device)

    # Additional precomputable matrices/tensors
    pee2: Tensor = kron4chain(idr, idq1, idn, ee2, order="BCDA")
    rho_0: Tensor = mknn(syscfg.nmax + 1, device=device)
    rho_ev_0: Tensor = kron4chain(rho_0, gg1, zz, gg2, order="BCAD")
    ad, a = mkada(syscfg.nmax + 1, device=device)

    # Functions
    ln = fpartial(
        ln_default,
        idr=idr,
        idq1=idq1,
        idq2=idq2,
        d=dimmax(syscfg.nmax, syscfg.nsize),
        n=syscfg.nsize,
        device=device,
    )
    ls = fpartial(ls_default, idr=idr, idq1=idq1, zn=zn, nz=nz, eg2=eg2, ge2=ge2, gg2=gg2, nn=nn)

    # Hamiltonians
    hr_0: Tensor = syscfg.freq * (ad @ a)
    hr: Tensor = kron4chain(hr_0, idq1, idn, idq2, order="BCAD")
    hn: Tensor = kron4chain(idr, idq1, m, idq2, order="BCAD")
    hq: Tensor = kron4chain(idr, (ee1 - gg1), idn, idq2, order="BCAD")

    _hrq: Tensor = kron4chain(a, eg1, idn, idq2, order="ABCD")
    hrq: Tensor = _hrq + _hrq.conj().T

    _hqn: Tensor = kron4chain(idr, eg1, zo, idq2, order="BCAD")
    hqn: Tensor = _hqn + _hqn.conj().T

    htot: Tensor = hn + hq + hr + hrq + hqn

    # Unoptimized evolution
    with torch.no_grad():
        p_sink_unoptim: Tensor = psink_evol(
            rho_ev_0,
            pee2,
            htot,
            syscfg.lamn,
            syscfg.lams,
            ln,
            ls,
            optcfg.dt,
            prbcfg.timeafter,
            optcfg.zfactor,
            device,
        )

    # Optimization machinery
    class SineDriver(torch.nn.Module):
        def __init__(self, ncomp: int, anscale: float, diagcorr: realnum) -> None:
            super().__init__()
            self.aq: torch.nn.Parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
            self.bq: torch.nn.Parameter = torch.nn.Parameter(torch.rand(ncomp, dtype=torch.float32), requires_grad=True)
            self.wq: torch.nn.Parameter = torch.nn.Parameter(torch.rand(ncomp, dtype=torch.float32), requires_grad=True)
            self.an: torch.nn.Parameter = torch.nn.Parameter(
                anscale * torch.rand(ncomp, dtype=torch.float32), requires_grad=True
            )
            self.bn: torch.nn.Parameter = torch.nn.Parameter(torch.rand(ncomp, dtype=torch.float32), requires_grad=True)
            self.wn: torch.nn.Parameter = torch.nn.Parameter(torch.rand(ncomp, dtype=torch.float32), requires_grad=True)
            self.diagcorr: realnum = diagcorr

        def forward(self):
            # Preparations
            rho: Tensor = rho_ev_0.clone()
            psink: numlike = 0
            mup: Tensor = m.clone()

            # Evolution
            for j in range(optcfg.timeo * optcfg.zfactor):
                htoti: Tensor = drive_hamiltonian(
                    self.an,
                    self.wn,
                    self.bn,
                    self.aq,
                    self.wq,
                    self.bq,
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
                    self.diagcorr,
                    optcfg.dt,
                    j,
                )
                rho: Tensor = rk4step(rho, htoti, syscfg.lamn, syscfg.lams, ln, ls, optcfg.dt)
                psink: Tensor = psink + torch.trace(pee2 @ rho)

            return -psink.real

    # Optimization
    model: SineDriver = SineDriver(
        optcfg.nsines,
        (0.001 if network == "FMO" else 1),
        0,
    ).to(device)

    # noinspection DuplicatedCode
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=optcfg.lr, weight_decay=0)

    # Training loop
    for i in trange(optcfg.niter):
        optimizer.zero_grad()
        loss: Tensor = model()
        loss.backward()
        optimizer.step()
        if verbose:
            tqdm.write(f"Iteration {i + 1}/{optcfg.niter}: Loss = {loss.item():.3f}")

    # Optimized evolution
    with torch.no_grad():
        p_sink_optim: Tensor = driven_psink_evol(
            rho_ev_0,
            pee2,
            model.an.clone().detach(),
            model.wn.clone().detach(),
            model.bn.clone().detach(),
            model.aq.clone().detach(),
            model.wq.clone().detach(),
            model.bq.clone().detach(),
            m,
            ee1,
            gg1,
            idr,
            idn,
            idq1,
            idq2,
            hr,
            hrq,
            hqn,
            model.diagcorr,
            syscfg.lamn,
            syscfg.lams,
            ln,
            ls,
            optcfg.dt,
            prbcfg.timeafter,
            optcfg.zfactor,
            device,
        )

    return p_sink_optim, p_sink_unoptim, model


# ──────────────────────────────────────────────────────────────────────────────
def _run_experiment_coupling(
    network: str,
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    device: strdev,
    nsize: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    # noinspection DuplicatedCode
    param_network_checks(network, nsize)

    # RNG seeding

    _ = torch.manual_seed(seed)

    # Experiment configuration
    syscfg: SystemConfig = SystemConfig(
        nsize=(8 if network == "FMO" else nsize),
        lamn=lamn,
        lams=0.05,
        freq=freq,
        betacoeff=1,
    )
    prbcfg: ProblemConfig = ProblemConfig(timeafter=timeafter)
    optcfg: OptimConfig = OptimConfig(dt=0.01, timeo=timeo, lr=lr, niter=epochs, nsines=0)

    # Network matrix
    m: Tensor = mknetwork(network, nsize, device)

    # Projectors
    ee1, ee2, gg1, gg2, eg1, eg2, _, ge2, zo, _, zz, nn, zn, nz, _, _, _ = mkallp_2q(syscfg.nsize, device=device)

    # Identity matrices
    idr, idq1, idq2, idn = mkids(syscfg.nsize, syscfg.nmax, device=device)

    # Additional precomputable matrices/tensors
    pee2: Tensor = kron4chain(idr, idq1, idn, ee2, order="BCDA")
    rho_0: Tensor = mknn(syscfg.nmax + 1, device=device)
    rho_ev_0: Tensor = kron4chain(rho_0, gg1, zz, gg2, order="BCAD")
    ad, a = mkada(syscfg.nmax + 1, device=device)

    # Functions
    ln = fpartial(
        ln_default,
        idr=idr,
        idq1=idq1,
        idq2=idq2,
        d=dimmax(syscfg.nmax, syscfg.nsize),
        n=syscfg.nsize,
        device=device,
    )
    ls = fpartial(ls_default, idr=idr, idq1=idq1, zn=zn, nz=nz, eg2=eg2, ge2=ge2, gg2=gg2, nn=nn)

    # Hamiltonians
    hr_0: Tensor = syscfg.freq * (ad @ a)
    hr: Tensor = kron4chain(hr_0, idq1, idn, idq2, order="BCAD")
    hn: Tensor = kron4chain(idr, idq1, m, idq2, order="BCAD")
    hq: Tensor = kron4chain(idr, (ee1 - gg1), idn, idq2, order="BCAD")

    _hrq: Tensor = kron4chain(a, eg1, idn, idq2, order="ABCD")
    hrq: Tensor = _hrq + _hrq.conj().T

    _hqn: Tensor = kron4chain(idr, eg1, zo, idq2, order="BCAD")
    hqn: Tensor = _hqn + _hqn.conj().T

    htot: Tensor = hn + hq + hr + hrq + hqn

    # Unoptimized evolution
    with torch.no_grad():
        p_sink_unoptim: Tensor = psink_evol(
            rho_ev_0,
            pee2,
            htot,
            syscfg.lamn,
            syscfg.lams,
            ln,
            ls,
            optcfg.dt,
            prbcfg.timeafter,
            optcfg.zfactor,
            device,
        )

    # Optimization machinery
    class CouplingEstimator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lrq: torch.nn.Parameter = torch.nn.Parameter(torch.rand(1, dtype=torch.complex64), requires_grad=True)
            self.lqn: torch.nn.Parameter = torch.nn.Parameter(torch.rand(1, dtype=torch.complex64), requires_grad=True)

        def forward(self):
            # Preparations
            rho: Tensor = rho_ev_0.clone()
            psink: numlike = 0
            htoti: Tensor = couple_hamiltonian(self.lrq, self.lqn, a, zo, eg1, idn, idr, idq2, hn, hq, hr)

            # Evolution
            for _ in range(optcfg.timeo * optcfg.zfactor):
                rho: Tensor = rk4step(rho, htoti, syscfg.lamn, syscfg.lams, ln, ls, optcfg.dt)
                psink: Tensor = psink + torch.trace(pee2 @ rho)

            return -psink.real

    # Optimization
    model: CouplingEstimator = CouplingEstimator().to(device)

    # noinspection DuplicatedCode
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=optcfg.lr, weight_decay=0)

    # Training loop
    for i in trange(optcfg.niter):
        optimizer.zero_grad()
        loss: Tensor = model()
        loss.backward()
        optimizer.step()
        if verbose:
            tqdm.write(f"Iteration {i + 1}/{optcfg.niter}: Loss = {loss.item():.3f}")

    # Optimized evolution
    with torch.no_grad():
        p_sink_optim: Tensor = coupled_psink_evol(
            rho_ev_0,
            pee2,
            model.lrq.clone().detach(),
            model.lqn.clone().detach(),
            a,
            zo,
            eg1,
            idr,
            idn,
            idq2,
            hn,
            hq,
            hr,
            syscfg.lamn,
            syscfg.lams,
            ln,
            ls,
            optcfg.dt,
            prbcfg.timeafter,
            optcfg.zfactor,
            device,
        )

    return p_sink_optim, p_sink_unoptim, model


# ──────────────────────────────────────────────────────────────────────────────
def _run_experiment_energy(
    network: str,
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    device: strdev,
    nsize: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    # noinspection DuplicatedCode
    param_network_checks(network, nsize)

    # RNG seeding

    _ = torch.manual_seed(seed)

    # Experiment configuration
    syscfg: SystemConfig = SystemConfig(
        nsize=(8 if network == "FMO" else nsize),
        lamn=lamn,
        lams=0.05,
        freq=freq,
        betacoeff=1,
    )
    prbcfg: ProblemConfig = ProblemConfig(timeafter=timeafter)
    optcfg: OptimConfig = OptimConfig(dt=0.01, timeo=timeo, lr=lr, niter=epochs, nsines=0)

    # Network matrix
    m: Tensor = mknetwork(network, nsize, device)

    # Projectors
    ee1, ee2, gg1, gg2, eg1, eg2, _, ge2, zo, _, zz, nn, zn, nz, _, _, _ = mkallp_2q(syscfg.nsize, device=device)

    # Identity matrices
    idr, idq1, idq2, idn = mkids(syscfg.nsize, syscfg.nmax, device=device)

    # Additional precomputable matrices/tensors
    pee2: Tensor = kron4chain(idr, idq1, idn, ee2, order="BCDA")
    rho_0: Tensor = mknn(syscfg.nmax + 1, device=device)
    rho_ev_0: Tensor = kron4chain(rho_0, gg1, zz, gg2, order="BCAD")
    ad, a = mkada(syscfg.nmax + 1, device=device)

    # Functions
    ln = fpartial(
        ln_default,
        idr=idr,
        idq1=idq1,
        idq2=idq2,
        d=dimmax(syscfg.nmax, syscfg.nsize),
        n=syscfg.nsize,
        device=device,
    )
    ls = fpartial(ls_default, idr=idr, idq1=idq1, zn=zn, nz=nz, eg2=eg2, ge2=ge2, gg2=gg2, nn=nn)

    # Hamiltonians
    hr_0: Tensor = syscfg.freq * (ad @ a)
    hr: Tensor = kron4chain(hr_0, idq1, idn, idq2, order="BCAD")
    hn: Tensor = kron4chain(idr, idq1, m, idq2, order="BCAD")
    hq: Tensor = kron4chain(idr, (ee1 - gg1), idn, idq2, order="BCAD")

    _hrq: Tensor = kron4chain(a, eg1, idn, idq2, order="ABCD")
    hrq: Tensor = _hrq + _hrq.conj().T

    _hqn: Tensor = kron4chain(idr, eg1, zo, idq2, order="BCAD")
    hqn: Tensor = _hqn + _hqn.conj().T

    htot: Tensor = hn + hq + hr + hrq + hqn

    # Unoptimized evolution
    with torch.no_grad():
        p_sink_unoptim: Tensor = psink_evol(
            rho_ev_0,
            pee2,
            htot,
            syscfg.lamn,
            syscfg.lams,
            ln,
            ls,
            optcfg.dt,
            prbcfg.timeafter,
            optcfg.zfactor,
            device,
        )

    # Optimization machinery
    class EnergyEstimator(torch.nn.Module):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.n = n
            self.e: torch.nn.Parameter = torch.nn.Parameter(torch.rand(n, dtype=torch.float32), requires_grad=True)

        def forward(self):
            # Preparations
            rho: Tensor = rho_ev_0.clone()
            psink = 0
            mup = m.clone()
            htoti = energize_hamiltonian(self.e, mup, self.n, idr, idq1, idq2, hq, hr, hrq, hqn, self.e.device)

            # Evolution
            for _ in range(optcfg.timeo * optcfg.zfactor):
                rho = rk4step(rho, htoti, syscfg.lamn, syscfg.lams, ln, ls, optcfg.dt)
                psink = psink + torch.trace(pee2 @ rho)

            return -psink.real

    # Optimization
    model: EnergyEstimator = EnergyEstimator(syscfg.nsize).to(device)

    # noinspection DuplicatedCode
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=optcfg.lr, weight_decay=0)

    # Training loop
    for i in trange(optcfg.niter):
        optimizer.zero_grad()
        loss: Tensor = model()
        loss.backward()
        optimizer.step()
        if verbose:
            tqdm.write(f"Iteration {i + 1}/{optcfg.niter}: Loss = {loss.item():.3f}")

    # Optimized evolution
    with torch.no_grad():
        p_sink_optim: Tensor = energized_psink_evol(
            rho_ev_0,
            pee2,
            model.e.clone().detach(),
            model.n,
            m,
            idr,
            idq1,
            idq2,
            hq,
            hr,
            hrq,
            hqn,
            syscfg.lamn,
            syscfg.lams,
            ln,
            ls,
            optcfg.dt,
            prbcfg.timeafter,
            optcfg.zfactor,
            device,
        )

    return p_sink_optim, p_sink_unoptim, model


# ──────────────────────────────────────────────────────────────────────────────
def run_experiment_fmo_drv(
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    nsines: int,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    device: strdev,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    return _run_experiment_driving(
        "FMO",
        freq,
        lamn,
        timeafter,
        nsines,
        timeo,
        epochs,
        lr,
        seed,
        device,
        verbose=verbose,
    )


def run_experiment_nn_drv(
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    nsines: int,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    nsize: int,
    device: strdev,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    return _run_experiment_driving(
        "NN",
        freq,
        lamn,
        timeafter,
        nsines,
        timeo,
        epochs,
        lr,
        seed,
        device,
        nsize=nsize,
        verbose=verbose,
    )


def run_experiment_star_drv(
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    nsines: int,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    nsize: int,
    device: strdev,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    return _run_experiment_driving(
        "star",
        freq,
        lamn,
        timeafter,
        nsines,
        timeo,
        epochs,
        lr,
        seed,
        device,
        nsize=nsize,
        verbose=verbose,
    )


# noinspection DuplicatedCode
def run_experiment_fmo_cpl(
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    device: strdev,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    return _run_experiment_coupling(
        "FMO",
        freq,
        lamn,
        timeafter,
        timeo,
        epochs,
        lr,
        seed,
        device,
        verbose=verbose,
    )


def run_experiment_nn_cpl(
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    nsize: int,
    device: strdev,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    return _run_experiment_coupling(
        "NN",
        freq,
        lamn,
        timeafter,
        timeo,
        epochs,
        lr,
        seed,
        device,
        nsize=nsize,
        verbose=verbose,
    )


def run_experiment_star_cpl(
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    nsize: int,
    device: strdev,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    return _run_experiment_coupling(
        "star",
        freq,
        lamn,
        timeafter,
        timeo,
        epochs,
        lr,
        seed,
        device,
        nsize=nsize,
        verbose=verbose,
    )


# noinspection DuplicatedCode
def run_experiment_fmo_erg(
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    device: strdev,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    return _run_experiment_energy(
        "FMO",
        freq,
        lamn,
        timeafter,
        timeo,
        epochs,
        lr,
        seed,
        device,
        verbose=verbose,
    )


def run_experiment_nn_erg(
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    nsize: int,
    device: strdev,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    return _run_experiment_energy(
        "NN",
        freq,
        lamn,
        timeafter,
        timeo,
        epochs,
        lr,
        seed,
        device,
        nsize=nsize,
        verbose=verbose,
    )


def run_experiment_star_erg(
    freq: realnum,
    lamn: realnum,
    timeafter: realnum,
    timeo: realnum,
    epochs: int,
    lr: realnum,
    seed: int,
    nsize: int,
    device: strdev,
    verbose: bool = False,
) -> Tuple[Tensor, Tensor, torch.nn.Module]:
    return _run_experiment_energy(
        "star",
        freq,
        lamn,
        timeafter,
        timeo,
        epochs,
        lr,
        seed,
        device,
        nsize=nsize,
        verbose=verbose,
    )


# ──────────────────────────────────────────────────────────────────────────────
