#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from ebtorch.typing import numlike
from ebtorch.typing import realnum
from safe_assert import safe_assert as sassert
from torch import Tensor

from .extra import stablediv


# ──────────────────────────────────────────────────────────────────────────────
def mkplots(
    pso: Tensor,
    psu: Tensor,
    eps: numlike,
    save: bool = False,
    exp_name: str = "experiment",
    x_rescale: realnum = 1,
    xr_int: bool = False,
) -> None:
    # Preparations
    pso = pso.cpu()
    psu = psu.cpu()
    plt.style.use("ggplot")
    # Constants
    timestr: str = "Time after optimization (s)"
    probstr: str = "Probability of hitting the sink"
    ratiostr: str = "Figure of merit (optimized/non-optimized)"
    drvstr: str = "Optimized system"
    nodrvstr: str = "Non-optimized system"
    # Compute rescaling factors
    sassert(x_rescale > 0, "`x_rescale` must be positive")
    sassert(pso.size() == psu.size(), "`pso` and `psu` tensors must have the same size")
    xlen: int = pso.size(0)
    rxstop: realnum = x_rescale * xlen if not xr_int else int(x_rescale * xlen)
    xax: np.ndarray = np.linspace(0, rxstop, xlen)
    # Parallel evolution
    plt.figure()
    plt.plot(xax, pso, label=drvstr)
    plt.plot(xax, psu, label=nodrvstr)
    plt.xlabel(timestr)
    plt.ylabel(probstr)
    plt.legend()
    if save:
        plt.savefig(f"{exp_name}_evolution.png")
    else:
        plt.show()
    # Figure of merit
    plt.figure()
    plt.plot(xax, stablediv(pso, psu, eps, True), label="FoM")
    plt.xlabel(timestr)
    plt.ylabel(ratiostr)
    plt.legend()
    if save:
        plt.savefig(f"{exp_name}_fom.png")
    else:
        plt.show()


def mkplot(
    curvepaths: Tuple[str, ...],
    curvenames: Tuple[str, ...],
    ratios: bool = False,
    title: str = "Figure",
    figname: str = "experiment",
    xstr: str = "x",
    ystr: str = "y",
    save: bool = False,
    x_rescale: realnum = 1,
    xr_int: bool = False,
    colors: Optional[Tuple[str, ...]] = None,
    fmt: Optional[Tuple[str, ...]] = None,
    nolegend: bool = False,
    inset: Tuple[Optional[int], Optional[int]] = (None, None),
) -> None:
    if colors is not None:
        colorlen: int = len(colors)
        cidx: int = -1

    if fmt is not None:
        fmtlen: int = len(fmt)
        fidx: int = -1

    # Styling ->
    plt.rcParams["text.usetex"] = True
    plt.style.use("ggplot")
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams["grid.color"] = "gainsboro"
    plt.rcParams["font.size"] = plt.rcParams["font.size"] * 2
    plt.rcParams["axes.formatter.useoffset"] = False
    # <- End styling

    plt.figure()
    clen: Optional[int] = None

    savec: Optional[Tensor] = None
    for c, n in zip(
        curvepaths,
        curvenames,
    ):
        # Load the curve
        curve: Tensor = torch.load(c, map_location="cpu", weights_only=False).cpu()
        curve: Tensor = torch.clamp(curve, 0, 1)
        # Check curve length
        if clen is None:
            clen = curve.size(0)
        else:
            sassert(clen == curve.size(0), "All curves must have the same length")
        # Rescale the x-axis
        xlen: int = curve.size(0)
        rxstop: realnum = x_rescale * xlen if not xr_int else int(x_rescale * xlen)
        xax: np.ndarray = np.linspace(0, rxstop, xlen)

        # Case: Ratios
        if ratios and (savec is None or "unoptim" in c):
            savec = curve
            continue
        elif ratios:
            curve = stablediv(curve, savec, 0.05, True)

        # Plot
        # noinspection PyUnboundLocalVariable
        plt.plot(
            xax[inset[0] : inset[-1]],
            curve[inset[0] : inset[-1]],
            label=n,
            color=(
                colors[(cidx := (cidx + 1)) % colorlen]  # NOSONAR
                if colors is not None
                else None
            ),
            linestyle=(fmt[(fidx := (fidx + 1)) % fmtlen] if fmt is not None else None),
        )

        # Labels
        if len(title) > 0:
            plt.title(title)
        if len(xstr) > 0:
            plt.xlabel(xstr)
        if len(ystr) > 0:
            plt.ylabel(ystr)

        if not nolegend:
            plt.legend()

    # Save or show
    if save:
        plt.savefig(f"graphs/png/{figname}.png", dpi=400, bbox_inches="tight")
    else:
        plt.show()


def easy_mkplot(
    curvepaths: Tuple[str, ...],
    curvenames: Tuple[str, ...],
    ratios: bool,
    title: str,
    figname: str,
    ystr: str,
    save: bool,
    colors: Optional[Tuple[str, ...]] = None,
    fmt: Optional[Tuple[str, ...]] = None,
    nolegend: bool = False,
    inset: Tuple[Optional[int], Optional[int]] = (None, None),
) -> None:
    mkplot(
        curvepaths,
        curvenames,
        ratios=ratios,
        title=title,
        figname=figname,
        xstr="Time after optimization (s)",
        ystr=ystr,
        save=save,
        x_rescale=0.01,
        xr_int=True,
        colors=colors,
        fmt=fmt,
        nolegend=nolegend,
        inset=inset,
    )
