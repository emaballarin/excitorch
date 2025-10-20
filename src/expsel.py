#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from functools import partial as fpartial

from runners import run_experiment_fmo_cpl
from runners import run_experiment_fmo_drv
from runners import run_experiment_fmo_erg
from runners import run_experiment_nn_cpl
from runners import run_experiment_nn_drv
from runners import run_experiment_nn_erg
from runners import run_experiment_star_cpl
from runners import run_experiment_star_drv
from runners import run_experiment_star_erg
from utils.experiments import ExpCfg

# ──────────────────────────────────────────────────────────────────────────────
__all__ = ["select_experiment"]


# ──────────────────────────────────────────────────────────────────────────────


def select_experiment(
    expcfg: ExpCfg,
) -> callable:
    # Network: FMO
    if expcfg.network == "FMO":
        argsdict = {
            "freq": expcfg.freq,
            "lamn": expcfg.lamn,
            "timeafter": expcfg.timeafter,
            "timeo": expcfg.timeo,
            "epochs": expcfg.epochs,
            "lr": expcfg.lr,
            "seed": expcfg.seed,
        }

        if expcfg.opt == "driving":
            return fpartial(
                run_experiment_fmo_drv,
                nsines=expcfg.nsines,
                **argsdict,
            )
        elif expcfg.opt == "coupling":
            return fpartial(
                run_experiment_fmo_cpl,
                **argsdict,
            )
        else:  # expcfg.opt == "energy":
            return fpartial(
                run_experiment_fmo_erg,
                **argsdict,
            )

    # Networks: NN or star
    else:
        argsdict = {
            "freq": expcfg.freq,
            "lamn": expcfg.lamn,
            "timeafter": expcfg.timeafter,
            "timeo": expcfg.timeo,
            "epochs": expcfg.epochs,
            "lr": expcfg.lr,
            "seed": expcfg.seed,
            "nsize": expcfg.nsize,
        }

        if expcfg.network == "NN":
            if expcfg.opt == "driving":
                return fpartial(
                    run_experiment_nn_drv,
                    nsines=expcfg.nsines,
                    **argsdict,
                )
            elif expcfg.opt == "coupling":
                return fpartial(
                    run_experiment_nn_cpl,
                    **argsdict,
                )
            else:  # expcfg.opt == "energy":
                return fpartial(
                    run_experiment_nn_erg,
                    **argsdict,
                )

        else:  # expcfg.network == "star"
            if expcfg.opt == "driving":
                return fpartial(
                    run_experiment_star_drv,
                    nsines=expcfg.nsines,
                    **argsdict,
                )
            elif expcfg.opt == "coupling":
                return fpartial(
                    run_experiment_star_cpl,
                    **argsdict,
                )
            else:  # expcfg.opt == "energy":
                return fpartial(
                    run_experiment_star_erg,
                    **argsdict,
                )
