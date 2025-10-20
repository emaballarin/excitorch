#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List
from typing import Tuple
from typing import Union

import torch
from expsel import select_experiment
from utils.experiments import ExpCfg

# ──────────────────────────────────────────────────────────────────────────────
__all__ = ["run_experiments"]


# ──────────────────────────────────────────────────────────────────────────────
def run_experiment(expcfg: ExpCfg):
    # Sanity check
    expcfg.chk()

    exprun = select_experiment(expcfg)

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run experiment
    p_sink_optim, p_sink_unoptim, model = exprun(device=device)

    # Model saving
    torch.save(model.state_dict(), f"{expcfg.namestr}_model_final.pth")

    # Graph points saving
    torch.save(p_sink_optim, f"{expcfg.namestr}_poptim_final.pth")
    torch.save(p_sink_unoptim, f"{expcfg.namestr}_punoptim_final.pth")


def run_experiments(expcfgs: Union[List[ExpCfg], Tuple[ExpCfg, ...], ExpCfg]):
    if isinstance(expcfgs, ExpCfg):
        expcfgs = [expcfgs]
    if len(expcfgs) == 0:
        return
    for expcfg in expcfgs:
        print(f"\n\nRunning experiment: {expcfg.namestr}\n")
        run_experiment(expcfg)
