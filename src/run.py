#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List

from ebtorch.nn.utils import TelegramBotEcho as TBEcho
from orchestrator import run_experiments
from utils.experiments import ExpCfg

# ──────────────────────────────────────────────────────────────────────────────
tbecho = TBEcho("EBSL_TGB_TOKEN", "EBSL_TGB_CHTID")
# ──────────────────────────────────────────────────────────────────────────────
# ─── rng seeding ──────────────────────────────────────────────────────────────
SEED: int = 541
# ─── epochs ───────────────────────────────────────────────────────────────────
EPOCHS: int = 30
# ─── time of optimization ─────────────────────────────────────────────────────
TIMEO: int = 30
# ─── time after optimization ─────────────────────────────────────────────────
TIMEAFTER: int = 1200
# ─── frequencies ──────────────────────────────────────────────────────────────
WRES_GOODGUESS: float = 0.25
WRES_NN: float = 0.2446
WRES_FMO: float = 0.2700
WRES_STAR: float = WRES_GOODGUESS
WOFFRES: float = 15
# ─── dephasing constants ──────────────────────────────────────────────────────
USL_DEPH: float = 0.1
STG_DEPH: float = 1
# ─── learning rates ───────────────────────────────────────────────────────────
LR_NN: float = 0.6
LR_FMO: float = 0.2
# ──────────────────────────────────────────────────────────────────────────────
# noinspection PyListCreation
experiments: List[ExpCfg] = []
# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F1E1",
        network="NN",
        opt="driving",
        freq=WRES_NN,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=1,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=4,
    )
)

experiments.append(
    ExpCfg(
        name="F1E2",
        network="NN",
        opt="coupling",
        freq=WRES_NN,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=None,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED + 1,
        nsize=4,
    )
)

experiments.append(
    ExpCfg(
        name="F1E3",
        network="NN",
        opt="energy",
        freq=WRES_NN,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=None,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=4,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F2E1",
        network="NN",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=1,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=4,
    )
)

experiments.append(
    ExpCfg(
        name="F2E2",
        network="NN",
        opt="coupling",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=None,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED + 1,
        nsize=4,
    )
)

experiments.append(
    ExpCfg(
        name="F2E3",
        network="NN",
        opt="energy",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=None,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=4,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F5E1",
        network="NN",
        opt="driving",
        freq=WOFFRES,
        lamn=STG_DEPH,
        timeafter=TIMEAFTER,
        nsines=1,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=4,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F8E1",
        network="star",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=2,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=8,
    )
)

experiments.append(
    ExpCfg(
        name="F8E2",
        network="star",
        opt="coupling",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=None,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED + 1,
        nsize=8,
    )
)

experiments.append(
    ExpCfg(
        name="F8E3",
        network="star",
        opt="energy",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=None,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=8,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F10E1",
        network="FMO",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=2,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_FMO,
        seed=SEED,
        nsize=None,
    )
)

experiments.append(
    ExpCfg(
        name="F10E2",
        network="FMO",
        opt="coupling",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=None,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_FMO,
        seed=SEED + 1,
        nsize=None,
    )
)

experiments.append(
    ExpCfg(
        name="F10E3",
        network="FMO",
        opt="energy",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=None,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_FMO,
        seed=SEED,
        nsize=None,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F3E2",
        network="NN",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=2,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=4,
    )
)

experiments.append(
    ExpCfg(
        name="F3E3",
        network="NN",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=7,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=4,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F7E2",
        network="star",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=1,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=8,
    )
)

experiments.append(
    ExpCfg(
        name="F7E3",
        network="star",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=7,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=8,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F9E1",
        network="FMO",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=1,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_FMO,
        seed=SEED,
        nsize=None,
    )
)

experiments.append(
    ExpCfg(
        name="F9E3",
        network="FMO",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=7,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_FMO,
        seed=SEED,
        nsize=None,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F4E1",
        network="NN",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=1,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=6,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F6E3",
        network="NN",
        opt="driving",
        freq=WOFFRES,
        lamn=10,
        timeafter=TIMEAFTER,
        nsines=1,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=4,
    )
)

# ──────────────────────────────────────────────────────────────────────────────

experiments.append(
    ExpCfg(
        name="F4E4",
        network="NN",
        opt="driving",
        freq=WOFFRES,
        lamn=USL_DEPH,
        timeafter=TIMEAFTER,
        nsines=1,
        timeo=TIMEO,
        epochs=EPOCHS,
        lr=LR_NN,
        seed=SEED,
        nsize=8,
    )
)


# ──────────────────────────────────────────────────────────────────────────────

tbecho.send("Experiments started...")
run_experiments(experiments)
tbecho.send("Experiments finished!")
# ──────────────────────────────────────────────────────────────────────────────
