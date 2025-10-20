#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
import math
from typing import Optional

from attrs import frozen
from ebtorch.typing import numlike
from ebtorch.typing import realnum


# ──────────────────────────────────────────────────────────────────────────────
@frozen
class SystemConfig:
    # Data
    nsize: int
    lamn: numlike
    lams: numlike
    freq: numlike
    betacoeff: numlike

    @property
    def nmax(self) -> int:
        return 3 * (int(1 / (math.exp(self.freq * self.betacoeff)) - 1) + 1)


# ──────────────────────────────────────────────────────────────────────────────
@frozen
class ProblemConfig:
    # Data
    timeafter: numlike


# ──────────────────────────────────────────────────────────────────────────────
@frozen
class OptimConfig:
    # Data
    dt: numlike
    timeo: numlike
    lr: realnum
    niter: int
    nsines: Optional[int] = None

    @property
    def zfactor(self, integer: bool = True) -> numlike:
        return int(1 / self.dt) if integer else 1 / self.dt


# ──────────────────────────────────────────────────────────────────────────────
