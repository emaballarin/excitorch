#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
from typing import List
from typing import Optional

from attrs import frozen
from ebtorch.typing import realnum
from safe_assert import safe_assert as sassert

# ──────────────────────────────────────────────────────────────────────────────
__all__: List[str] = ["ExpCfg"]


# ──────────────────────────────────────────────────────────────────────────────
@frozen
class ExpCfg:
    name: str
    network: str
    opt: str
    freq: realnum
    lamn: realnum
    timeafter: realnum
    nsines: Optional[int]
    timeo: realnum
    epochs: int
    lr: realnum
    seed: int
    nsize: Optional[int]

    def _chkstr(self) -> None:
        sassert(
            self.network in ("FMO", "NN", "star"),
            "`network` must be either `FMO`, `NN`, or `star`",
        )
        sassert(
            self.opt in ("driving", "coupling", "energy"),
            "`opt` must be either `driving`, `coupling` or `energy`",
        )

    def _chkparams(self):
        sassert(self.freq > 0, "`freq` must be greater than 0")
        sassert(self.lamn > 0, "`lamn` must be greater than 0")
        sassert(self.timeafter > 0, "`timeafter` must be greater than 0")
        sassert(self.timeo > 0, "`timeo` must be greater than 0")
        sassert(self.epochs > 0, "`epochs` must be greater than 0")
        sassert(self.lr > 0, "`lr` must be greater than 0")
        sassert(self.seed > 0, "`seed` must be greater than 0")
        sassert(self.nsines is None or self.nsines > 0, "`nsines` must be greater than 0")
        sassert(self.nsize is None or self.nsize > 0, "`nsize` must be greater than 0")
        sassert(
            (self.network != "FMO") or (self.nsize is None),
            "`nsize` must not be set for `network` `FMO`",
        )
        sassert(
            (self.opt == "driving") or (self.nsines is None),
            "`nsines` must not be set when `opt` is `coupling` or `energy`",
        )

    def chk(self):
        self._chkstr()
        self._chkparams()

    @property
    def namestr(self) -> str:
        return f"{self.name}_{self.network}_{self.opt}"
