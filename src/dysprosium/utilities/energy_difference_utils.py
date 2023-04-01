from __future__ import annotations

import numpy as np  # type: ignore

from dysprosium.utilities.dataclasses import Constant, SystemParameters  # type: ignore


class EnergyDifferenceUtils:
    def __init__(self, system_parameters: SystemParameters, energy_difference):
        # self.atm = atomic_parameters
        self.sys = system_parameters
        self.dE = energy_difference

    def n_abs(self, dE):
        return 1 / (-1 + np.exp(abs(dE) * self.sys.beta))

    def n_emi(self, dE):
        return 1 + (1 / (-1 + np.exp(abs(dE) * self.sys.beta)))

    def n_a(self):
        return 1 / (-1 + np.exp(abs(self.dE) * self.sys.beta))

    def n_b(self):
        return 1 / (-1 + np.exp(abs(self.dE) * self.sys.beta))

    def fermiTS(self, bias):
        return (self.dE - bias) / (np.exp((self.dE - bias) * self.sys.beta) - 1)

    def fermiST(self, bias):
        return (self.dE + bias) / (np.exp((self.dE + bias) * self.sys.beta) - 1)

    def fermiTT(self):
        return (self.dE) / (np.exp((self.dE) * self.sys.beta) - 1)

    def fermiSS(self):
        return (self.dE) / (np.exp((self.dE) * self.sys.beta) - 1)

    def ph_exc(self):
        return (
            ((np.abs(self.dE) ** 2) / (-1 + np.exp(np.abs(self.dE) * self.sys.beta)))
            * (3 * (self.sys.pheff / Constant.Dydensity) * Constant.meVtoJ)
            / (2 * Constant.rho * (Constant.c ** 4) * (Constant.h_bar ** 3))
        )

    def ph_rel(self):
        return (
            (
                (np.abs(self.dE) ** 2)
                * (1 + (1 / (-1 + np.exp(np.abs(self.dE) * self.sys.beta))))
            )
            * (3 * (self.sys.pheff / Constant.Dydensity) * Constant.meVtoJ)
            / (2 * Constant.rho * (Constant.c ** 4) * (Constant.h_bar ** 3))
        )

    def RmT(self, surface_only: bool):
        if surface_only:
            rmt = 0 * self.fermiTS(self.sys.bias)
        else:
            rmt = (
                (2 * np.pi / Constant.h_bar)
                * (self.sys.zeta ** 2)
                * (
                    self.sys.SigmaS
                    * self.sys.SigmaT
                    * self.sys.SmRTS
                    * self.fermiTS(self.sys.bias)
                    + self.sys.SigmaT
                    * self.sys.SigmaT
                    * self.sys.SmRTT
                    * self.fermiTT()
                )
            )
        return rmt.T

    def RpT(self, surface_only: bool):
        if surface_only:
            rpt = 0 * self.fermiTS(self.sys.bias)
        else:
            rpt = (
                (2 * np.pi / Constant.h_bar)
                * (self.sys.zeta ** 2)
                * (
                    self.sys.SigmaS
                    * self.sys.SigmaT
                    * self.sys.SpRTS
                    * self.fermiTS(self.sys.bias)
                    + self.sys.SigmaT
                    * self.sys.SigmaT
                    * self.sys.SpRTT
                    * self.fermiTT()
                )
            )
        return rpt.T

    def RzT(self, surface_only: bool):
        if surface_only:
            rzt = 0 * self.fermiTS(self.sys.bias)
        else:
            rzt = (
                (2 * np.pi / Constant.h_bar)
                * (self.sys.zeta ** 2)
                * (
                    self.sys.SigmaS
                    * self.sys.SigmaT
                    * self.sys.SzRTS
                    * self.fermiTS(self.sys.bias)
                    + self.sys.SigmaT
                    * self.sys.SigmaT
                    * self.sys.SzRTT
                    * self.fermiTT()
                )
            )
        return rzt.T

    def RmS(self, sec_el, surface_only: bool):
        if surface_only:
            rms = (
                (2 * np.pi / Constant.h_bar)
                * (self.sys.zeta ** 2)
                * (
                    self.sys.SigmaS * self.sys.SigmaS * self.sys.SmRSS * self.fermiSS()
                    + (sec_el * self.dE * ((5000 + self.dE) ** -4))
                )
            )
        else:
            rms = (
                (2 * np.pi / Constant.h_bar)
                * (self.sys.zeta ** 2)
                * (
                    self.sys.SigmaS
                    * self.sys.SigmaT
                    * self.sys.SmRST
                    * self.fermiST(self.sys.bias)
                    + self.sys.SigmaS
                    * self.sys.SigmaS
                    * self.sys.SmRSS
                    * self.fermiSS()
                )
            )
        return rms.T

    def RpS(self, sec_el, surface_only: bool):
        if surface_only:
            rps = (
                (2 * np.pi / Constant.h_bar)
                * (self.sys.zeta ** 2)
                * (
                    self.sys.SigmaS * self.sys.SigmaS * self.sys.SpRSS * self.fermiSS()
                    + (sec_el * self.dE * ((5000 + self.dE) ** -4))
                )
            )
        else:
            rps = (
                (2 * np.pi / Constant.h_bar)
                * (self.sys.zeta ** 2)
                * (
                    self.sys.SigmaS
                    * self.sys.SigmaT
                    * self.sys.SpRST
                    * self.fermiST(self.sys.bias)
                    + self.sys.SigmaS
                    * self.sys.SigmaS
                    * self.sys.SpRSS
                    * self.fermiSS()
                )
            )
        return rps.T

    def RzS(self, sec_el, surface_only: bool):
        if surface_only:
            rzs = (
                (2 * np.pi / Constant.h_bar)
                * (self.sys.zeta ** 2)
                * (
                    self.sys.SigmaS * self.sys.SigmaS * self.sys.SzRSS * self.fermiSS()
                    + (sec_el * self.dE * ((5000 + self.dE) ** -4))
                )
            )
        else:
            rzs = (
                (2 * np.pi / Constant.h_bar)
                * (self.sys.zeta ** 2)
                * (
                    self.sys.SigmaS
                    * self.sys.SigmaT
                    * self.sys.SzRST
                    * self.fermiST(self.sys.bias)
                    + self.sys.SigmaS
                    * self.sys.SigmaS
                    * self.sys.SzRSS
                    * self.fermiSS()
                )
            )
        return rzs.T
