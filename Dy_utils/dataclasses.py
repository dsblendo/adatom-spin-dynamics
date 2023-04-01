from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from parse import compile


@dataclass
class Constant:
    """Convenience class for holding global constants."""

    mu_b: float = 0.05788  # meV/K
    k_b: float = 0.08617  # meV/K
    h_bar: float = 6.58211928 * (10 ** -13)  # meV-S

    # Couplings
    A_sf: float = 3.1
    A_df: float = 7.3
    A_sd: float = 500
    F: float = 0.0006
    g: float = 0.54
    g_z: float = 1.25

    # phonons
    c: float = 2.1 * (10 ** 4)
    rho: float = 7.7 * (10 ** -8)
    meVtoJ: float = 1.6 * (10 ** -22)

    e: float = 1.6 * (10 ** -19)
    Dydensity: float = 0.01
    phi_o: float = 3 * (10 ** -3)  # photons nm−2 s−1


@dataclass
class AtomicParameters:
    """Atomic parameters."""

    v_s: float = 1 / 2
    v_d: float = 1 / 2
    s: float = 8
    n_s: float = 0


@dataclass
class SystemParameters:
    """System parameters."""

    T: float = 6.5

    bias: float = 1
    I_o: float = 18 * (10 ** -12)

    tipu: float = 0.1
    tipd: float = 0.9
    surfu: float = 0.5
    surfd: float = 0.5

    B_x: float = 0.0
    B_z: float = 0.0

    zeta: float = 0.65
    SigmaS: float = 6.5 * (10 ** -2)
    pheff: float = 2 * (10 ** -4)
    B_20: float = -5.18416895e-01
    B_40: float = 1.04712807e-03
    B_60: float = -4.75920969e-07
    B_63: float = 0
    B_66: float = -0.0000075

    beta: float = field(init=False)
    tippol: float = field(init=False)
    surfpol: float = field(init=False)
    SigmaT: float = field(init=False)
    SpRTS: float = field(init=False)
    SmRTS: float = field(init=False)
    SzRTS: float = field(init=False)
    SpRTT: float = field(init=False)
    SmRTT: float = field(init=False)
    SzRTT: float = field(init=False)
    SmRST: float = field(init=False)
    SpRST: float = field(init=False)
    SzRST: float = field(init=False)
    SpRSS: float = field(init=False)
    SmRSS: float = field(init=False)
    SzRSS: float = field(init=False)

    def __post_init__(self):
        self.beta = 1 / (self.T * Constant.k_b)

        self.tippol = (self.tipu - self.tipd) / (self.tipu + self.tipd)
        self.surfpol = (self.surfu - self.surfd) / (self.surfu + self.surfd)
        self.SigmaT = abs(
            (4 * Constant.h_bar * self.I_o)
            / (
                2
                * np.pi
                * Constant.e
                * (self.fermi(self.bias) - self.fermi(-self.bias))
                * self.SigmaS
                * (self.surfu * self.tipu + self.surfd * self.tipd)
            )
        )
        self.SpRTS = self.tipu * self.surfd
        self.SmRTS = self.tipd * self.surfu
        self.SzRTS = self.surfu * self.tipu + self.surfd * self.tipd

        self.SpRTT = self.tipd * self.tipu
        self.SmRTT = self.tipu * self.tipd
        self.SzRTT = self.tipu * self.tipu + self.tipd * self.tipd

        self.SmRST = self.surfd * self.tipu
        self.SpRST = self.surfu * self.tipd
        self.SzRST = self.surfu * self.tipu + self.surfd * self.tipd

        self.SpRSS = self.surfd * self.surfu
        self.SmRSS = self.surfu * self.surfd
        self.SzRSS = self.surfu * self.surfu + self.surfd * self.surfd

    def fermi(self, bias):
        return -bias / (np.exp(-bias * self.beta) - 1)


@dataclass
class LevelCrossing:
    """Level crossing"""

    B_z: float
    E: float  # meV
    lower_eig: str
    upper_eig: str
    level_splitting: float
    lower_ind_values: int
    upper_ind_values: int
    J_z_lower: float | None = None
    J_z_upper: float | None = None
    Jz_dif: float | None = None
    Ns_z_lower: float | None = None
    Ns_z_upper: float | None = None
    rel_time: float | None = None

    @classmethod
    def load_csv(cls, directory: Path, filename: str) -> Iterator[LevelCrossing]:
        path = directory / filename
        with open(path, "r") as file:
            next(file)
            for line in file:
                data = line.strip().split(",")
                level_crossing = LevelCrossing(
                    B_z=float(data[1]),
                    E=float(data[2]),
                    lower_eig=data[3],
                    upper_eig=data[4],
                    level_splitting=float(data[5]),
                    lower_ind_values=int(data[6]),
                    upper_ind_values=int(data[7]),
                )
                yield level_crossing

    @staticmethod
    def parse_single_num(string_value: str) -> float:
        format_splitting = compile("|{num} >")
        parse_result = format_splitting.parse(string_value)
        return float(parse_result["num"])

    @staticmethod
    def parse_double_num(string_value: str) -> tuple[float, float]:
        format_splitting = compile("|{num1}, , {num2}>")
        parse_result = format_splitting.parse(string_value)
        return (float(parse_result["num1"]), float(parse_result["num2"]))

    def populate_attr(self, n_s, sweep_rate):
        if n_s == 5 / 2:
            self.J_z_lower, self.Ns_z_lower = self.parse_double_num(self.lower_eig)
            self.J_z_upper, self.Ns_z_upper = self.parse_double_num(self.lower_eig)
        else:
            self.J_z_lower = self.parse_single_num(self.lower_eig)
            self.J_z_upper = self.parse_single_num(self.upper_eig)
        self.Jz_dif = abs(self.J_z_lower) + abs(self.J_z_upper)
        self.rel_time = (
            self.level_splitting / self.Jz_dif * Constant.mu_b
        ) / sweep_rate
