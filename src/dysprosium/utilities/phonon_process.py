from __future__ import annotations

import numpy as np  # type: ignore

from dysprosium.utilities.dataclasses import Constant  # type: ignore
from dysprosium.utilities.energy_difference_utils import EnergyDifferenceUtils  # type: ignore
from dysprosium.utilities.operators import Operators  # type: ignore


class PhononProcess(Operators):
    def get_idx(self):
        if self.sys.B_z > 0:
            idx = (
                self.extract_states(B_x=self.sys.B_x, B_z=-1, rescale=True)
                .sort_values("4f_z")
                .index.values
                - 1
            )
        elif self.sys.B_z < 0:
            idx = (
                self.extract_states(B_x=self.sys.B_x, B_z=1, rescale=True)
                .sort_values("4f_z")
                .index.values
                - 1
            )
        else:
            idx = [3, 1, 6, 7, 8, 5, 2, 4]
        return idx

    def get_energies(self, idx):
        states_df = (
            self.extract_states(self.sys.B_x, self.sys.B_z, rescale=True)
            .reset_index()
            .drop(columns={"index"})
            .loc[idx]
            .iloc[::-1, :]
            .reset_index()
            .drop(columns={"index"})
        )
        energies = states_df["E (meV)"].to_numpy()
        return energies

    def bose_dist(self, energies, energy_difference):
        energy_diff_fcns = EnergyDifferenceUtils(self.sys, energy_difference)
        bose_dist = np.subtract.outer(energies, energies)
        bose_dist[bose_dist > 0] = energy_diff_fcns.n_abs(bose_dist[bose_dist > 0])
        bose_dist[bose_dist < 0] = energy_diff_fcns.n_emi(bose_dist[bose_dist < 0])
        return bose_dist

    def method_Direct(self, energy_diff, resh, dos, states, bose_dist):
        rat_mat = np.zeros(energy_diff.shape)
        for x in zip(
            np.indices(energy_diff.shape).reshape(2, resh).tolist()[0],
            np.indices(energy_diff.shape).reshape(2, resh).tolist()[1],
        ):
            if x[1] == x[0]:
                continue
            else:
                if dos == "Debye":
                    rat_mat[x[0], x[1]] = (
                        (
                            (
                                3
                                * (self.sys.pheff / Constant.Dydensity)
                                * Constant.meVtoJ
                            )
                            / (
                                2
                                * Constant.rho
                                * (Constant.c ** 4)
                                * (Constant.h_bar ** 3)
                            )
                        )
                        * (
                            self.Hph()
                            .matrix_element(states[x[0]].dag(), states[x[1]])
                            .real
                            ** 2
                        )
                        * bose_dist[x[0], x[1]]
                        * (energy_diff[x[0], x[1]] ** 2)
                    )
                else:
                    rat_mat[x[0], x[1]] = (
                        (2 * np.pi / (Constant.h_bar ** 2))
                        * (
                            self.Hph()
                            .matrix_element(states[x[0]].dag(), states[x[1]])
                            .real
                            ** 2
                        )
                        * bose_dist[x[0], x[1]]
                    )
        np.fill_diagonal(rat_mat.T, -rat_mat.T.sum(axis=0))
        return rat_mat

    def method_Orbach(self, energy_diff, resh, idx, states, dos, bose_dist):
        rat_mat = np.zeros(energy_diff.shape)
        for x in zip(
            np.indices(energy_diff.shape).reshape(2, resh).tolist()[0],
            np.indices(energy_diff.shape).reshape(2, resh).tolist()[1],
        ):
            if x[1] == x[0]:
                continue
            c_array = [i for i in idx if i not in [x[1], x[0]]]
            sum_ab = 0
            for c in c_array:
                mat_ca = self.Hph().matrix_element(states[c].dag(), states[x[1]]).real
                mat_bc = self.Hph().matrix_element(states[x[0]].dag(), states[c]).real
                try:
                    me = (mat_ca * mat_bc) ** 2
                #                 me = (((mat_ca*mat_bc)**2)/((mat_ca**2)+(mat_bc**2)))
                except ZeroDivisionError:
                    me = 0.0
                if dos == "Debye":
                    sum_ab += (
                        me
                        * abs(energy_diff[c, x[1]])
                        * abs(energy_diff[x[0], c])
                        * (bose_dist[c, x[1]] * bose_dist[x[0], c])
                    )
                else:
                    sum_ab += me * (bose_dist[c, x[1]] * bose_dist[x[0], c])
                rat_mat[x[0], x[1]] += (
                    (3 * (self.sys.pheff / Constant.Dydensity) * Constant.meVtoJ)
                    / (2 * Constant.rho * (Constant.c ** 4) * (Constant.h_bar ** 3))
                ) * sum_ab
        np.fill_diagonal(rat_mat.T, -rat_mat.T.sum(axis=0))
        return rat_mat

    def method_Raman(self, energy_diff, resh, idx, states, dos, bose_dist):
        rat_mat = np.zeros(energy_diff.shape)
        # H_a = [self.Jp(), self.Jm(), self.Jz()]
        H_a = [self.Hph_t1(), self.Hph_t2(), self.Hph_t3(), self.Hph_t4()]
        for h in zip(
            np.indices((len(H_a), len(H_a))).reshape(2, -1).tolist()[0],
            np.indices((len(H_a), len(H_a))).reshape(2, -1).tolist()[1],
        ):
            for x in zip(
                np.indices(energy_diff.shape).reshape(2, resh).tolist()[0],
                np.indices(energy_diff.shape).reshape(2, resh).tolist()[1],
            ):
                if x[1] == x[0]:
                    continue
                c_array = [i for i in idx if i not in [x[1], x[0]]]
                sum_ab = 0
                for c in c_array:
                    mat_ca = (
                        H_a[h[0]].matrix_element(states[c].dag(), states[x[1]]).real
                    )
                    mat_bc = (
                        H_a[h[1]].matrix_element(states[x[0]].dag(), states[c]).real
                    )
                    if dos == "Debye":
                        sum_ab += (
                            (mat_ca * mat_bc)
                            / (abs(energy_diff[c, x[1]]) + abs(energy_diff[x[0], c]))
                            * abs(energy_diff[c, x[1]])
                            * abs(energy_diff[x[0], c])
                            * (bose_dist[c, x[1]] * bose_dist[x[0], c]) ** 0.5
                        ) ** 2
                    else:
                        sum_ab += (
                            (
                                (mat_ca * mat_bc)
                                / (
                                    abs(energy_diff[c, x[1]])
                                    + abs(energy_diff[x[0], c])
                                )
                                * (bose_dist[c, x[1]] * bose_dist[x[0], c]) ** 0.5
                            )
                        ) ** 2
                rat_mat[x[0], x[1]] += (
                    (3 * (self.sys.pheff / Constant.Dydensity) * Constant.meVtoJ)
                    / (2 * Constant.rho * (Constant.c ** 4) * (Constant.h_bar ** 3))
                ) * sum_ab
        np.fill_diagonal(rat_mat.T, -rat_mat.T.sum(axis=0))
        return rat_mat

    def get_ph_trates(self, phonon_trates_method: str, dos: str):
        """
        Args:
            phonon_trates_method: Options between Direct, Orbach, and Raman
        """
        idx = self.get_idx()
        energies = self.get_energies(idx)
        energy_diff = np.subtract.outer(energies, energies)

        bose_dist = self.bose_dist(energies, energy_diff)
        resh = energy_diff.shape[0] * energy_diff.shape[1]

        H_tot = self.Htot(self.sys.B_x, self.sys.B_z)
        states = H_tot.eigenstates()[1][idx]
        if phonon_trates_method == "Direct":
            rat_mat = self.method_Direct(energy_diff, resh, dos, states, bose_dist)
        elif phonon_trates_method == "Orbach":
            rat_mat = self.method_Orbach(energy_diff, resh, idx, states, dos, bose_dist)
        elif phonon_trates_method == "Raman":
            rat_mat = self.method_Raman(energy_diff, resh, idx, states, dos, bose_dist)

        return rat_mat, energy_diff
