from __future__ import annotations

import numpy as np

from Dy_utils.energy_difference_utils import EnergyDifferenceUtils
from Dy_utils.level_splittings import empty_level_splitting
from Dy_utils.operators import Operators


class ElectronRates(Operators):
    def get_idx(self, states_df):
        if self.atm.n_s == 5 / 2:
            lx = empty_level_splitting()
            states_df["4f_z"] = states_df["4f_z"].round(2)
            if ~np.isnan(lx["J_z_lower"]):
                states_df.loc[lx["lower_ind_values"], "4f_z"] = lx["J_z_lower"]
                states_df.loc[lx["lower_ind_values"], "Ns_z"] = lx["Ns_z_lower"]
                states_df.loc[lx["upper_ind_values"], "4f_z"] = lx["J_z_upper"]
                states_df.loc[lx["upper_ind_values"], "Ns_z"] = lx["Ns_z_upper"]
            five_idx = states_df[(states_df["4f_z"] < 5.8) & (states_df["4f_z"] > 0)][
                "4f_z"
            ].index.values
            mfive_idx = states_df[(states_df["4f_z"] > -5.8) & (states_df["4f_z"] < 0)][
                "4f_z"
            ].index.values
            states_df.loc[mfive_idx, "4f_z"] = -5
            states_df.loc[five_idx, "4f_z"] = 5
            idx = (
                states_df.sort_values(
                    ["4f_z", "Ns_z"], ascending=[True, True]
                ).index.values
                - 1
            )
        else:
            if self.sys.B_z == 0:
                idx = [3, 1, 6, 7, 8, 5, 2, 4]
            elif self.sys.B_z == 2.7280786:
                idx = [5, 2, 6, 8, 7, 4, 1, 3]
            elif self.sys.B_z == -2.7280786:
                idx = [3, 1, 4, 7, 8, 6, 2, 5]
            else:
                idx = states_df.sort_values("4f_z").index.values - 1
        return idx

    def get_energies(self, states_df):
        if self.atm.n_s == 5 / 2:
            energies = states_df.sort_values(["4f_z", "Ns_z"], ascending=[True, True])[
                "E (meV)"
            ].to_numpy()
        else:
            energies = energies = states_df.sort_values("4f_z")["E (meV)"].to_numpy()
        return energies

    def mat_ele_el(self, energy_diff, states):
        if self.atm.n_s == 5 / 2:
            shapey = 2304
        else:
            shapey = 64

        mat_ele_el_z = np.zeros(energy_diff.shape)
        mat_ele_el_p = np.zeros(energy_diff.shape)
        mat_ele_el_m = np.zeros(energy_diff.shape)
        for x in zip(
            np.indices(energy_diff.shape).reshape(2, shapey).tolist()[0],
            np.indices(energy_diff.shape).reshape(2, shapey).tolist()[1],
        ):
            if self.atm.v_d == 0:
                mat_ele_el_z[x[0], x[1]] = (
                    2
                    * (self.Jz().matrix_element(states[x[0]].dag(), states[x[1]]).real)
                    ** 2
                )
                mat_ele_el_p[x[0], x[1]] = (
                    self.Jp().matrix_element(states[x[0]].dag(), states[x[1]]).real ** 2
                )
                mat_ele_el_m[x[0], x[1]] = (
                    self.Jm().matrix_element(states[x[0]].dag(), states[x[1]]).real ** 2
                )
            else:
                mat_ele_el_z[x[0], x[1]] = (
                    2
                    * (self.Dz().matrix_element(states[x[0]].dag(), states[x[1]]).real)
                    ** 2
                )
                mat_ele_el_p[x[0], x[1]] = (
                    self.Dp().matrix_element(states[x[0]].dag(), states[x[1]]).real ** 2
                )
                mat_ele_el_m[x[0], x[1]] = (
                    self.Dm().matrix_element(states[x[0]].dag(), states[x[1]]).real ** 2
                )

        return mat_ele_el_z, mat_ele_el_p, mat_ele_el_m

    def el_rat_mat(self, energy_diff, states, rat_mat, sec_el, surface_only):
        energy_diff_fcns = EnergyDifferenceUtils(self.sys, energy_diff)
        mat_ele_el_z, mat_ele_el_p, mat_ele_el_m = self.mat_ele_el(energy_diff, states)
        ele_trates_T = (
            np.multiply(mat_ele_el_m, energy_diff_fcns.RmT(surface_only))
            + np.multiply(mat_ele_el_p, energy_diff_fcns.RpT(surface_only))
            + np.multiply(mat_ele_el_z, energy_diff_fcns.RzT(surface_only))
        )
        ele_trates_S = (
            np.multiply(mat_ele_el_m, energy_diff_fcns.RmS(sec_el, surface_only))
            + np.multiply(mat_ele_el_p, energy_diff_fcns.RpS(sec_el, surface_only))
            + np.multiply(mat_ele_el_z, energy_diff_fcns.RzS(sec_el, surface_only))
        )
        ele_trates = ele_trates_T + ele_trates_S
        el_rat_mat = ele_trates + rat_mat
        np.fill_diagonal(el_rat_mat, 0)
        np.fill_diagonal(el_rat_mat, -el_rat_mat.sum(axis=0))

        return el_rat_mat
