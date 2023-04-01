from __future__ import annotations

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit

from Dy_utils.dataclasses import Constant
from Dy_utils.electron_rates import ElectronRates
from Dy_utils.operators import Operators
from Dy_utils.plotting import plot_master, plot_tau_fit


def dP_dt(P, t, rat_mat):
    P_t = np.dot(rat_mat, P)
    return P_t


def fit_func(t, alpha_star, beta_star, tau_star):
    return alpha_star + (beta_star * np.exp((-t / tau_star)))


class MasterEquation(ElectronRates, Operators):
    def get_omega(self, B_df):
        states_df = self.extract_states(
            B_x=self.sys.B_x, B_z=self.sys.B_z, rescale=False
        )
        if self.atm.n_s == 5 / 2:
            print(states_df.head(48))
            st = (
                self.extract_states(B_x=self.sys.B_x, B_z=-6, rescale=False)[
                    ["4f_z", "Ns_z"]
                ]
                .round(1)
                .sort_values(["4f_z", "Ns_z"], ascending=[True, True])
                .reset_index()
                .drop(columns={"index"})
            )
            omega = np.zeros((st.shape[0], st.shape[0]))
            for row in B_df[~B_df.J_z_lower.isna()].iterrows():
                idl = st[
                    (st["4f_z"] == row[1]["J_z_lower"])
                    & (st["Ns_z"] == row[1]["Ns_z_lower"])
                ].index.values
                idu = st[
                    (st["4f_z"] == row[1]["J_z_upper"])
                    & (st["Ns_z"] == row[1]["Ns_z_upper"])
                ].index.values
                omega[idl, idu] = row[1]["level_splitting"]
                omega[idu, idl] = row[1]["level_splitting"]
        else:
            print(states_df.head(8))
            st = (
                self.extract_states(B_x=self.sys.B_x, B_z=-6, rescale=False)[["4f_z"]]
                .round(1)
                .sort_values(["4f_z"])
                .reset_index()
                .drop(columns={"index"})
            )
            omega = np.zeros((st.shape[0], st.shape[0]))
            for row in B_df[~B_df.J_z_lower.isna()].iterrows():
                idl = st[(st["4f_z"] == row[1]["J_z_lower"])].index.values
                idu = st[(st["4f_z"] == row[1]["J_z_upper"])].index.values
                omega[idl, idu] = row[1]["level_splitting"]
                omega[idu, idl] = row[1]["level_splitting"]
        return omega

    def get_el_trates(self, rat_mat, sec_el, surface_only):
        states_df = self.extract_states(self.sys.B_x, self.sys.B_z, rescale=False)

        energies = self.get_energies(states_df)
        energy_diff = -np.subtract.outer(energies, energies)

        H_tot = self.Htot(self.sys.B_x, self.sys.B_z)
        idx = self.get_idx(states_df)
        states = H_tot.eigenstates()[1][idx]

        el_rat_mat = self.el_rat_mat(energy_diff, states, rat_mat, sec_el, surface_only)

        return el_rat_mat

    def det_QTM(self, B_df, el_rat_mat, energy_diff):
        omega = self.get_omega(B_df)
        tau = np.zeros((el_rat_mat.shape))
        np.fill_diagonal(tau, -1 / np.diag(el_rat_mat))
        if self.atm.n_s == 5 / 2:
            tau = (
                np.diag(tau)[:24]
                * np.flip(np.diag(tau)[24:])
                / (np.diag(tau)[:24] + np.flip(np.diag(tau)[24:]))
            )
        else:
            tau = (
                np.diag(tau)[:4]
                * np.flip(np.diag(tau)[4:])
                / (np.diag(tau)[:4] + np.flip(np.diag(tau)[4:]))
            )
        tau_mat = np.zeros((el_rat_mat.shape))
        np.fill_diagonal(tau_mat, np.append(tau, np.flip(tau)))
        qtm_rates = (2 * np.fliplr(tau_mat) * (omega / Constant.h_bar) ** 2) / (
            1 + (np.fliplr(tau_mat) * energy_diff / Constant.h_bar) ** 2
        )
        rat_matQTM = el_rat_mat + qtm_rates
        np.fill_diagonal(rat_matQTM, 0)
        np.fill_diagonal(rat_matQTM, -rat_matQTM.sum(axis=0))
        return rat_matQTM

    def converge_check(self, soln):
        if self.atm.n_s == 5 / 2:
            soln_range = soln[-500:, 0:24].sum(axis=1)
        else:
            soln_range = soln[-500:, 0:4].sum(axis=1)

        occ_gradient = np.gradient(soln_range).mean()
        if occ_gradient > 10 ** -10:
            print("Convergence warning: dP/dt is still too large! Increasing t_max..")
            converged = False
        else:
            converged = True

        return converged

    def compute_soln(self, P, t, el_rat_matQTM, converge_master, plot: bool):
        if converge_master == True:
            converged = False
            while converged == False:
                soln = odeint(dP_dt, P, t, args=(el_rat_matQTM,))
                converged = self.converge_check(soln)
                t = np.linspace(0, t[-1] * 10, 10 ** 6)
            t = np.linspace(0, t[-1] / 10, 10 ** 6)
        else:
            soln = odeint(dP_dt, P, t, args=(el_rat_matQTM,))

        if plot:
            plot_master(t, soln, n_s=self.atm.n_s)

        return t, soln

    def tau_fit(self, t, soln, P, plot: bool):
        valid = ~(np.isnan(t) | np.isnan(soln[:].sum(axis=1)))
        t = t[valid]
        soln = soln[valid, :]

        if self.atm.n_s == 0:
            if P[6] > P[1]:
                soln_range = soln[0:, :4].sum(axis=1)
            else:
                soln_range = soln[0:, 4:].sum(axis=1)
        else:
            if P[36:42].sum() > P[6:12].sum():
                soln_range = soln[0:, :24].sum(axis=1)
            else:
                soln_range = soln[0:, 24:].sum(axis=1)

        bd = (
            [-0.01 + soln_range[-1], -0.05 - soln_range[-1], 10 ** -21],
            [0.01 + soln_range[-1], +0.05 - soln_range[-1], t[-1]],
        )
        try:
            popt, pcov = curve_fit(
                fit_func, t[0:], soln_range, method="trf", maxfev=5000, bounds=bd
            )
        except ValueError:
            popt, pcov = curve_fit(
                fit_func, t[0:], soln_range, method="dogbox", maxfev=5000, bounds=bd
            )

        if plot:
            t_plot = t[0:]
            fit_func_plot = fit_func(t[0:], popt[0], popt[1], popt[2])
            plot_tau_fit(t_plot, fit_func_plot, soln_plot=soln_range)

        return popt, pcov

    def master(self, P, t, el_rat_matQTM, converge_master=True, plot=False):
        t, soln = self.compute_soln(P, t, el_rat_matQTM, converge_master, plot)

        if self.atm.n_s == 5 / 2:
            occ_minus, occ_plus = soln[-1][0:24].sum(), soln[-1][24:].sum()
        else:
            occ_minus, occ_plus = soln[-1][0:4].sum(), soln[-1][4:].sum()

        t_iter = soln[:, 6].shape[0]
        t = np.linspace(0, t[-1], t_iter)
        popt, pcov = self.tau_fit(t, soln, P, plot)
        tau_el = popt[2]
        count = 0
        while popt[2] == 1:
            t = t / 10
            count += 1
            popt, pcov = self.tau_fit(t, soln, P, plot)
            tau_el = popt[2] * (10 ** count)
            print("Fit failed. Attempting time rescale...")
        print("with QTM: ")
        print("occ_minus:", "occ_plus:", "tau*_el:", "ln(1/tau):")
        print(
            f"{round(occ_minus,3)}\t{round(occ_plus,3)}\t{tau_el}\t{round(np.log(1/tau_el),3)}"
        )

        return occ_minus, occ_plus, tau_el, soln
