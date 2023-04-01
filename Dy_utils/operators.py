from __future__ import annotations

import pandas as pd
from qutip import operators, tensor

from Dy_utils.dataclasses import AtomicParameters, Constant, SystemParameters


class Operators:
    def __init__(
        self, atomic_parameters: AtomicParameters, system_parameters: SystemParameters
    ):
        self.atm = atomic_parameters
        self.sys = system_parameters

    @staticmethod
    def splus(s):
        return operators.jmat(s, "+")

    @staticmethod
    def sminus(s):
        return operators.jmat(s, "-")

    @staticmethod
    def sx(s):
        return operators.jmat(s, "x")

    @staticmethod
    def sy(s):
        return operators.jmat(s, "y")

    @staticmethod
    def sz(s):
        return operators.jmat(s, "z")

    @staticmethod
    def idm(s):
        return operators.identity(int(2 * s + 1))

    def Dx(self):
        return tensor(
            [
                self.sx(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.idm(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def Dy(self):
        return tensor(
            [
                self.sy(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.idm(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def Dz(self):
        return tensor(
            [
                self.sz(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.idm(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def Ix(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.sx(self.atm.v_s),
                self.idm(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def Iy(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.sy(self.atm.v_s),
                self.idm(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def Iz(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.sz(self.atm.v_s),
                self.idm(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def Jx(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.sx(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def Jy(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.sy(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def Jz(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.sz(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def Fx(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.idm(self.atm.s),
                self.sx(self.atm.n_s),
            ]
        )

    def Fy(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.idm(self.atm.s),
                self.sy(self.atm.n_s),
            ]
        )

    def Fz(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.idm(self.atm.s),
                self.sz(self.atm.n_s),
            ]
        )

    def iddm(self):
        return tensor(
            [
                self.idm(self.atm.v_d),
                self.idm(self.atm.v_s),
                self.idm(self.atm.s),
                self.idm(self.atm.n_s),
            ]
        )

    def J2(self):
        return (self.Jx() ** 2) + (self.Jy() ** 2) + (self.Jz() ** 2)

    def Jp(self):
        return self.Jx() + 1j * self.Jy()

    def Jm(self):
        return self.Jx() - 1j * self.Jy()

    def Im(self):
        return self.Ix() - 1j * self.Iy()

    def Dp(self):
        return self.Dx() + 1j * self.Dy()

    def Dm(self):
        return self.Dx() - 1j * self.Dy()

    def H_val(self):
        H_val = (
            -Constant.A_sf
            * (self.Ix() * self.Jx() + self.Iy() * self.Jy() + self.Iz() * self.Jz())
            - Constant.A_df
            * (self.Dx() * self.Jx() + self.Dy() * self.Jy() + self.Dz() * self.Jz())
            - Constant.A_sd
            * (self.Ix() * self.Dx() + self.Iy() * self.Dy() + self.Iz() * self.Dz())
        )
        return H_val

    def H_hf(self):
        H_hf = -Constant.F * (
            self.Fx() * (self.Jx() + self.Ix() + self.Dx())
            + self.Fy() * (self.Jy() + self.Iy() + self.Dy())
            + self.Fz() * (self.Jz() + self.Iz() + self.Dz())
        )
        return H_hf

    def H_z(self, B_x, B_z):
        H_z = (
            Constant.mu_b
            * B_z
            * (
                Constant.g_z * self.Jz()
                + Constant.g * self.Iz()
                + Constant.g * self.Dz()
            )
        ) + (
            Constant.mu_b
            * B_x
            * (
                Constant.g_z * self.Jx()
                + Constant.g * self.Ix()
                + Constant.g * self.Dx()
            )
        )
        if self.atm.n_s == 5 / 2 and self.atm.v_d == 1 / 2:
            H_z.dims = [[2, 2, 17, 6], [2, 2, 17, 6]]
        elif self.atm.n_s == 0 and self.atm.v_d == 1 / 2:
            H_z.dims = [[2, 2, 17], [2, 2, 17]]
        elif self.atm.n_s == 5 / 2 and self.atm.v_d == 0:
            H_z.dims = [[17, 6], [17, 6]]
        else:
            H_z.dims = [[17], [17]]
        return -H_z

    def H_B20(self):
        t1 = self.Jz() * self.Jz()
        t2 = (1 / 3) * self.J2()
        t2.dims = t1.dims
        H_B20 = 3 * self.sys.B_20 * (t1 - t2)
        return H_B20

    def H_B21(self):
        H_B21 = (
            (1 / 4)
            * self.sys.B_21
            * (
                self.Jz() * (self.Jp() + self.Jm())
                + (self.Jp() + self.Jm()) * self.Jz()
            )
        )
        return H_B21

    def H_B22(self):
        H_B22 = self.sys.B_22 * (self.Jx() * self.Jx() - self.Jy() * self.Jy())
        return H_B22

    def H_B40(self):
        t1 = self.Jz() * self.Jz()
        t2 = self.J2()
        t2.dims = t1.dims
        t3 = t1 * t2
        H_B40 = self.sys.B_40 * (
            35 * (t1 * t1) - 30 * t3 + 25 * t1 - 6 * t2 + 3 * (t2 * t2)
        )
        return H_B40

    def H_B42(self):
        t1 = self.Jz() * self.Jz()
        t2 = self.J2()
        t2.dims = t1.dims
        t3 = self.iddm()
        t3.dims = t1.dims
        t4 = self.Jp() * self.Jp() + self.Jm() * self.Jm()
        H_B42 = (
            (1 / 4)
            * self.sys.B_42
            * (((7 * t1 - t2 - 5 * t3) * t4) + (t4 * (7 * t1 - t2 - 5 * t3)))
        )
        return H_B42

    def H_B43(self):
        t1 = self.Jp() * self.Jp()
        t2 = self.Jp()
        t2.dims = t1.dims
        t3 = self.Jm() * self.Jm()
        t4 = self.Jm()
        t4.dims = t3.dims
        t5 = (t1 * t2) + (t3 * t4)
        t6 = self.Jz()
        t6.dims = t5.dims
        H_B43 = (1 / 4) * self.sys.B_43 * ((t6 * t5) + (t5 * t6))
        return H_B43

    def H_B44(self):
        H_B44 = (
            (1 / 2)
            * self.sys.B_44
            * (
                (self.Jp() * self.Jp()) * (self.Jp() * self.Jp())
                + (self.Jm() * self.Jm()) * (self.Jm() * self.Jm())
            )
        )
        return H_B44

    def H_B60(self):
        t1 = self.Jz() * self.Jz()
        t2 = self.J2()
        t2.dims = t1.dims
        H_B60 = self.sys.B_60 * (
            (231 * t1 * t1 * t1)
            - (315 * t2 * t1 * t1)
            + (735 * t1 * t1)
            + (105 * t2 * t2 * t1 - 525 * t2 * t1 + 294 * t1)
            - (5 * t2 * t2 * t2)
            + (40 * t2 * t2)
            - (60 * t2)
        )
        return H_B60

    # B62 is broken due to idm
    def H_B62(self):
        t1 = self.Jz() * self.Jz()
        t2 = self.Jp() + self.Jm()
        t2.dims = t1.dims
        t3 = self.J2()
        t3.dims = t1.dims
        H_B62 = (1 / 4) * (
            (
                (
                    (33 * t1 * t1)
                    - ((18 * t3 + 123 * self.idm(self.atm.s)) * t1)
                    + (5 * t3 * t3 + 10 * t3 + 102 * self.idm(self.atm.s)) * self.Jz()
                )
                * t2
            )
            + (
                t2
                * (
                    (33 * t1 * t1)
                    - ((18 * t3 + 123 * self.idm(self.atm.s)) * t1)
                    + (5 * t3 * t3 + 10 * t3 + 102 * self.idm(self.atm.s)) * self.Jz()
                )
            )
        )
        return H_B62

    def H_B63(self):
        t1 = self.Jz() * self.Jz()
        t2 = self.Jz()
        t2.dims = t1.dims
        t3 = self.Jp() * self.Jp()
        t4 = self.Jp()
        t4.dims = t1.dims
        t5 = self.Jm() * self.Jm()
        t6 = self.Jm()
        t6.dims = t1.dims
        t7 = self.J2()
        t7.dims = t1.dims
        H_B63 = (
            (1 / 4)
            * self.sys.B_63
            * (
                ((11 * t1 * t2 - 3 * t7 * t2 - 59 * t2) * (t3 * t4 + t5 * t6))
                + ((t3 * t4 + t5 * t6) * (11 * t1 * t2 - 3 * t7 * t2 - 59 * t2))
            )
        )
        return H_B63

    def H_B64(self):
        t1 = self.Jz() * self.Jz()
        t2 = self.J2()
        t2.dims = t1.dims
        t3 = self.Jp() * self.Jp()
        t4 = self.Jm() * self.Jm()
        t5 = self.iddm()
        t5.dims = t1.dims
        H_B64 = (
            (1 / 4)
            * self.sys.B_64
            * (
                ((11 * t1 - t2 - 38 * t5) * (t3 * t3 + t4 * t4))
                + ((t3 * t3 + t4 * t4) * (11 * t1 - t2 - 38 * t5))
            )
        )
        return H_B64

    def H_B66(self):
        H_B66 = (
            (1 / 2)
            * self.sys.B_66
            * (
                (
                    (self.Jp() * self.Jp())
                    * (self.Jp() * self.Jp())
                    * (self.Jp() * self.Jp())
                )
                + (self.Jm() * self.Jm())
                * (self.Jm() * self.Jm())
                * (self.Jm() * self.Jm())
            )
        )
        return H_B66

    def Htot(self, B_x, B_z):
        H_tot = (
            self.H_val()
            + self.H_hf()
            + self.H_z(B_x, B_z)
            + self.H_B20()
            + self.H_B40()
            + self.H_B60()
            + self.H_B63()
            + self.H_B66()
        )
        return H_tot

    def Hph(self):
        t1 = self.Jp() * self.Jp()
        t2 = self.Jm() * self.Jm()
        t3 = self.Jp() * self.Jz() + self.Jz() * self.Jp()
        t4 = self.Jm() * self.Jz() + self.Jz() * self.Jm()
        t5 = self.Jz() * self.Jz()
        H_ph = t1 + t2 + t3 + t4 + t5
        return H_ph

    def Hph_t1(self):
        t1 = self.Jp() * self.Jp()
        return t1

    def Hph_t2(self):
        t2 = self.Jm() * self.Jm()
        return t2

    def Hph_t3(self):
        t3 = self.Jp() * self.Jz() + self.Jz() * self.Jp()
        return t3

    def Hph_t4(self):
        t4 = self.Jm() * self.Jz() + self.Jz() * self.Jm()
        return t4

    def Hte(self):
        H_te = (1 * self.Dz()) + self.Dp() + self.Dm()
        H_te.dims = [[2, 2, 17], [2, 2, 17]]
        return H_te

    def determine_state(self, bra, ket, energy):
        if self.atm.n_s == 5 / 2:
            state = {
                "4f_z": self.Jz().matrix_element(bra, ket).real,
                "5d_z": self.Dz().matrix_element(bra, ket).real,
                "6s_z": self.Iz().matrix_element(bra, ket).real,
                "Ns_z": self.Fz().matrix_element(bra, ket).real,
                "E (meV)": energy,
            }
        else:
            state = {
                "4f_z": self.Jz().matrix_element(bra, ket).real,
                "5d_z": self.Dz().matrix_element(bra, ket).real,
                "6s_z": self.Iz().matrix_element(bra, ket).real,
                "E (meV)": energy,
            }
        return state

    def get_states(self, num_states, H_tot, rescale: bool):
        for i in range(num_states):
            energies = H_tot.eigenstates()[0]
            if rescale:
                energies = H_tot.eigenstates()[0] - H_tot.eigenstates()[0][0]
            energies = energies.real
            state = self.determine_state(
                bra=H_tot.eigenstates()[1][i].dag(),
                ket=H_tot.eigenstates()[1][i],
                energy=energies[i],
            )
            yield state

    def extract_states(self, B_x, B_z, rescale: bool):
        if self.atm.n_s == 5 / 2:
            num_states = 48
        else:
            num_states = 8
        states_df = pd.DataFrame(
            [
                state
                for state in self.get_states(
                    num_states, H_tot=self.Htot(B_x, B_z), rescale=rescale
                )
            ]
        )
        states_df.index += 1

        return states_df
