"""Test."""
#!/usr/bin/env python
from __future__ import annotations

# import pandas as pd  # type: ignore

# from parse import compile  # type: ignore

from dysprosium.utilities.dataclasses import (  # type: ignore
    AtomicParameters,
    # Constant,
    # LevelCrossing,
    SystemParameters,
)

# from Dy_utils.level_splittings import construct_B_df, construct_level_splitting_df
from dysprosium.utilities.operators import Operators  # type: ignore

# from Dy_utils.phonon_process import PhononProcess
# from Dy_utils.plotting import plot_rates

# from level_splittings import LEVEL_SPLITTINGS


def main():
    """Docstring.
    """
    atomic_parameters = AtomicParameters()
    system_parameters = SystemParameters(B_x=0, B_z=1)

    # # Pick which phonon process, rat_mat is the rate matrix, energy diff is an 8x8 matrix
    # # with the energy differences for the field value that you use
    # # Possible phonon_trates_method: Direct, Orbach, Raman
    # ph = PhononProcess(atomic_parameters, system_parameters)
    # rat_mat, energy_diff = ph.get_ph_trates(phonon_trates_method="Raman", dos="Debye")

    wave_fcn = Operators(atomic_parameters, system_parameters)
    states = wave_fcn.extract_states(system_parameters.B_x, system_parameters.B_z, True)
    print(states)


if __name__ == "__main__":
    main()
