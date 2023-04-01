from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from dysprosium.utilities.dataclasses import LevelCrossing  # type: ignore


def empty_level_splitting():
    # the three rows are utility functions to pass the avoided level crossing
    # information into the functions that calculate the rates
    arr = np.empty((1, 6))
    arr[:] = np.nan
    lx = pd.DataFrame(
        arr,
        columns=[
            "J_z_lower",
            "J_z_upper",
            "level_splitting",
            "lower_ind_values",
            "rel_time",
            "upper_ind_values",
        ],
    ).loc[0]
    return lx


def get_local_files(directory: Path, filenames: list[str], n_s, sweep_rate):
    for filename in filenames:
        for level_crossing in LevelCrossing.load_csv(directory, filename):
            level_crossing.populate_attr(n_s, sweep_rate)
            yield level_crossing


def construct_level_splitting_df(
    directory: Path, filenames: list[str], n_s, sweep_rate
):
    df = pd.DataFrame(
        file for file in get_local_files(directory, filenames, n_s, sweep_rate)
    )
    if n_s == 5 / 2:
        columns = [
            "B_z",
            "level_splitting",
            "Jz_dif",
            "J_z_lower",
            "Ns_z_lower",
            "lower_ind_values",
            "J_z_upper",
            "Ns_z_upper",
            "upper_ind_values",
            "rel_time",
        ]
    else:
        columns = [
            "B_z",
            "level_splitting",
            "Jz_dif",
            "J_z_lower",
            "lower_ind_values",
            "J_z_upper",
            "upper_ind_values",
            "rel_time",
        ]
    df = df[df["level_splitting"] > 10 ** -9].sort_values("B_z")[columns]
    df = df.reset_index().drop(columns={"index"})

    return df


def construct_B_df(
    directory: Path, filenames: list[str], n_s, sweep_rate, sweep_num, B_63, B_66
):
    df = construct_level_splitting_df(directory, filenames, n_s, sweep_rate)

    B_sweep = np.linspace(-6, 6, sweep_num)
    sweep_df = pd.DataFrame(B_sweep, columns=["B_z"])
    sweep_df["rel_time"] = (
        np.diff(B_sweep).mean() * ((sweep_num - 1) / sweep_num) / sweep_rate
    )
    for row in df.iterrows():
        sweep_df.loc[abs(sweep_df["B_z"] - row[1]["B_z"]).idxmin(), "rel_time"] = (
            sweep_df.loc[abs(sweep_df["B_z"] - row[1]["B_z"]).idxmin(), "rel_time"]
            - row[1]["rel_time"]
        )
    if n_s == 5 / 2:
        B_df = (
            pd.concat(
                [
                    df[
                        [
                            "B_z",
                            "level_splitting",
                            "rel_time",
                            "J_z_lower",
                            "Ns_z_lower",
                            "lower_ind_values",
                            "J_z_upper",
                            "Ns_z_upper",
                            "upper_ind_values",
                        ]
                    ],
                    sweep_df,
                ],
                axis=0,
            )
            .sort_values("B_z")
            .reset_index()
            .drop(columns={"index"})
        )
        B_df["deltaS"] = (
            3
            * np.where(
                B_df.J_z_upper > 0,
                0.333
                * (
                    (B_df.J_z_upper - B_df.J_z_lower)
                    + (B_df.Ns_z_upper - B_df.Ns_z_lower)
                    + 2
                ),
                0.333
                * (
                    (B_df.J_z_lower - B_df.J_z_upper)
                    + (B_df.Ns_z_upper - B_df.Ns_z_lower)
                    + 2
                ),
            ).round()
        )
    else:
        B_df = (
            pd.concat(
                [
                    df[
                        [
                            "B_z",
                            "level_splitting",
                            "rel_time",
                            "J_z_lower",
                            "lower_ind_values",
                            "J_z_upper",
                            "upper_ind_values",
                        ]
                    ],
                    sweep_df,
                ],
                axis=0,
            )
            .sort_values("B_z")
            .reset_index()
            .drop(columns={"index"})
        )
        B_df["deltaS"] = (
            3
            * np.where(
                B_df.J_z_upper > 0,
                0.333 * ((B_df.J_z_upper - B_df.J_z_lower) + 2),
                0.333 * ((B_df.J_z_lower - B_df.J_z_upper) + 2),
            ).round()
        )

    scaling_B66 = (B_66 / (-0.0000075)) ** 3
    print("B66 scaling: ", scaling_B66)
    idx_B66 = B_df[
        ~(np.isnan(B_df.J_z_lower)) & ((B_df.deltaS == 12) | (B_df.deltaS == 18))
    ].index
    B_df.loc[idx_B66, "rel_time"] = B_df.loc[idx_B66, "rel_time"] * scaling_B66
    B_df.loc[idx_B66, "level_splitting"] = (
        B_df.loc[idx_B66, "level_splitting"] * scaling_B66
    )

    if B_63 != 0:
        scaling_B63 = B_63 / (-0.0000075 * (2 / 75))
        print("B63 scaling: ", scaling_B63)
        idx_B63 = B_df[~(np.isnan(B_df.J_z_lower)) & (B_df.deltaS == 15)].index
        B_df.loc[idx_B63, "rel_time"] = B_df.loc[idx_B63, "rel_time"] * scaling_B63
        B_df.loc[idx_B63, "level_splitting"] = (
            B_df.loc[idx_B63, "level_splitting"] * scaling_B63
        )

    # small correction to the J_z values
    B_df.loc[14, "J_z_lower"] = -5.5
    B_df.loc[14, "J_z_upper"] = 5.5

    return B_df
