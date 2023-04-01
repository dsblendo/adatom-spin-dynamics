from __future__ import annotations

import math

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import numpy.ma as ma  # type: ignore


def format_master(axis, col, row):
    axis[col, row].set_xlabel("Time ", fontsize=22, labelpad=10)
    axis[col, row].set_ylabel("P", fontsize=22, rotation=90, labelpad=10)
    axis[col, row].tick_params(axis="both", which="major", labelsize=22)
    axis[col, row].grid(color="black", alpha=0.5)
    axis[col, row].legend(loc="best", fontsize=18)


def plot_master(t, soln, n_s, savename="master_el_wQTM"):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

    if n_s == 5 / 2:
        ax[0, 0].plot(t, soln[:, 0:6].sum(axis=1), c="blue", alpha=1, label="-8")
        ax[0, 0].plot(t, soln[:, 42:].sum(axis=1), c="orange", alpha=1, label="+8")
        ax[0, 1].plot(t, soln[:, 6:12].sum(axis=1), c="blue", alpha=1, label="-7")
        ax[0, 1].plot(t, soln[:, 36:42].sum(axis=1), c="orange", alpha=1, label="+7")
        ax[1, 0].plot(t, soln[:, 12:18].sum(axis=1), c="blue", alpha=1, label="-6")
        ax[1, 0].plot(t, soln[:, 30:36].sum(axis=1), c="orange", alpha=1, label="+6")
        ax[1, 1].plot(t, soln[:, 18:24].sum(axis=1), c="blue", alpha=1, label="-5")
        ax[1, 1].plot(t, soln[:, 24:30].sum(axis=1), c="orange", alpha=1, label="+5")
    else:
        ax[0, 0].plot(t, soln[:, 0], c="blue", alpha=1, label="-8")
        ax[0, 0].plot(t, soln[:, 7], c="orange", alpha=1, label="+8")
        ax[0, 1].plot(t, soln[:, 1], c="blue", alpha=1, label="-7")
        ax[0, 1].plot(t, soln[:, 6], c="orange", alpha=1, label="+7")
        ax[1, 0].plot(t, soln[:, 2], c="blue", alpha=1, label="-6")
        ax[1, 0].plot(t, soln[:, 5], c="orange", alpha=1, label="+6")
        ax[1, 1].plot(t, soln[:, 3], c="blue", alpha=1, label="-5")
        ax[1, 1].plot(t, soln[:, 4], c="orange", alpha=1, label="+5")

    for col, row in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        format_master(ax, col, row)
    plt.tight_layout()
    plt.show()

    fig.savefig(f"{savename}.pdf")

    return


def plot_tau_fit(t_plot, fit_func_plot, soln_plot):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(t_plot, fit_func_plot, c="black", ls="dotted", alpha=1, label="fit")
    ax.plot(t_plot, soln_plot, c="blue", alpha=0.4, label="+5, +6, +7, +8")
    ax.set_xlabel("Time (S)", fontsize=16, labelpad=4)
    ax.set_ylabel("P", fontsize=16, rotation=90, labelpad=4)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(color="black", alpha=0.5)
    ax.legend(fontsize=16)
    fig.tight_layout()
    plt.show()


def magnitude(value):
    if value == 0:
        v = int(math.floor(math.log10(abs(1e-100))))
    elif math.isnan(value):
        v = int(math.floor(math.log10(abs(1e-100))))
    else:
        v = int(math.floor(math.log10(abs(value))))
    return v


def disp_value(value):
    if value < 1e-20:
        v = "<{:.0e}".format(1e-20)
    else:
        v = "{:.0e}".format(value)
    return v


def plot_rates(B_z, rat_mat, savename, display: bool):
    sinks = ma.masked_where(rat_mat >= 0, rat_mat)
    sinks2 = ma.masked_where(rat_mat >= 0, rat_mat)
    sources = ma.masked_where(rat_mat < 0, rat_mat)
    sources2 = ma.masked_where(rat_mat < 0, rat_mat)
    for i in range(sinks.data.shape[0]):
        for j in range(sinks.data.shape[1]):
            sinks.data[i, j] = magnitude(sinks.data[i, j])
    for i in range(sources.data.shape[0]):
        for j in range(sources.data.shape[1]):
            sources.data[i, j] = magnitude(sources.data[i, j])

    states = ["-8", "-7", "-6", "-5", "5", "6", "7", "8"]

    fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [30, 1]}, figsize=(12, 8))
    ima = ax[0].imshow(sources, cmap="Blues", vmin=-10, vmax=12)
    imb = ax[0].imshow(sinks, cmap="Reds", vmin=-4, vmax=18)

    cbara = ax[0].figure.colorbar(ima, ax=ax[0], pad=0.02)
    cbara.ax.set_title("Source Rate \n Order of Mag", fontsize=12, pad=20)
    cbara.ax.tick_params(labelsize=16)

    cbarb = ax[0].figure.colorbar(imb, ax=ax[0], pad=0.02)
    cbarb.ax.set_title("Sink Rate \n Order of Mag", fontsize=12, pad=20)
    cbarb.ax.tick_params(labelsize=16)

    ax[0].set_xticks(np.arange(len(states)))
    ax[0].set_yticks(np.arange(len(states)))
    ax[0].set_xticklabels(states, fontsize=16)
    ax[0].set_yticklabels(states, fontsize=16)

    plt.setp(ax[0].get_xticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    for i in range(len(states)):
        for j in range(len(states)):
            if i == j:
                text = ax[0].text(
                    j,
                    i,
                    disp_value(sinks2.data[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                )
            else:
                text = ax[0].text(
                    j,
                    i,
                    disp_value(sources2.data[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                )

    ax[0].xaxis.tick_top()
    ax[0].tick_params(bottom=False, left=False, top=False)
    ax[0].xaxis.set_label_position("top")
    ax[0].set_xlabel("Initial State", fontsize=16, labelpad=10)
    ax[0].set_ylabel("Final State", fontsize=16, labelpad=10)

    ax[1].plot([0, 1], [B_z, B_z], c="black", alpha=1, lw=4)
    ax[1].tick_params(labelbottom=False, bottom=False)
    ax[1].set_title("B_z", fontsize=16, pad=20)
    ax[1].set_yticks(np.linspace(-6, 6, 13))
    ax[1].set_yticklabels(np.linspace(-6, 6, 13), fontsize=16)
    ax[1].yaxis.set_label_position("right")

    fig.tight_layout()

    fig.savefig(f"{savename}_{round(B_z,2)}T.pdf")

    if display == True:
        plt.show()
    else:
        plt.close()
    return
