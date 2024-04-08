from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_acqui_f(
    gp,
    fig,
    pos,
    val,
    shape=None,
    old_aqf=None,
    last_spectrum=None,
    settings=None,
) -> tuple[plt.Figure | Any, np.ndarray]:
    """Plot the acquisition function of a GP

    Args:
        gp (GaussianProcess): GP object

    Returns:
        None
    """

    positions, values = gp.x_data, gp.y_data

    if shape is None:
        shape = settings["plots"]["posterior_map_shape"]
    x_pred_0 = np.empty((np.prod(shape), 3))
    x_pred_1 = np.empty((np.prod(shape), 3))
    counter = 0
    x = np.linspace(0, shape[0] - 1, shape[0])
    y = np.linspace(0, shape[1] - 1, shape[1])

    lim_x = gp.input_space_bounds[0]
    lim_y = gp.input_space_bounds[1]

    delta_x = (lim_x[1] - lim_x[0]) / shape[0]
    delta_y = (lim_y[1] - lim_y[0]) / shape[1]

    for i in x:
        for j in y:
            # x_pred[counter] = np.array([15*i/100,-15+15*j/100,0])
            x_pred_0[counter] = np.array(
                [delta_x * i + lim_x[0], delta_y * j + lim_y[0], 0]
            )
            x_pred_1[counter] = np.array(
                [delta_x * i + lim_x[0], delta_y * j + lim_y[0], 1]
            )
            counter += 1

    PM0 = np.reshape(gp.posterior_mean(x_pred_0)["f(x)"], shape)
    PV0 = np.reshape(gp.posterior_covariance(x_pred_0)["v(x)"], shape)
    sPV0 = np.sqrt(PV0)
    PM1 = np.reshape(gp.posterior_mean(x_pred_1)["f(x)"], shape)
    PV1 = np.reshape(gp.posterior_covariance(x_pred_1)["v(x)"], shape)
    sPV1 = np.sqrt(PV1)

    a = settings["acquisition_function"]["params"]["a"]
    norm = settings["acquisition_function"]["params"]["norm"]
    w = settings["acquisition_function"]["params"]["weights"]
    if w is None:
        w = (1, 1)
    aqf = norm * (a * np.sqrt(w[0] * PV0 + w[1] * PV1) + (w[0] * PM0 + w[1] * PM1))
    aqf = np.rot90(aqf, k=-1)[:, ::-1]

    # fig,ax=plt.subplots(3,1,figsize=(8,6))
    # ax[0].imshow(PV)
    # ax[1].imshow(PM)
    # ax[2].imshow(a[:,2].reshape(100,250))
    # for aa in ax:
    #     aa.scatter(points[:-1,1],points[:-1,0],s=1,color='orange')
    #     aa.scatter(points[-1,1],points[-1,0],s=2,color='red')
    if fig is None:
        fig = plt.figure("ACQ func", figsize=(12, 8), layout="constrained")
    else:
        fig.clear()

    ax = [
        fig.add_subplot(331),
        fig.add_subplot(332),
        fig.add_subplot(333),
        fig.add_subplot(334),
        fig.add_subplot(335),
        fig.add_subplot(336),
        fig.add_subplot(337),
        fig.add_subplot(338),
        fig.add_subplot(339),
    ]

    # fig,ax = plt.subplots(2,2,)
    ax = np.asarray(ax).reshape(3, 3)
    for i, PM, PV in zip(range(2), [PM0, PM1], [sPV0, sPV1]):
        PM = np.rot90(PM, k=-1)[:, ::-1]
        PV = np.rot90(PV, k=-1)[:, ::-1]
        pmmax = PM.max()
        pvmax = PV.max()
        PM /= pmmax
        PV /= pvmax

        ax[i, 0].imshow(
            PM, clim=[0, 1], extent=[*lim_x, *lim_y], origin="lower", aspect="equal"
        )
        ax[i, 0].set_title(f"PM: {pmmax:.3f}")
        ax[i, 1].imshow(
            PV, clim=[0, 1], extent=[*lim_x, *lim_y], origin="lower", aspect="equal"
        )
        ax[i, 1].set_title(f"PV: {a * np.sqrt(pvmax):.3f}")

        ax[i, 0].scatter(positions[:, 0], positions[:, 1], s=15, c="r", alpha=0.5)
        ax[i, 1].scatter(positions[:, 0], positions[:, 1], s=15, c="r", alpha=0.5)
        ax[i, 0].scatter(positions[-1, 0], positions[-1, 1], s=30, c="white")
        ax[i, 1].scatter(positions[-1, 0], positions[-1, 1], s=30, c="white")

    ax[0, 2].imshow(
        np.zeros_like(PM),
        clim=[0, 1],
        extent=[*lim_x, *lim_y],
        origin="lower",
        aspect="equal",
    )
    ax[0, 2].scatter(
        pos[:, 0], pos[:, 1], s=25, c=val[:, 0], cmap="viridis", marker="o"
    )
    ax[1, 2].imshow(
        np.zeros_like(PM),
        clim=[0, 1],
        extent=[*lim_x, *lim_y],
        origin="lower",
        aspect="equal",
    )
    ax[1, 2].scatter(
        pos[:, 0], pos[:, 1], s=25, c=val[:, 1], cmap="viridis", marker="o"
    )
    ax[0, 2].scatter(pos[-1, 0], pos[-1, 1], s=25, c="r", marker="o")
    ax[0, 2].plot(pos[:, 0], pos[:, 1], c="w", alpha=0.5)
    ax[1, 2].scatter(pos[-1, 0], pos[-1, 1], s=25, c="r", marker="o")
    ax[1, 2].plot(pos[:, 0], pos[:, 1], c="w", alpha=0.5)

    ax[2, 0].set_title(f"Aq func {aqf.max():.2f}")
    ax[2, 0].imshow(
        aqf,
        extent=[*lim_x, *lim_y],
        origin="lower",
        clim=np.quantile(aqf, (0.01, 0.99)),
        aspect="equal",
    )
    if old_aqf is not None:
        diff = old_aqf - aqf
        ax[2, 1].set_title("aqf changes")
        ax[2, 1].imshow(
            diff, extent=[*lim_x, *lim_y], origin="lower", cmap="bwr", aspect="equal"
        )
    if last_spectrum is not None:
        ax[2, 2].imshow(
            last_spectrum,
            clim=np.quantile(last_spectrum, (0.02, 0.98)),
            origin="lower",
            cmap="terrain",
            aspect="equal",
        )
    # ax[i,0].figure.canvas.draw()
    # ax[i,1].figure.canvas.draw()

    plt.pause(0.01)
    return fig, aqf


if __name__ == "__main__":
    pass
