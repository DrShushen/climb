import matplotlib.pyplot as plt
import numpy as np

models = [
    "bnn",
    "gp",
    "qr",
    "conformal_copula",
    "conformal",
    "copula",
    "ensemble",
    "mcd",
]
metrics = ["picp", "deficet", "excess"]

group_by_dict = {
    "dist == 'normal' & noise_variance == 2": "prop",
    "noise_variance == 2": "dist",
    "dist == 'normal' & prop == 0.5 & copula_n_samples==1000": "noise_variance",
    "dist == 'normal' & prop == 0.5": "noise_variance",
}


# +
def plot_graph(
    mean_prop,
    std_prop,
    mean_var,
    std_var,
    mean_dist,
    std_dist,
    metric,
    ylabel,
):
    colormap = plt.cm.nipy_spectral
    colors = [colormap(i) for i in np.linspace(0, 1, 10)]

    tmp_color = colors[1]

    colors[1] = colors[-2]
    colors[-2] = tmp_color

    SMALL_SIZE = 20
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 20

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    if ylabel == "Coverage":
        ax[0].axhline(
            y=0.95,
            color=colors[0],
            linestyle="--",
            label="Guarantee=0.95",
            lw=5,
        )
        ax[1].axhline(
            y=0.95,
            color=colors[0],
            linestyle="--",
            label="Guarantee=0.95",
            lw=5,
        )
        ax[2].axhline(
            y=0.95,
            color=colors[0],
            linestyle="--",
            label="Guarantee=0.95",
            lw=5,
        )

    model = "conformal_copula"
    ax[0].errorbar(
        list(mean_prop.index),
        f"{metric}_{model}",
        data=mean_prop,
        yerr=std_prop[f"{metric}_{model}"],
        capsize=3,
        label="* Data-SUITE *",
        color=colors[1],
        lw=4,
    )

    for idx, model in enumerate(sorted(models)):
        if model != "conformal_copula":
            try:
                ax[0].errorbar(
                    list(mean_prop.index),
                    f"{metric}_{model}",
                    data=mean_prop,
                    yerr=std_prop[f"{metric}_{model}"],
                    capsize=3,
                    label=model.upper(),
                    color=colors[idx + 2],
                )

            except BaseException:
                pass

    model = "conformal_copula"
    ax[1].errorbar(
        list(mean_var.index),
        f"{metric}_{model}",
        data=mean_var,
        yerr=std_var[f"{metric}_{model}"],
        capsize=3,
        label=model.capitalize(),
        color=colors[1],
        lw=4,
    )

    for idx, model in enumerate(sorted(models)):
        if model != "conformal_copula":
            ax[1].errorbar(
                list(mean_var.index),
                f"{metric}_{model}",
                data=mean_var,
                yerr=std_var[f"{metric}_{model}"],
                capsize=3,
                label=model.upper(),
                color=colors[idx + 2],
            )

    model = "conformal_copula"
    ax[2].errorbar(
        list(mean_dist.index),
        f"{metric}_{model}",
        data=mean_dist,
        yerr=std_dist[f"{metric}_{model}"],
        capsize=3,
        label=model.capitalize(),
        color=colors[1],
        lw=4,
    )

    for idx, model in enumerate(sorted(models)):
        if model != "conformal_copula":
            ax[2].errorbar(
                list(mean_dist.index),
                f"{metric}_{model}",
                data=mean_dist,
                yerr=std_dist[f"{metric}_{model}"],
                capsize=3,
                label=model.upper(),
                color=colors[idx + 2],
            )

    ax[0].set_xlabel("(a) Proportion corrupted")
    ax[1].set_xlabel("(b) Noise variance")
    ax[2].set_xlabel("(c) Noise distribution")

    ax[0].set_ylabel(ylabel)

    ax[0].legend(loc=(3.5, 0.1))

    ax[0].grid()
    ax[1].grid()
    ax[2].grid()

    # Save the whole figure.
    fig.savefig(f"./results/synthetic/{ylabel}.png")

    plt.show()


# -


def get_metrics_graphs(df):
    for metric in metrics:
        cols_to_drop = []
        for i in df.columns:
            if "excess" in i:
                if "all" not in i:
                    cols_to_drop.append(i)

        query_conditions = "dist == 'normal' & noise_variance == 2"

        groupby_val = group_by_dict[query_conditions]

        sub_df = df.query(query_conditions)

        features = []

        for col in sub_df.columns:
            if metric in col:
                features.append(col)

        if groupby_val != "dist":
            features.extend(["dist", groupby_val])
        else:
            features.extend([groupby_val])
        mean_prop = sub_df[features].groupby(groupby_val).mean()
        std_prop = sub_df[features].groupby(groupby_val).std()

        query_conditions = "noise_variance == 2"

        groupby_val = group_by_dict[query_conditions]

        sub_df = df.query(query_conditions)

        features = []

        for col in sub_df.columns:
            if metric in col:
                features.append(col)

        if groupby_val != "dist":
            features.extend(["dist", groupby_val])
        else:
            features.extend([groupby_val])
        mean_dist = sub_df[features].groupby(groupby_val).mean()
        std_dist = sub_df[features].groupby(groupby_val).std()

        query_conditions = "dist == 'normal' & prop == 0.5 & copula_n_samples==1000"

        groupby_val = group_by_dict[query_conditions]

        sub_df = df.query(query_conditions)

        features = []

        for col in sub_df.columns:
            if metric in col:
                features.append(col)

        if groupby_val != "dist":
            features.extend(["dist", groupby_val])
        else:
            features.extend([groupby_val])
        mean_var = sub_df[features].groupby(groupby_val).mean()

        query_conditions = "dist == 'normal' & prop == 0.5"

        groupby_val = group_by_dict[query_conditions]

        sub_df = df.query(query_conditions)

        features = []

        for col in sub_df.columns:
            if metric in col:
                features.append(col)

        if groupby_val != "dist":
            features.extend(["dist", groupby_val])
        else:
            features.extend([groupby_val])
        std_var = sub_df[features].groupby(groupby_val).std()

        if groupby_val == "noise_variance":
            if 3 in list(std_var.index):
                std_var = std_var.drop([3])

        if metric == "picp":
            ylabel = "Coverage"
        if metric == "excess":
            ylabel = "Excess"
        if metric == "deficet":
            ylabel = "Deficet"

        plot_graph(
            mean_prop,
            std_prop,
            mean_var,
            std_var,
            mean_dist,
            std_dist,
            metric,
            ylabel=ylabel,
        )


def get_mse_table(df, column=0):
    metric = "mse"

    if column == 0:
        query_conditions = "dist == 'normal' & noise_variance == 2"
    else:
        query_conditions = "dist == 'normal' & prop == 0.5"

    groupby_val = group_by_dict[query_conditions]

    sub_df = df.query(query_conditions)

    features = []

    for col in sub_df.columns:
        if metric in col:
            features.append(col)

    if groupby_val != "dist":
        features.extend(["dist", groupby_val])
    else:
        features.extend([groupby_val])
    mean_df = sub_df[features].groupby(groupby_val).mean()
    cols_to_drop = []
    for i in mean_df.columns:
        if "large_sigma" in i or "rmse" in i:
            cols_to_drop.append(i)
    mean_df = mean_df.drop(columns=cols_to_drop)
    return mean_df
