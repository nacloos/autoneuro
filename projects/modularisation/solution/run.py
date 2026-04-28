from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import GeneralDataset
from utils.graphs import analyse_connectivity
from utils.network import MODEL
from utils.plots import plot_connectivity, plot_loss, plot_spectrum, plot_test


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main():
    colors_blue = ["#b3e5fc", "#0091ea"]
    colors_red = ["#f18e86", "#e83b47"]
    colors_yellow = ["#fb9d32", "#fec787"]
    colors_purple = ["#d69bc5", "#a8509f"]
    colors_grey_orange = ["#c3c3c3", "#f8a834"]
    colors = [
        colors_blue,
        colors_red,
        colors_yellow,
        colors_purple,
        colors_grey_orange,
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", [colors[0][1], "white", colors[1][1]]
    )

    encode_num = 4
    decode_num = 4
    k = encode_num
    hidden_num = 32
    layer_num = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 145
    np.random.seed(seed)

    run_type = "nonlinear"  # "linear" for f(x) = x, or "nonlinear" for randomly generated f(x)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if run_type == "linear":

        def morph(x):
            return x

        common_settings = {
            "L2": 1e-4,
            "bias": False,
            "nonlinear": False,
            "hidden_layer": layer_num,
        }
    else:
        amps = 0.7 + 0.3 * np.random.rand(encode_num)
        fres = 0.4 + 1.4 * np.random.rand(encode_num)
        phis = 2 * np.pi * np.random.rand(encode_num)

        def morph(x):
            return amps * np.sin(fres * x + phis)

        common_settings = {
            "L2": 1e-4,
            "bias": True,
            "nonlinear": True,
            "hidden_layer": layer_num,
        }

    fig, axes = plt.subplots(1, encode_num, figsize=(9, 2), sharex=True, sharey=True)
    xx = np.linspace(-3, 3, 100)
    yy = np.array([morph([x] * encode_num) for x in xx])
    for i in range(encode_num):
        axes[i].plot(xx, yy[:, i])
    plt.close(fig)

    train_dataset = GeneralDataset(
        encode_num=encode_num, mapfun=morph, num_samples=1000
    )
    test_dataset = GeneralDataset(encode_num=encode_num, mapfun=morph, num_samples=1000)
    train_dataloader = DataLoader(train_dataset, batch_size=200)

    lr = 1e-3
    wd = 0
    num_epochs = 10000

    settings = [
        {"name": "Noise0", "noise_coef": 0.1, "noise_expt": 0},
        {"name": "Noise1", "noise_coef": 0.1, "noise_expt": 1},
        {"name": "Noise2", "noise_coef": 0.1, "noise_expt": 2},
        {"name": "L2"},
        {"name": "L1", "L1": 1e-3},
        {"name": "Act", "Act": 1e-3},
        {"name": "WeightedAct", "WeightedAct": 1e-3},
        {"name": "WeightedActSigmoid", "WeightedAct": 1e-3, "use_sigmoid": True},
        {"name": "WeightedActTanh", "WeightedAct": 1e-3, "use_tanh": True},
    ]

    for setting in settings:
        file_name = setting["name"]
        folder = RESULTS_DIR / file_name
        folder.mkdir(parents=True, exist_ok=True)
        print(f"running setting {file_name}...")

        model = MODEL(
            encode_num=encode_num,
            decode_num=decode_num,
            hidden_num=hidden_num,
            **(setting | common_settings),
        )
        model.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        losses = []
        model.train()
        for epoch in range(num_epochs):
            losses.append(
                model.fit(
                    train_dataloader,
                    train_dataset,
                    test_dataset,
                    optimizer,
                    criterion,
                    device,
                )
            )
            print(
                (
                    f"epoch {epoch:07d}, test loss {losses[-1][0]:.5f}, "
                    f"valid vloss {losses[-1][1]:.5f}, L2 norm {losses[-1][3]:.5f}"
                ),
                end="\r",
            )
        losses = np.array(losses)
        print()

        plot_test(model, test_dataset, device, morph, folder, "scatter")
        plot_loss(losses, folder, "losses")
        wmats, eig_raw, eig, idx = analyse_connectivity(model, k=encode_num)
        plot_spectrum(eig, folder, "spectrum")
        plot_connectivity(wmats, eig, idx, k, cmap, folder, "connectivity")
        torch.save(model, folder / "model.pth")
        with open(folder / "losses.npy", "wb") as f:
            np.save(f, losses)


if __name__ == "__main__":
    main()
