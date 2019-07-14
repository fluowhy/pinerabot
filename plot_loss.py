import numpy as np
import matplotlib.pyplot as plt


def plot_loss(dpi=500):
    losses = np.load("losses.npy")
    plt.clf()
    plt.plot(losses[:, 0], color="navy", label="train")
    plt.plot(losses[:, 1], color="red", label="test")
    plt.plot(losses[:, 2], color="green", label="val")
    plt.savefig("train_curve", dpi=dpi)
    return


if __name__ == "__main__":
    plot_loss()