"""
main.py
-------
End-to-end pipeline:

    PID -> 4 motor RPMs -> PyBullet drone -> collect data
        -> DMDc (SVD) -> A, B -> prediction -> RMSE -> plots

Run with:

    python main.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from data_collection import collect_data
from dmdc import fit_dmdc, predict, rmse


DATA_DIR = "data"
MODEL_DIR = "models"
PLOT_DIR = "plots"


def ensure_dirs():
    for d in (DATA_DIR, MODEL_DIR, PLOT_DIR):
        os.makedirs(d, exist_ok=True)


def main():
    ensure_dirs()

    # ---------------------------------------------------------------
    # 1. Collect flight data using the PID controller in PyBullet
    # ---------------------------------------------------------------
    print("Collecting flight data with PID controller...")
    X, X_next, U = collect_data(num_steps=3000)

    np.savez(
        os.path.join(DATA_DIR, "collected_data.npz"),
        X=X, X_next=X_next, U=U,
    )
    print(f"Collected {X.shape[1]} samples "
          f"(state dim {X.shape[0]}, input dim {U.shape[0]}).")

    # ---------------------------------------------------------------
    # 2. Identify the linear model with SVD-based DMDc
    # ---------------------------------------------------------------
    print("Fitting DMDc model via SVD...")
    A, B, s = fit_dmdc(X, X_next, U)
    print(f"A shape: {A.shape}, B shape: {B.shape}")

    np.savez(
        os.path.join(MODEL_DIR, "dmdc_model.npz"),
        A=A, B=B, singular_values=s,
    )

    # ---------------------------------------------------------------
    # 3. Validate: predict a held-out portion of the trajectory
    # ---------------------------------------------------------------
    split = X.shape[1] // 2
    x0 = X[:, split]
    U_test = U[:, split:]
    X_true = np.hstack([X[:, split:split + 1], X_next[:, split:]])

    X_pred = predict(A, B, x0, U_test)

    error = rmse(X_true, X_pred)
    print(f"Prediction RMSE over held-out trajectory: {error:.6f}")

    # ---------------------------------------------------------------
    # 4. Plots
    # ---------------------------------------------------------------
    plot_positions(X_true, X_pred)
    plot_velocities(X_true, X_pred)
    plot_singular_values(s)

    print(f"Plots saved in '{PLOT_DIR}/', model saved in '{MODEL_DIR}/', "
          f"data saved in '{DATA_DIR}/'.")


def plot_positions(X_true, X_pred):
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i in range(3):
        axes[i].plot(X_true[i, :], label="true")
        axes[i].plot(X_pred[i, :], "--", label="predicted")
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
    axes[-1].set_xlabel("time step")
    fig.suptitle("Position: true vs DMDc prediction")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "position.png"))
    plt.close(fig)


def plot_velocities(X_true, X_pred):
    labels = ["vx", "vy", "vz"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for i in range(3):
        axes[i].plot(X_true[i + 3, :], label="true")
        axes[i].plot(X_pred[i + 3, :], "--", label="predicted")
        axes[i].set_ylabel(labels[i])
        axes[i].legend()
    axes[-1].set_xlabel("time step")
    fig.suptitle("Velocity: true vs DMDc prediction")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "velocity.png"))
    plt.close(fig)


def plot_singular_values(s):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(s, "o-")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value (log scale)")
    ax.set_title("Singular values of Omega = [X; U]")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "singular_values.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
