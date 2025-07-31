import json
import os
import numpy as np
from ..MMFProblem.mmf import MMFfunction
from .integration import EvolutionaryAlgorithm
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

for i in [5]:
    #Read in Data
    json_path = "src\\fronts_combined_makeMMF11Function\\gen_099.json"

    with open(json_path, "r", encoding="utf-8") as fp:
        gen_data = json.load(fp)

    
    #Initialise Evolutionary Algorithm
    Algo = EvolutionaryAlgorithm("NSGA2", 2, 2, "makeMMF11Function")

    X_ip2 = np.asarray(gen_data["ip2"])
    X_nsga2 = np.asarray(gen_data["nsga2"])

    #Compute Objective Values
    mnf1 = MMFfunction(Algo, smoofname="makeMMF11Function")
    F_ip2 = mnf1.evaluate(X_ip2, return_values_of=["F"])
    F_nsga2 = mnf1.evaluate(X_nsga2, return_values_of=["F"])

    #Compute Nondomination Ranks
    nds   = NonDominatedSorting()
    ranks = np.zeros(len(F_ip2), dtype=int)
    ranks_nsga2 = np.zeros(len(F_nsga2), dtype=int)
    for r, idx in enumerate(nds.do(F_nsga2, only_non_dominated_front=False)):
        ranks_nsga2[idx] = r + 1

    #Draw 3D Plots
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")

    # Set the color normalization and colormap
    norm = colors.PowerNorm(gamma=0.4,
                            vmin=ranks.min(), vmax=ranks.max())
    cmap = plt.get_cmap("plasma")

    # Create the scatter plot
    sc = ax.scatter(X_ip2[:, 1],  
                    X_ip2[:, 0],  
                    ranks,
                    c=ranks, cmap=cmap, norm=norm,
                    s=45, edgecolor="k", linewidths=0.25)

    # Set the view angle
    ax.view_init(elev=25, azim=255)


    # Set axis labels and title
    ax.set_xlabel(r"$x_2$", labelpad=8)
    ax.set_ylabel(r"$x_1$", labelpad=10)
    ax.set_zlabel("Nondomination Rank", labelpad=8)
    ax.set_title(f"Gen 2 · IP2 – Decision Space (γ = 0.4)")

    cbar = fig.colorbar(sc, pad=0.08)
    cbar.set_label("Rank")
    # Show the plot
    plt.tight_layout()
    plt.show()

    # Draw 3D Plots for NSGA-II
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")

    # Set the color normalization and colormap
    norm = colors.PowerNorm(gamma=0.4,
                            vmin=ranks.min(), vmax=ranks.max())
    cmap = plt.get_cmap("plasma")                     

    # Create the scatter plot for NSGA-II
    sc = ax.scatter(X_nsga2[:, 1], X_nsga2[:, 0], ranks_nsga2,
                    c=ranks_nsga2, cmap=cmap, norm=norm,
                    s=50, edgecolor="k", linewidths=0.3)

    # Set the view angle
    ax.view_init(elev=25, azim=255)

    # Set axis labels and title
    ax.set_xlabel(r"$x_2$", labelpad=8)
    ax.set_ylabel(r"$x_1$", labelpad=10)
    ax.set_zlabel("Nondomination Rank", labelpad=8)
    ax.set_title(f"Gen 2 · NSGA-II – Decision Space (γ = 0.4)")

    cbar = fig.colorbar(sc, pad=0.1)
    cbar.set_label("Nondomination Rank")

    # Show the plot
    plt.tight_layout()
    plt.show()