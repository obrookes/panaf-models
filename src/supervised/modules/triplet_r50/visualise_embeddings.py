# core libraries
import os
import sys
import cv2
import argparse
import numpy as np

# Matplotlib / TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects


# Define our own plot function
def scatter(x, labels, filename):
    # Get the number of classes (number of unique labels)
    num_classes = int(np.amax(np.unique(labels)))
    print(f"Number of classes: {num_classes}")

    # Choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_classes + 1))

    print(palette)
    print(np.shape(palette))

    labels = [int(x) for x in labels]

    # Map the colours to different labels
    label_colours = np.array([palette[int(i)] for i in labels])

    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")

    # Plot the points

    ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=label_colours, marker="o")

    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    ax.axis("off")
    ax.axis("tight")
    plt.tight_layout()

    # Save it to file
    plt.savefig(filename + ".pdf")


def alt_scatter(x, labels, filename):

    classes = [
        "camera_interaction",
        "climbing_down",
        "climbing_up",
        "hanging",
        "running",
        "sitting",
        "sitting_on_back",
        "standing",
        "walking",
    ]

    # Get the number of classes (number of unique labels)
    num_classes = np.unique(labels).shape[0]

    # Choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", num_classes + 1))

    # Map the colours to different labels
    label_colours = np.array([palette[int(labels[i])] for i in range(labels.shape[0])])

    # Function to get class
    def get_class(x):
        return classes[int(x)]

    # Create our figure/plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect="equal")

    component_one = x[:, 0]
    component_two = x[:, 1]

    for i, u in enumerate(list(map(int, np.unique(labels)))):
        xi = [
            component_one[j] for j in range(len(component_one)) if int(labels[j]) == u
        ]
        yi = [
            component_two[j] for j in range(len(component_two)) if int(labels[j]) == u
        ]
        plt.scatter(xi, yi, c=palette[i], label=get_class(u))

    ax.legend()

    plt.title("Reciprocal Softmax Triplet loss")
    # Do some formatting
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis("off")
    ax.axis("tight")
    plt.tight_layout()

    # Save it to file
    plt.savefig(f"{filename}.pdf")

    # Show plt
    plt.show()


# Load and visualise embeddings via t-SNE
def plotEmbeddings(args):
    # Ensure there's something there
    if not os.path.exists(args.embeddings_file):
        print(f"No embeddings file at path: {args.embeddings_file}, exiting.")
        sys.exit(1)

    # Load the embeddings into memory
    embeddings = np.load(args.embeddings_file)

    print("Loaded embeddings")

    # Visualise the learned embedding via t-SNE
    visualiser = TSNE(n_components=2, perplexity=args.perplexity)

    # Reduce dimensionality
    reduction = visualiser.fit_transform(embeddings["embeddings"])

    print("Visualisation computed")

    # Show results
    alt_scatter(
        reduction, embeddings["labels"], os.path.basename(args.embeddings_file)[:-4]
    )


# Main/entry method
if __name__ == "__main__":
    # Collate command line arguments
    parser = argparse.ArgumentParser(
        description="Parameters for visualising the embeddings via TSNE"
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        required=True,
        help="Path to embeddings .npz file you want to visalise",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="Perplexity parameter for t-SNE, consider values between 5 and 50",
    )
    args = parser.parse_args()

    # Let's plot!
    plotEmbeddings(args)
