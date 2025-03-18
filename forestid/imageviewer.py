from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import numpy as np


def plot_random_images(df, n=1, with_info=True):
    """
    Plot a random image from the dataframe.

    Parameters:
    df (pandas.DataFrame): DataFrame containing image information
    with_info (bool): Whether to display image information

    Returns:
    int: Index of the randomly selected row
    """
    # Select a random row
    random_row = df.sample(n)

    # Create plot
    _, ax = plt.subplots(1, n, figsize=(n * 5, 4))
    if n == 1:
        ax = [ax]

    for ix in range(0, n):
        # Get the image path
        img_path = random_row["image_path"].values[ix]
        img = Image.open(img_path)
        ax[ix].imshow(img)
        ax[ix].axis("off")

        if with_info:
            # Get image dimensions
            width, height = img.size

            # Prepare information text
            info_text = (
                f"Image ID: {random_row['image_id'].values[ix]}\n"
                f"Species: {random_row['species'].values[ix]}\n"
                f"Dimensions: {width}x{height} pixels\n"
            )

            # Add information text as title
            ax[ix].set_title(info_text, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_metadata_distribution(
    df: pd.DataFrame,
    columns=[
        "height",
        "width",
        "mean_intensity",
        "std_intensity",
        "max_intensity",
        "min_intensity",
    ],
):
    """
    Plot distributions of image sizes and intensities.

    Parameters:
    df (pandas.DataFrame): DataFrame with image analysis data
    """
    # Create a figure with multiple subplots
    num_cols = len(columns)
    num_rows = math.ceil(num_cols / 3)
    fig, axes = plt.subplots(
        num_rows, min(num_cols, 3), figsize=(6 * min(num_cols, 3), 6 * num_rows)
    )
    # Ensure axes is always a 1D array
    if num_rows == 1:
        axes = np.array(axes).flatten()
    else:
        axes = axes.flatten()

    for i, col in enumerate(columns):
        if col in df.columns:
            sns.boxplot(df[col], ax=axes[i])
            axes[i].set_title(f'{col.replace("_", " ").title()} Distribution')
            axes[i].set_xlabel(col.replace("_", " ").title())

    plt.tight_layout()
    plt.show()


def plot_random_predictions(df: pd.DataFrame, n=9, missclassified=True):
    # Create a figure with multiple subplots

    if missclassified:
        df = df[df["gt"] != df["prediction"]]
        n = min([len(df), n])

    num_cols = n
    num_rows = math.ceil(num_cols / 3)
    fig, axes = plt.subplots(
        num_rows, min(num_cols, 3), figsize=(6 * min(num_cols, 3), 6 * num_rows)
    )

    print(num_cols, num_rows)
    # Select a random row
    random_row = df.sample(n)

    # Ensure axes is always a 1D array
    if num_rows == 1:
        axes = np.array(axes).flatten()
    else:
        axes = axes.flatten()

    for i in range(n):
        img_path = random_row["image_path"].values[i]
        img = Image.open(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")

        info_text = (
            f"Image ID: {random_row['image_id'].values[i]}\n"
            f"GT: {random_row['gt'].values[i]}\n"
            f"Prediction: {random_row['prediction'].values[i]}\n"
        )

        axes[i].set_title(info_text)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(df: pd.DataFrame, confusion_matrix):
    class_names = df["gt"].unique()

    # Convert the confusion matrix into a DataFrame with class names
    conf_matrix_df = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names
    )

    # Plot the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        annot_kws={"size": 16},
        linewidths=1,
        linecolor="black",
    )

    # Add labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


def plot_train_losses(train_losses: list, val_losses: list):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.plot(val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
