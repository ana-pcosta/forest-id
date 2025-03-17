from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np


class ImageHandler:
    def __init__(self):
        pass

    def data_folders_to_df(self, path: str):
        """
        Read a folder of images into a dataframe.

        Parameters:
        base_folder (str): Path to the base folder containing image folders

        Returns:
        pandas.DataFrame: DataFrame containing image paths, IDs, and folder names (species)
        """

        base_path = Path(path)
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

        # Find all image files in all subdirectories
        image_files = [
            f for ext in image_extensions for f in base_path.glob(f"**/*{ext}")
        ]

        # Create dataframe directly from the list of paths
        df = pd.DataFrame(
            {
                "image_path": [str(f) for f in image_files],
                "image_id": [f.stem for f in image_files],
                "species": [f.parent.name for f in image_files],
            }
        )

        return df

    def analyze_image_metadata(self, df: pd.DataFrame):
        """
        Analyze the distribution of image dimensions and intensities.

        Parameters:
        df (pandas.DataFrame): DataFrame containing image information

        Returns:
        pandas.DataFrame: Original DataFrame with added columns for width, height, and file size
        """
        widths = []
        heights = []
        min_intensities = []
        max_intensities = []
        mean_intensities = []
        std_intensities = []

        for path in df["image_path"]:
            # Get image dimensions
            with Image.open(path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)

                # Convert to grayscale if image is RGB
                if img.mode == "RGB":
                    img_gray = img.convert("L")
                else:
                    img_gray = img

                # Convert to numpy array
                img_array = np.array(img_gray)

                # Calculate stats of intensity
                mean_intensities.append(np.mean(img_array))
                std_intensities.append(np.std(img_array))
                min_intensities.append(np.min(img_array))
                max_intensities.append(np.max(img_array))

        # Add columns to the dataframe
        df["width"] = widths
        df["height"] = heights
        df["mean_intensity"] = mean_intensities
        df["std_intensity"] = std_intensities
        df["min_intensity"] = min_intensities
        df["max_intensity"] = max_intensities

        return df
