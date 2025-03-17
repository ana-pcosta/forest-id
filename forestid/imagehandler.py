from pathlib import Path
import pandas as pd
from forestid import ROOT_PATH

def data_folders_to_df(path:str):
    """
    Read a folder of images into a dataframe.
    
    Parameters:
    base_folder (str): Path to the base folder containing image folders
    
    Returns:
    pandas.DataFrame: DataFrame containing image paths, IDs, and folder names (species)
    """

    base_path = Path(base_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Find all image files in all subdirectories
    image_files = [f for ext in image_extensions for f in base_path.glob(f'**/*{ext}')]
    
    # Create dataframe directly from the list of paths
    df = pd.DataFrame({
        'image_path': [str(f) for f in image_files],
        'image_id': [f.stem for f in image_files],
        'species': [f.parent.name for f in image_files]
    })
    
    return df