import pandas as pd
import os

NAS_DIR = r"\\132.64.186.144\HartLabNAS\Projects\CFG"
VANILLA_FILENAME = r"vanillaMeasures.csv"


def get_vanilla(nas_dir=NAS_DIR):
    """
    Returns the most up-to-date version of the vanilla data.
    :param nas_dir: path to the CFG directory in the lab's NAS, as mapped on the user's machine.
    :return: pd.DataFrame with shape (614, 36)
    """
    return pd.read_csv(os.path.join(nas_dir, VANILLA_FILENAME))
