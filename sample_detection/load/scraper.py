import pandas as pd
from ast import literal_eval


def load_sample_info(sample_info_path: str) -> pd.DataFrame:

    """Load sample info from csv, with sample times as lists (as expected by other
    functions in this library).

    :param sample_info_path: Path to sample info
    :type sample_info_path: str

    :return: _description_
    :rtype: _type_
    """

    df = pd.read_csv(sample_info_path)
    df["sample_in_times"] = df["sample_in_times"].apply(literal_eval)
    df["sample_from_times"] = df["sample_from_times"].apply(literal_eval)
    return df
