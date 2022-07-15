import click
import logging
import os
import pandas as pd

from networkx import Graph
from networkx.algorithms.components.connected import connected_components
from random import sample
from typing import Dict, List, Set

from sample_detection.scrape.load import load_sample_info


def split_data(
    df: pd.DataFrame, test_size: float, val_size: float
) -> Dict[str, pd.DataFrame]:

    """Given a dataframe of samples, split it into test, validation and training sets.

    Note: This doesn't naively sample rows from the data frame. To avoid data leakage (if a song
    has been sampled a lot, it may appear in more than 1 of the test/train/val sets), we construct
    a graph (songs are nodes, with edges between songs if one has sampled the other) and randomly
    sample connected components from this graph. As a result, the proportions of the test/val/train
    sets are not exact, but are approximate.

    :param df: Dataframe to partition into test/validation/training sets
    :type df: pd.DataFrame

    :param test_size: Approximate proportion of df to allocate to the test set
    :type test_size: float

    :param val_size: Approximate proportion of df to allocate to the validation set
    :type val_size: float

    :return: Dict with dataframes of sample info for the train/val/test sets.
    :rtype: Dict[str, pd.DataFrame]

    """

    def df_from_components(
        df: pd.DataFrame, components: List[Set[str]]
    ) -> pd.DataFrame:

        """Create a dataframe from the connected components in the graph.

        :param df: Dataframe of all samples
        :type df: pd.DataFrame

        :param components: A list of connected components
        :type test_size: List[Set[str]]

        :return: df, with only the samples in the connected components
        :rtype: pd.DataFrame
        """

        component_ids = {x for set in components for x in set}
        rows = df["sample_in_ytid"].isin(component_ids) | df["sample_from_ytid"].isin(
            component_ids
        )
        return df[rows]

    edges = df[["sample_in_ytid", "sample_from_ytid"]].to_records(index=False).tolist()

    # Construct graph between songs:
    G = Graph(edges)
    components = [component for component in connected_components(G)]
    n_components = len(components)

    # nx orders connected components by size, so we need to shuffle the list of components before randomly sampling:
    components = sample(components, n_components)

    # split into test/val/train:
    test_components = components[0 : int(n_components * test_size)]
    validation_components = components[
        int(n_components * test_size) : int(n_components * (test_size + val_size))
    ]
    train_components = components[int(n_components * (test_size + val_size)) :]

    return {
        "train": df_from_components(df, train_components),
        "validate": df_from_components(df, validation_components),
        "test": df_from_components(df, test_components),
    }


@click.command()
@click.option("--test-size", default=0.15)
@click.option("--val-size", default=0.15)
@click.option("--direct-samples-only", default=True)
@click.argument("sample_info_path", type=click.Path())
@click.argument("save_dir", type=click.Path())
def main(sample_info_path, save_dir, test_size, val_size, direct_samples_only):

    logger = logging.getLogger(__name__)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df = load_sample_info(sample_info_path)

    if direct_samples_only:
        samples_to_keep = df["sample_type"].str.contains("Direct")

    logger.info(
        "Ensuring that there is a one-to-one correspondence between samples in songs..."
    )
    sample_times_match = [
        len(row["sample_from_times"]) == len(row["sample_in_times"])
        for _, row in df.iterrows()
    ]
    samples_to_keep = samples_to_keep & pd.Series(sample_times_match)

    df = df[samples_to_keep].copy()
    df.to_csv(os.path.join(save_dir, "all_samples.csv"), index=False)

    logger.info("Dividing data into train/val/test splits...")
    data_splits = split_data(df, test_size, val_size)
    for data_use, df in data_splits.items():
        df.to_csv(os.path.join(save_dir, f"{data_use}.csv"), index=False)

    logger.info("Done!")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
