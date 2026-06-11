"""This ETL transforms the bug issue prediction dataset to a standard format."""

import logging
import pickle
import time

import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    """Main function to run the ETL process"""

    logger.info("Reading the original files...")
    # WARNING: Pandas<2.0.0 is required to read the pickle files
    with open("./train_data_all.p", "rb") as f:
        train_dict = pickle.load(f)
    with open("./test_data_all.p", "rb") as f:
        test_dict = pickle.load(f)

    train_list = []
    for project_name, values in train_dict.items():
        values["project"] = project_name
        values["id"] = values["id"].astype(str)
        values["label"] = values["classification"].astype(int)
        values.drop(columns=["classification", "discussion"], inplace=True)
        train_list.append(values)
    train_df = pd.concat(train_list, ignore_index=True)

    test_list = []
    for project_name, values in test_dict.items():
        values["project"] = project_name
        values["id"] = values["id"].astype(str)
        values["label"] = values["classification"].astype(int)
        values.drop(columns=["classification", "discussion"], inplace=True)
        test_list.append(values)
    test_df = pd.concat(test_list, ignore_index=True)

    logger.info("Total rows to write for the train dataset: %d ", train_df.shape[0])
    logger.info("Total rows to write for the test dataset: %d ", test_df.shape[0])

    train_df.to_csv("./train_data_all.csv", index=False)
    test_df.to_csv("./test_data_all.csv", index=False)


if __name__ == "__main__":
    start_time = time.time()
    main()
    logger.info("Execution time: %.2f seconds", time.time() - start_time)
