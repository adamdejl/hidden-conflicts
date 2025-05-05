import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from txai.utils import preprocess_train_data, preprocess_test_data, IdentityTransformer


class TitanicDataset(Dataset):
    __create_key = object()

    @classmethod
    def create_datasets(
        cls,
        label_name="Survived",
        split_seed=43,
        test_size=0.2,
        simple=False,
        simple_drop_columns=[]
    ):
        train_dataset = TitanicDataset(
            cls.__create_key,
            label_name=label_name,
            split_seed=split_seed,
            test_size=test_size,
            simple=simple,
            simple_drop_columns=simple_drop_columns,
            train=True,
        )
        test_dataset = TitanicDataset(
            cls.__create_key,
            label_name=label_name,
            split_seed=split_seed,
            test_size=test_size,
            simple=simple,
            simple_drop_columns=simple_drop_columns,
            train=False,
        )
        return train_dataset, test_dataset

    def __init__(
        self,
        create_key=None,
        label_name="Survived",
        split_seed=43,
        test_size=0.2,
        simple=False,
        simple_drop_columns=[],
        train=True,
    ):
        # Ensure that the dataset is being constructed properly
        if create_key != TitanicDataset.__create_key:
            raise ValueError(
                "Illegal initialisation attempt — please use create_datasets to initialise."
            )

        # Load the preprocessed simple MIMIC dataframe
        if os.environ.get("TXAI_DATA_DIR"):
            data_dir_path = os.environ.get("TXAI_DATA_DIR")
        else:
            # Attempt to auto-discover the data directory
            current_dir = os.getcwd()
            parent_dir = os.path.dirname(current_dir)
            data_dir_path = None

            while parent_dir != current_dir:
                data_dir = os.path.join(current_dir, "data")
                if os.path.isdir(data_dir):
                    data_dir_path = data_dir
                    break
                current_dir = parent_dir
                parent_dir = os.path.dirname(current_dir)

            if data_dir_path is None:
                raise ValueError(
                    "Cannot find location of the data directory, specify one manually"
                )
        try:
            data_df = pd.read_csv(os.path.join(data_dir_path, "titanic_train.csv"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Titanic data file not found at identified data directory {data_dir_path}"
            )

        # Split the dataset into train and test
        if simple:
            # Drop outliers for simple dataset
            data_df = data_df[data_df.Fare < 500]
        x = data_df.drop(columns=[label_name, "PassengerId", "Name", "Ticket", "Cabin", "Embarked"])
        x[['Age']] = x[['Age']].fillna(x[['Age']].median())
        if simple:
            x = x.drop(columns=["SibSp", "Parch", "Pclass"])
            x = x.drop(columns=simple_drop_columns)
        y = data_df[label_name]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=split_seed, shuffle=True
        )

        # Preprocess the data
        x_train_processed, preprocessor = preprocess_train_data(
            x_train, categorical_features=["Sex"]
        )
        x_train = pd.DataFrame(
            x_train_processed, columns=preprocessor.get_feature_names_out()
        )
        x_test_processed = preprocess_test_data(x_test, preprocessor)
        x_test = pd.DataFrame(
            x_test_processed, columns=preprocessor.get_feature_names_out()
        )

        # Select data partition and convert to tensors
        if train:
            samples = x_train
            labels = y_train
        else:
            samples = x_test
            labels = y_test
        self.raw_samples = torch.tensor(samples.to_numpy(), dtype=torch.float32)
        if simple:
            self.samples = self.raw_samples[:, :-1]
        else:
            self.samples = self.raw_samples
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long)
        self.features = preprocessor.get_feature_names_out()
        self.preprocessor = preprocessor
        self.simple = simple

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
