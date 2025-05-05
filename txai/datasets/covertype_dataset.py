import os
import pandas as pd
import torch
import sklearn.datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from torch.utils.data import Dataset, DataLoader

from txai.utils import preprocess_train_data, preprocess_test_data


class CovertypeDataset(Dataset):
    __create_key = object()

    @classmethod
    def create_datasets(
        cls,
        label_name="Cover_Type",
        split_seed=43,
        test_size=0.2,
    ):
        train_dataset = CovertypeDataset(
            cls.__create_key,
            label_name=label_name,
            split_seed=split_seed,
            test_size=test_size,
            train=True,
        )
        test_dataset = CovertypeDataset(
            cls.__create_key,
            label_name=label_name,
            split_seed=split_seed,
            test_size=test_size,
            train=False,
        )
        return train_dataset, test_dataset

    def __init__(
        self,
        create_key=None,
        label_name="Cover_Type",
        split_seed=43,
        test_size=0.2,
        train=True,
    ):
        # Ensure that the dataset is being constructed properly
        if create_key != CovertypeDataset.__create_key:
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
        covertype = sklearn.datasets.fetch_covtype(
            data_home=data_dir_path,
            random_state=42,
            shuffle=False,
            as_frame=True
        )

        x = covertype['data']
        y = covertype['target'] - 1

        # Split the dataset into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=split_seed, shuffle=True
        )

        # Preprocess the data
        categorical_features = [
            'Wilderness_Area_0', 'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3',
            'Soil_Type_0', 'Soil_Type_1', 'Soil_Type_2', 'Soil_Type_3', 'Soil_Type_4',
            'Soil_Type_5', 'Soil_Type_6', 'Soil_Type_7', 'Soil_Type_8', 'Soil_Type_9',
            'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12', 'Soil_Type_13', 'Soil_Type_14',
            'Soil_Type_15', 'Soil_Type_16', 'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19',
            'Soil_Type_20', 'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24',
            'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28', 'Soil_Type_29',
            'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32', 'Soil_Type_33', 'Soil_Type_34',
            'Soil_Type_35', 'Soil_Type_36', 'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39',
        ]
        x_train_processed, preprocessor = preprocess_train_data(
            x_train,
            categorical_features=categorical_features,
            categorical_encoder=FunctionTransformer(
                lambda x: x,
                inverse_func=lambda x: x,
                feature_names_out='one-to-one'
            )
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
        self.samples = torch.tensor(samples.to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.long)
        self.features = preprocessor.get_feature_names_out()
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
