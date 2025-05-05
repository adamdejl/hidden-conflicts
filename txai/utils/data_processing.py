import numpy as np
import pandas as pd
import warnings

from sklearn.base import OneToOneFeatureMixin, TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.utils import resample


class IdentityTransformer(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        X = self._validate_data(
            X,
            reset=True,
        )
        return self

    def transform(self, X):
        X = self._validate_data(
            X,
            reset=False,
        )
        return X

    def inverse_transform(self, X):
        X = self._validate_data(
            X,
            reset=False,
        )
        return X

class InvertibleColumnTransformer(ColumnTransformer):
    def inverse_transform(self, X):
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        if X.shape[1] != len(self.get_feature_names_out()):
            raise ValueError(
                "X and the fitted transformer have different numbers of columns"
            )

        inverted_X_base = np.zeros((X.shape[0], self.n_features_in_))
        columns = [c for cs in self._columns for c in cs]
        inverted_X = pd.DataFrame(data=inverted_X_base, columns=columns)
        for name, indices in self.output_indices_.items():
            transformer = self.named_transformers_.get(name, None)
            if transformer is None:
                continue

            selected_X = X[:, indices.start : indices.stop]
            if isinstance(transformer, OneHotEncoder):
                # Assumed only one column changing encoder at the end
                categories = transformer.inverse_transform(selected_X)
                inverted_X.loc[
                    :, columns[indices.start : indices.start + len(categories[0])]
                ] = categories
            elif isinstance(transformer, IdentityTransformer):
                # Identity transformer
                inverted_X.loc[
                    :, [columns[i] for i in range(indices.start, indices.stop)]
                ] = selected_X
            else:
                # Assumed scaler-type transformer
                inverted_X.loc[
                    :, [columns[i] for i in range(indices.start, indices.stop)]
                ] = transformer.inverse_transform(selected_X)

        return inverted_X


def preprocess_train_data(
    df,
    scaled_features=None,
    categorical_features=None,
    scaler=RobustScaler(quantile_range=(10, 90)),
    categorical_encoder=OneHotEncoder(handle_unknown="ignore"),
):
    if scaled_features is None and categorical_features is None:
        warnings.warn("No features specified for preprocessing, using raw data.")
        scaled_features = []
        categorical_features = []
    elif scaled_features is None:
        scaled_features = [c for c in df.columns if c not in categorical_features]
    elif categorical_features is None:
        categorical_features = [c for c in df.columns if c not in scaled_features]

    preprocessor = InvertibleColumnTransformer(
        transformers=[
            ("num", scaler, scaled_features),
            ("cat", categorical_encoder, categorical_features),
        ],
        remainder="passthrough",
    )

    preprocessed_df = preprocessor.fit_transform(df)
    return preprocessed_df, preprocessor


def preprocess_test_data(df, preprocessor):
    preprocessed_df = preprocessor.transform(df)
    return preprocessed_df


def upsample_dataset(df, label_name, target_positive_share=0.5, seed=42):
    neg_df = df[~df[label_name]]
    num_neg = len(neg_df)
    pos_df = df[df[label_name]]
    num_pos = len(pos_df)
    num_total = num_neg + num_pos

    if (num_pos / num_total) > target_positive_share:
        # Upsample the negative class
        num_extra = int(
            (num_pos - num_total * target_positive_share) / target_positive_share
        )
        extra_df = resample(
            neg_df, replace=True, n_samples=num_extra, random_state=seed
        )
    else:
        # Upsample the positive class
        num_extra = int(
            (num_total * target_positive_share - num_pos) / (1 - target_positive_share)
        )
        extra_df = resample(
            pos_df, replace=True, n_samples=num_extra, random_state=seed
        )

    resampled_df = pd.concat([df, extra_df]).sample(frac=1, random_state=seed + 1)
    return resampled_df
