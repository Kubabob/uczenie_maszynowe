#!/usr/bin/env python
# coding: utf-8

# %% In[1]:
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split


# %% In[2]:
def RMSE(y_hat: pl.Series, y_obs: pl.DataFrame) -> float:
    return (((y_hat - y_obs.to_series()) ** 2).sum() / y_obs.shape[0]) ** 0.5


# %% In[3]:
input_matrix = pl.read_excel(source="dane_leki.xlsx", engine="openpyxl")


# %% In[4]:
input_matrix.head()


# %% In[5]:
Y_obs = input_matrix.select(pl.nth(1))


# %% In[6]:
Y_obs.head()


# %% In[7]:
descriptors = input_matrix.select(pl.nth([2, 3, 4, 5]))


# %% In[8]:
descriptors.head()


# %% In[9]:
pca_model = PCA(n_components=4)


# %% In[10]:
pca_model.fit(descriptors)


# %% In[11]:
pca_model.explained_variance_


# %% In[12]:
pca_model.components_


# %% In[19]:
PC = pl.DataFrame(
    pca_model.fit_transform(descriptors),
    schema=[f"PC{i+1}" for i in range(pca_model.n_components_)],
)


# %% In[20]:
PC


# %% In[17]:
X_training, X_validation, Y_training, Y_validation = train_test_split(
    PC, Y_obs, test_size=0.33, random_state=42
)


# %% In[18]:
X_training


# %% In[19]:
Y_training


# %% In[20]:
KFold_model = KFold(n_splits=10, shuffle=True, random_state=0)


# %% In[21]:
validation_sets = []
for training_set, validation_set in KFold_model.split(X_training, Y_training):
    validation_sets.append(validation_set)
    print(f"Training: {training_set}\nValidation: {validation_set}")


# %% In[22]:
descriptors


# %% In[71]:
LinearRegression().fit(
    X=X_training.with_row_index()
    .filter(~pl.col("index").is_in(validation_set))
    .drop(pl.col("index")),
    y=Y_training.with_row_index()
    .filter(~pl.col("index").is_in(validation_set))
    .drop(pl.col("index")),
)


# %% In[88]:
def RMSE_cv() -> pl.Float64:
    residues = pl.Series(dtype=pl.Float64)

    for idx, validation_set in enumerate(validation_sets):
        x = (
            X_training.with_row_index()
            .filter(~pl.col("index").is_in(validation_set))
            .drop(pl.col("index"))
        )

        y = (
            Y_training.with_row_index()
            .filter(~pl.col("index").is_in(validation_set))
            .drop(pl.col("index"))
        )

        PCR_model = LinearRegression().fit(X=x, y=y)

        prediction = PCR_model.predict(
            X_training.with_row_index()
            .filter(pl.col("index").is_in(validation_set))
            .drop(pl.col("index"))
        )

        references = (
            Y_training.with_row_index()
            .filter(pl.col("index").is_in(validation_set))
            .drop(pl.col("index"))
        )

        residues.append(pl.Series((prediction - references).flatten()))

    return np.sqrt((residues**2).sum())


# %%
RMSE_cv()


# %%
def RMSE_c():
    x


# %%
def principal_component_plot(n: int):
    pca_model = PCA(n_components=n)

    PC = pl.DataFrame(
        pca_model.fit_transform(descriptors),
        schema=[f"PC{i+1}" for i in range(pca_model.n_components_)],
    )

    KFold_model = KFold(n_splits=10, shuffle=True, random_state=0)

    X_training, _, Y_training, _ = train_test_split(
        PC, Y_obs, test_size=0.33, random_state=42
    )

    validation_sets = [
        validation_set
        for (_, validation_set) in KFold_model.split(X_training, Y_training)
    ]

    _RMSE_cv = RMSE_cv()

    RMSE_c
