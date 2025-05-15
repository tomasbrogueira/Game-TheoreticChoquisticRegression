import pandas as pd
import numpy as np
from typing import List, Union

# -------------------------------------------------------------------
# Section 1: Bias and Scaling Operations
# -------------------------------------------------------------------

def add_bias(
    df: pd.DataFrame,
    features: List[Union[str, int]],
    bias: float
) -> pd.DataFrame:
    """
    Add a constant bias to the specified features.
    """
    df_out = df.copy()
    for f in features:
        if isinstance(f, int):
            df_out.iloc[:, f] = df_out.iloc[:, f] + bias
        else:
            df_out.loc[:, f] = df_out.loc[:, f] + bias
    return df_out

def scale_features(
    df: pd.DataFrame,
    features: List[Union[str, int]],
    factor: float
) -> pd.DataFrame:
    """
    Multiply specified features by a given factor.
    """
    df_out = df.copy()
    for f in features:
        if isinstance(f, int):
            df_out.iloc[:, f] = df_out.iloc[:, f] * factor
        else:
            df_out.loc[:, f] = df_out.loc[:, f] * factor
    return df_out

# -------------------------------------------------------------------
# Section 2: Noise Injection
# -------------------------------------------------------------------

def add_gaussian_noise(
    df: pd.DataFrame,
    features: List[Union[str, int]],
    std: float,
    random_state: int = None
) -> pd.DataFrame:
    """
    Add zero-mean Gaussian noise with given std to selected features.
    """
    rng = np.random.default_rng(random_state)
    df_out = df.copy()
    for f in features:
        noise = rng.normal(0, std, size=df_out.shape[0])
        if isinstance(f, int):
            df_out.iloc[:, f] = df_out.iloc[:, f] + noise
        else:
            df_out.loc[:, f] = df_out.loc[:, f] + noise
    return df_out

# -------------------------------------------------------------------
# Section 3: Non-Linear Transformations
# -------------------------------------------------------------------

def log_transform(
    df: pd.DataFrame,
    features: List[Union[str, int]],
    offset: float = 1e-6
) -> pd.DataFrame:
    """
    Apply natural log to features (adds small offset to avoid log(0)).
    """
    df_out = df.copy()
    for f in features:
        if isinstance(f, int):
            df_out.iloc[:, f] = np.log(df_out.iloc[:, f] + offset)
        else:
            df_out.loc[:, f] = np.log(df_out.loc[:, f] + offset)
    return df_out

def power_transform(
    df: pd.DataFrame,
    features: List[Union[str, int]],
    exponent: float
) -> pd.DataFrame:
    """
    Raise specified features to a given power.
    """
    df_out = df.copy()
    for f in features:
        if isinstance(f, int):
            df_out.iloc[:, f] = np.power(df_out.iloc[:, f], exponent)
        else:
            df_out.loc[:, f] = np.power(df_out.loc[:, f], exponent)
    return df_out

def tanh_transform(
    df: pd.DataFrame,
    features: List[Union[str, int]]
) -> pd.DataFrame:
    """
    Apply hyperbolic tangent to specified features.
    """
    df_out = df.copy()
    for f in features:
        if isinstance(f, int):
            df_out.iloc[:, f] = np.tanh(df_out.iloc[:, f])
        else:
            df_out.loc[:, f] = np.tanh(df_out.loc[:, f])
    return df_out

# -------------------------------------------------------------------
# Section 4: Thresholding & Clipping
# -------------------------------------------------------------------

def threshold_features(
    df: pd.DataFrame,
    features: List[Union[str, int]],
    threshold: float,
    above_value: float,
    below_value: float = 0.0
) -> pd.DataFrame:
    """
    For each value in features: if val >= threshold, set to above_value; else below_value.
    """
    df_out = df.copy()
    for f in features:
        if isinstance(f, int):
            col = df_out.iloc[:, f]
            df_out.iloc[:, f] = np.where(col >= threshold, above_value, below_value)
        else:
            col = df_out.loc[:, f]
            df_out.loc[:, f] = np.where(col >= threshold, above_value, below_value)
    return df_out

def clip_features(
    df: pd.DataFrame,
    features: List[Union[str, int]],
    min_val: float,
    max_val: float
) -> pd.DataFrame:
    """
    Clip specified features to lie within [min_val, max_val].
    """
    df_out = df.copy()
    for f in features:
        if isinstance(f, int):
            df_out.iloc[:, f] = df_out.iloc[:, f].clip(min_val, max_val)
        else:
            df_out.loc[:, f] = df_out.loc[:, f].clip(min_val, max_val)
    return df_out

# -------------------------------------------------------------------
# Usage example:
# -------------------------------------------------------------------
# from operations_impact import add_bias, add_gaussian_noise
# df_biased = add_bias(df, ['X1', 'X3'], bias=0.5)
# df_noisy  = add_gaussian_noise(df, [0,2,5], std=0.1, random_state=42)