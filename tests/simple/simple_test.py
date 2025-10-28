from pathlib import Path

import numpy as np
import pandas as pd

from core.models.regression import ChoquisticRegression


def test_choquistic_regression_runs_on_diabetes_dataset():
	"""Fit and evaluate ChoquisticRegression on the diabetes dataset."""
	data_path = Path(__file__).resolve().parents[2] / "data" / "diabetes.csv"
	df = pd.read_csv(data_path)

	X = df.drop(columns="Outcome")
	y = df["Outcome"].to_numpy()

	model = ChoquisticRegression(
		representation="shapley",
		k_add=8,
		C=10.0,
		max_iter=500,
		random_state=0,
	)

	params = model.get_params()
	assert params["representation"] == "shapley"
	assert params["k_add"] == 8
	assert params["C"] == 10.0

	model.set_params(C=5.0, solver="liblinear")
	assert model.C == 5.0
	assert model.solver == "liblinear"
	assert model.get_params()["C"] == 5.0
	assert model.get_params()["solver"] == "liblinear"

	model.fit(X, y)

	coefficients = model.model_.coef_[0]
	assert np.isfinite(coefficients).all()

	preds = model.predict(X)
	probas = model.predict_proba(X)

	score = model.score(X, y)

	assert preds.shape == y.shape
	assert set(np.unique(preds)).issubset({0, 1})
	assert probas.shape == (len(X), 2)
	assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)
	assert np.isfinite(probas).all()
	assert (preds == y).mean() > 0.7
	assert 0.7 < score <= 1.0
	assert np.array_equal(model.model_.classes_, np.array([0, 1]))

