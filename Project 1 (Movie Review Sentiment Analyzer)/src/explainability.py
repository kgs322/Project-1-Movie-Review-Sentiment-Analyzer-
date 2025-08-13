# shap_explain.py
from __future__ import annotations

import numpy as np
import shap
from typing import Callable, Iterable, List, Optional, Sequence, Union

TextLike = Union[str, Sequence[str]]

def _default_tokenizer(x: str) -> List[str]:
    # Very light tokenizer if you don't pass one (spaces only).
    return x.split()

def _wrap_model_to_proba_matrix(
    model: object,
    target_label: Optional[Union[int, str]] = None
) -> Callable[[Sequence[str]], np.ndarray]:
    """
    Wrap different model types into a callable f(texts) -> (n_samples, n_classes) probs.
    Supports:
      - scikit-learn models with predict_proba
      - HuggingFace TextClassificationPipeline
      - custom callables returning prob arrays or list-of-dicts like HF
    """
    # 1) scikit-learn style
    if hasattr(model, "predict_proba"):
        def f_sklearn(texts: Sequence[str]) -> np.ndarray:
            proba = model.predict_proba(list(texts))
            proba = np.asarray(proba, dtype=float)
            if proba.ndim == 1:
                proba = proba[:, None]
            return proba
        return f_sklearn

    # 2) HuggingFace pipeline style (TextClassificationPipeline)
    #    We call with return_all_scores=True to get a list of {label, score}.
    #    We'll map labels to columns in a stable order.
    try:
        from transformers.pipelines import TextClassificationPipeline  # type: ignore
        if isinstance(model, TextClassificationPipeline):
            # Discover label set and order by running on a tiny sample:
            sample = ["__label_probe__"]
            results = model(sample, return_all_scores=True, truncation=True)
            # results: List[List[{'label': 'NEGATIVE', 'score': 0.5}, ...]]
            labels = [d["label"] for d in results[0]]
            label_to_idx = {lab: i for i, lab in enumerate(labels)}

            # If target_label is an int, ensure it's in range; if str, ensure exists:
            if isinstance(target_label, str) and target_label not in label_to_idx:
                # Fall back silently to whatever labels came from the model
                pass

            def f_hf(texts: Sequence[str]) -> np.ndarray:
                outs = model(list(texts), return_all_scores=True, truncation=True)
                # Build (n_samples, n_classes) in the fixed label order
                proba = np.zeros((len(outs), len(labels)), dtype=float)
                for i, row in enumerate(outs):
                    for d in row:
                        proba[i, label_to_idx[d["label"]]] = float(d["score"])
                return proba
            return f_hf
    except Exception:
        pass  # transformers not installed or not a HF pipeline; fall through

    # 3) Generic callable: try to call and coerce to (n_samples, n_classes)
    if callable(model):
        def f_callable(texts: Sequence[str]) -> np.ndarray:
            out = model(list(texts))
            # Possible shapes:
            # - np.array (n, c)
            # - list of dicts (like HF)
            # - list of floats (binary)
            if isinstance(out, list) and out and isinstance(out[0], dict) and "score" in out[0]:
                # List[Dict[label, score]] for a single sample? Normalize to (n, c)
                # Assume all entries share the same label set ordering.
                labels = [d["label"] for d in out]
                proba = np.array([d["score"] for d in out], dtype=float)[None, :]
                return proba
            out = np.asarray(out)
            if out.ndim == 1:
                out = out[:, None]
            return out
        return f_callable

    raise TypeError(
        "Unsupported model type. Provide a scikit-learn estimator with predict_proba, "
        "a HuggingFace TextClassificationPipeline, or a callable that returns probabilities."
    )

def explain_predictions(
    model: object,
    texts: Sequence[str],
    *,
    tokenizer: Optional[Callable[[str], List[str]]] = None,
    background_texts: Optional[Sequence[str]] = None,
    target_label: Optional[Union[int, str]] = None,
    max_evals: Optional[int] = 5000,
    plot: bool = False
) -> shap.Explanation:
    """
    Explain predictions for text inputs using SHAP with a text masker.

    Args:
        model: scikit-learn estimator (with predict_proba), HuggingFace TextClassificationPipeline,
               or a callable f(texts)->probabilities.
        texts: List of raw text strings to explain.
        tokenizer: Optional tokenizer for the SHAP text masker (defaults to whitespace split).
        background_texts: Optional background corpus for the masker (defaults to a small subset of texts).
        target_label: Optional class index or label name (used mainly for plotting labels; model wrapper returns all classes).
        max_evals: Max evaluations for SHAP (passed to explainer call; Kernel-type explainers use it).
        plot: If True, produce a summary plot (requires a display or notebook).

    Returns:
        shap.Explanation containing per-token attributions.
    """
    if not texts:
        raise ValueError("`texts` must be a non-empty sequence of strings.")

    tokenizer = tokenizer or _default_tokenizer

    # Prepare prediction function returning (n_samples, n_classes) probabilities
    f = _wrap_model_to_proba_matrix(model, target_label=target_label)

    # Background for the masker; keep it small for performance
    if background_texts is None:
        bg = list(texts[:10])  # a few examples are enough for text masker
    else:
        bg = list(background_texts)

    # SHAP text masker + explainer
    text_masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(f, text_masker)

    # Compute explanations
    explanation = explainer(list(texts), max_evals=max_evals)

    # Optional plot (summary for class 0 or specified target; SHAP handles multiclass)
    if plot:
        try:
            shap.summary_plot(explanation, texts, class_names=None)
        except Exception:
            # Avoid crashing if running headless; user can plot later in a notebook
            pass

    return explanation


# ------------------------- Examples -------------------------
if __name__ == "__main__":
    # Example A: scikit-learn pipeline
    try:
        from sklearn.pipeline import make_pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        train_texts = ["I loved this movie", "Terrible plot and bad acting", "Great fun!", "Awful waste of time"]
        y = [1, 0, 1, 0]

        clf = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
        clf.fit(train_texts, y)

        sample_texts = ["I absolutely loved it!", "It was terrible."]
        exp = explain_predictions(clf, sample_texts, plot=False)
        # You can visualize in notebooks:
        # shap.text_plot(exp[0])  # explanation for first sample
    except Exception:
        pass

    # Example B: HuggingFace pipeline (optional)
    # from transformers import pipeline
    # hf = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    # sample_texts = ["I absolutely loved it!", "It was terrible."]
    # exp = explain_predictions(hf, sample_texts, target_label="POSITIVE", plot=False)
    # shap.text_plot(exp[0])  # explanation for first sample