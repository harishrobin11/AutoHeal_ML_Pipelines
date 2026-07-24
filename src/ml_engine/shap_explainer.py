import numpy as np
from typing import Dict, Any, List
from src.ml_engine.train_classifier import RootCauseClassifier

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

class SHAPExplainerEngine:
    """
    Provides local feature attribution explanations using SHAP (SHapley Additive exPlanations)
    for model predictions on telemetry anomalies.
    """

    def __init__(self, classifier: RootCauseClassifier):
        self.classifier = classifier
        self.explainer = None

    def initialize_explainer(self, background_logs: List[Dict[str, Any]]):
        """Initializes TreeExplainer using background telemetry logs."""
        X_bg, _ = self.classifier.extract_features(background_logs)
        if HAS_SHAP and self.classifier.is_trained:
            try:
                self.explainer = shap.TreeExplainer(self.classifier.model, X_bg)
            except Exception:
                self.explainer = None

    def explain(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes SHAP feature importance values for a given anomaly log.
        Returns a structured dictionary of feature attributions.
        """
        X, _ = self.classifier.extract_features([log])
        feature_names = self.classifier.FEATURE_NAMES

        if HAS_SHAP and self.explainer and self.classifier.is_trained:
            try:
                shap_vals = self.explainer.shap_values(X)
                if isinstance(shap_vals, list):
                    vals = np.abs(shap_vals[0]).flatten()
                else:
                    vals = np.abs(shap_vals).flatten()
                
                importance_dict = {
                    name: round(float(val), 4) 
                    for name, val in zip(feature_names, vals)
                }
            except Exception:
                importance_dict = self._heuristic_fallback(log)
        else:
            importance_dict = self._heuristic_fallback(log)

        sorted_attributions = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        top_feature = next(iter(sorted_attributions))

        return {
            "top_feature": top_feature,
            "feature_attributions": sorted_attributions,
            "summary": f"Top anomalous factor identified: {top_feature} (importance score: {sorted_attributions[top_feature]})"
        }

    def _heuristic_fallback(self, log: Dict[str, Any]) -> Dict[str, float]:
        """Fallback feature importance calculation if SHAP is unavailable."""
        err_msg = str(log.get("error_message") or "")
        resp_time = log.get("response_time_ms", 45)
        
        return {
            "missing_user_tier": 0.85 if "user_tier" in err_msg or "KeyError" in err_msg else 0.05,
            "response_time_ms": 0.90 if resp_time > 300 else 0.10,
            "has_key_error": 0.80 if "KeyError" in err_msg else 0.02,
            "status_code": 0.70 if log.get("status_code", 200) >= 400 else 0.05,
            "has_type_error": 0.75 if "TypeError" in err_msg else 0.01,
            "is_error": 0.50 if log.get("status_code", 200) >= 400 else 0.0,
            "payload_length": 0.10
        }
