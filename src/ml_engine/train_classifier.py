import numpy as np
import json
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    # Test instantiating model to verify libomp library loading
    _dummy_check = xgb.XGBClassifier()
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

class RootCauseClassifier:
    """
    ML Classifier (XGBoost / RandomForest) trained on telemetry stream features 
    to classify failure mode root causes (SchemaBreak, LatencySpike, TypeMismatch, Normal).
    """
    
    FAILURE_CLASSES = ["Normal", "SchemaBreak", "LatencySpike", "TypeMismatch", "ServiceUnavailable"]
    FEATURE_NAMES = [
        "status_code",
        "response_time_ms",
        "is_error",
        "has_key_error",
        "has_type_error",
        "missing_user_tier",
        "payload_length"
    ]

    def __init__(self):
        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                eval_metric="mlogloss"
            )
        else:
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.FAILURE_CLASSES)
        self.is_trained = False

    def extract_features(self, logs: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Transforms raw telemetry log dictionaries into a numerical feature matrix."""
        feature_rows = []
        labels = []

        for log in logs:
            payload = {}
            if log.get("payload_json"):
                try:
                    payload = json.loads(log["payload_json"])
                except Exception:
                    pass

            err_msg = str(log.get("error_message") or "")
            
            features = [
                float(log.get("status_code", 200)),
                float(log.get("response_time_ms", 45.0)),
                1.0 if log.get("status_code", 200) >= 400 else 0.0,
                1.0 if "KeyError" in err_msg else 0.0,
                1.0 if "TypeError" in err_msg else 0.0,
                1.0 if "user_tier" not in payload else 0.0,
                float(len(str(log.get("payload_json") or "")))
            ]
            feature_rows.append(features)

            # Determine label for training
            if "KeyError" in err_msg or "user_tier" not in payload:
                labels.append("SchemaBreak")
            elif "TypeError" in err_msg:
                labels.append("TypeMismatch")
            elif log.get("status_code", 200) >= 500:
                labels.append("ServiceUnavailable")
            elif log.get("response_time_ms", 0) > 300.0:
                labels.append("LatencySpike")
            else:
                labels.append("Normal")

        return np.array(feature_rows), labels

    def train(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trains XGBoost/RandomForest model on provided telemetry data logs."""
        X, y_raw = self.extract_features(logs)
        self.label_encoder.fit(y_raw)
        y = self.label_encoder.transform(y_raw)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        train_acc = float(np.mean(self.model.predict(X) == y))
        return {
            "status": "trained",
            "samples": len(logs),
            "accuracy": round(train_acc, 4),
            "classes": list(self.label_encoder.classes_)
        }

    def predict(self, single_log: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts the failure mode class and class probabilities for a single telemetry record."""
        X, _ = self.extract_features([single_log])
        
        if not self.is_trained:
            err_msg = str(single_log.get("error_message") or "")
            if "KeyError" in err_msg:
                pred_label = "SchemaBreak"
            elif single_log.get("response_time_ms", 0) > 300:
                pred_label = "LatencySpike"
            elif "TypeError" in err_msg:
                pred_label = "TypeMismatch"
            else:
                pred_label = "Normal"
            return {"predicted_class": pred_label, "confidence": 0.95, "probabilities": {pred_label: 0.95}}

        pred_idx = self.model.predict(X)[0]
        probas = self.model.predict_proba(X)[0]
        
        pred_label = self.label_encoder.inverse_transform([pred_idx])[0]
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.label_encoder.classes_, probas)}
        
        return {
            "predicted_class": pred_label,
            "confidence": round(float(np.max(probas)), 4),
            "probabilities": prob_dict
        }
