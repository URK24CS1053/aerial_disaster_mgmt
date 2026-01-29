#!/usr/bin/env python3
"""
Retrain the fusion model with 14-feature vectors.

This script retrains the model to work with the new 14-element feature vector format
used by the refactored PerceptionOutput schema.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_making import fusion_engine
from decision_making import logger_config

logger = logger_config.get_logger(__name__)


def main():
    """Retrain the model with 14-feature format."""
    print("\n" + "="*70)
    print("RETRAINING FUSION MODEL")
    print("="*70)
    
    try:
        # Train the model (this will use the updated _create_training_data)
        logger.info("Training model with 14-feature format...")
        model = fusion_engine.train_model(save=True)
        
        if model:
            logger.info("[OK] Model successfully trained and saved")
            print("\n[OK] Model successfully trained and saved to fusion_model.pkl")
            
            # Verify the model
            print("\nModel details:")
            print(f"  - Number of features: {model.n_features_in_}")
            print(f"  - Number of classes: {model.n_classes_}")
            print(f"  - Feature importances (top 5):")
            
            # Get top 5 important features
            importances = model.feature_importances_
            top_indices = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)[:5]
            
            feature_names = [
                "detected", "detection_conf", "motion_state", "motion_score", 
                "motion_conf", "heat_present", "thermal_intensity", "thermal_stability",
                "breathing_proxy", "breathing_conf", "thermal_conf", "visibility",
                "signal_quality", "num_uncertainty"
            ]
            
            for rank, idx in enumerate(top_indices, 1):
                print(f"    {rank}. {feature_names[idx]}: {importances[idx]:.4f}")
            
            return 0
        else:
            logger.error("[FAILED] Model training returned None")
            print("\n[FAILED] Model training returned None")
            return 1
            
    except Exception as e:
        logger.error(f"[ERROR] Model retraining failed: {e}", exc_info=True)
        print(f"\n[ERROR] Model retraining failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
