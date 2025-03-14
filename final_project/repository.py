from dagster import Definitions
from final_project.dagster_assets import preprocess_data, feature_engineering, train_model, evaluate_model

defs = Definitions(
    assets=[preprocess_data, feature_engineering, train_model, evaluate_model]
)


