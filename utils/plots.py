import plotly.express as px
import pandas as pd
import numpy as np
from config import RESULTS_PATH, RESULTS_PATH_DICT


def plot_results(dataset_name: str, degree:int = 3, interaction_only=False)->None:
    max_max_features = 25
    df = pd.read_csv(RESULTS_PATH_DICT.get(dataset_name))

    conditions = (df['dataset_name'] == dataset_name) \
                 & ((df['degree'] == degree) | (df['degree'].isna())) \
                 & ((df['interaction_only'] == interaction_only) | df['interaction_only'].isna()) \
                 & (df['max_features'] <= 25)

    fig = px.line(df[conditions], x='max_features', y='mse', color='is_benchmark', markers=True)
    return fig