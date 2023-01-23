import streamlit as st
import pandas as pd
import numpy as np
from config import DATASETS_NAMES_LABELS, RESULTS_PATH
from utils.data_processing import load_data
from utils.models import xgboost
from sklearn.metrics import mean_absolute_error
from utils.plots import plot_results

@st.cache
def st_load_data(dataset_name: str) -> pd.DataFrame:
    return load_data(dataset_name)


st.title('ProfitAI')

st.header('Train single model')

dataset_name = st.selectbox(
    'Please choose dataset',
    DATASETS_NAMES_LABELS.keys()
)
data = load_data(dataset_name)
label = DATASETS_NAMES_LABELS.get(dataset_name)

with st.expander("Show raw data"):
    st.subheader(f'{dataset_name} dataset')
    st.text(f'the label is: {label}')
    st.write(data)

st.subheader('Params for feature selection')
st.subheader('Choose :blue[degree, interaction_only and max_features]')
degree = st.slider('degree', 1, 6, 1)
interaction_only = st.checkbox('interaction_only')
use_max_features = st.checkbox('Set max_features?')
if use_max_features:
    max_features = st.slider('max_features', data.shape[1], 25, 1)
else:
    max_features = None

st.subheader('Train our model')

if st.button('Start training'):
    data_load_state = st.text('Training...')

    x = data.drop(label, axis=1)
    y = data[label]

    # init model
    xgb_pipe = xgboost(
        x,
        y,
        test_size=0.2,
        max_features=max_features,
        degree=degree,
        interaction_only=interaction_only
    )
    # train
    xgb_pipe.fit()
    # predict
    y_hat = xgb_pipe.predict()
    y = xgb_pipe.y_test.values
    data_load_state = st.text('Predicting...')

    mse = mean_absolute_error(y_hat, y)

    data_load_state = st.text(f'Done!, MSE: {mse:.3f}')
    new_result = {
        'dataset_name': dataset_name,
        'degree': degree,
        'interaction_only': interaction_only,
        'max_features': max_features,
        'mse': mse
    }
    results_data = pd.read_csv(RESULTS_PATH)
    results_data = results_data.append(new_result, ignore_index=True)

st.header('Results of all models')
dataset_name_to_plot = st.selectbox(
    'Select dataset to display results',
    DATASETS_NAMES_LABELS.keys()
)
st.subheader(f'data: {dataset_name_to_plot}')
st.plotly_chart(plot_results(dataset_name_to_plot), use_conatiner_width=True)



