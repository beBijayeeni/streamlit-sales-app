import streamlit as st
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
import os
import plotly.graph_objects as go
import plotly.express as px
import base64

# Constants
N_STEPS = 3
N_FEATURES = 1
EPOCHS = 200

def create_pie_chart(sales_data):
    fig = px.pie(sales_data, values='Net_sales', names=sales_data.index.year, hole=0.5)
    fig.update_traces(textinfo='percent+label', hoverinfo='label+percent+value', textposition="outside")
    return fig

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def preprocess_data(sales_data: pd.DataFrame) -> pd.DataFrame:
    sales_data.index = pd.to_datetime(sales_data['Year'], format='%Y')
    sales_data = sales_data.drop('Year', axis=1)
    return sales_data

def prepare_data(timeseries_data: pd.Series, n_steps: int) -> tuple:
    X, y = [], []
    for i in range(len(timeseries_data)):
        end_ix = i + n_steps
        if end_ix > len(timeseries_data) - 1:
            break
        seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_and_train_model(X: np.ndarray, y: np.ndarray) -> Sequential:
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(N_STEPS, N_FEATURES)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=EPOCHS, verbose=1)
    return model

def predict_next_years(model: Sequential, n_steps: int, n_features: int, n_predictions: int, timeseries_data: pd.Series) -> list:
    predictions = list(timeseries_data[-n_steps:])
    for _ in range(n_predictions):
        x_input = np.array(predictions[-n_steps:]).reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        predictions.append(yhat[0][0])
    return predictions[-n_predictions:]

st.set_page_config(page_title="Prediction!!!", page_icon=":chart_with_upwards_trend:", layout="centered")
logo_url = "https://e7.pngegg.com/pngimages/686/908/png-clipart-walmart-logo-walmart-logo-advertising-coupon-walmart-vertical-logo-text-retail-thumbnail.png"  # Replace with your logo URL or file path
st.markdown(f'''
    <div class="header">
        <img src="{logo_url}" width="60" height="60">
        <h1 class="hover-effect">Walmart Sales Forecast</h1>
    </div>
    ''', unsafe_allow_html=True)

# CSS for hover effect
st.markdown(
    """
    <style>
    .hover-effect:hover {
        color: blue;
        font-size: 2.5em;
        transition: 0.3s;
    }
    .header {
        display: flex;
        align-items: center;
    }
    .header img {
        margin-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    Walmart Inc. is an American multinational retail corporation that operates a chain of hypermarkets, discount department stores, and grocery stores. 
    Founded by Sam Walton in 1962, Walmart has grown to become one of the largest companies in the world by revenue. 
    The company is known for its wide range of products at low prices, making it a popular shopping destination for millions of customers.
    """
)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    sales_data = load_data(uploaded_file)
    sales_data = preprocess_data(sales_data)
    timeseries_data = sales_data['Net_sales']

    X, y = prepare_data(timeseries_data, N_STEPS)
    X = X.reshape((X.shape[0], X.shape[1], N_FEATURES))

    model = create_and_train_model(X, y)

    st.subheader("Historical Sales Proportion by Year")
    st.plotly_chart(create_pie_chart(sales_data), use_container_width=True)

    st.markdown(f'''
            <div class="header">
                <h1 class="hover-effect">Prediction of Yearly Sales Using LSTM</h1>
            </div>
            ''', unsafe_allow_html=True)

    num_years = st.selectbox("Upto which year do you want to predict next?", range(2026, 2035), index=4) - 2024

    next_years = predict_next_years(model, N_STEPS, N_FEATURES, num_years, timeseries_data)

    st.subheader(f"Predictions for the next {num_years} years:")
    for i in range(num_years):
        st.write(f"Prediction for year {2025 + i}: {next_years[i]:.2f} in billion U.S. dollars")

    st.success("Prediction completed successfully!")

    years = list(sales_data.index.year) + [year for year in range(sales_data.index.year[-1] + 1, sales_data.index.year[-1] + num_years + 1)]

    fig = go.Figure()

    all_years = years
    all_sales = list(sales_data['Net_sales']) + list(next_years)

    fig.add_trace(go.Bar(x=years[:len(sales_data)], y=sales_data['Net_sales'],
                         name='Historical Sales',
                         hovertemplate='Year: %{x}<br>Sales: $%{y:.2f} billion<extra></extra>',
                         marker_color='blue', opacity=0.6))

    fig.add_trace(go.Bar(x=years[len(sales_data):], y=next_years,
                         name='Predicted Sales',
                         hovertemplate='Year: %{x}<br>Predicted Sales: $%{y:.2f} billion<extra></extra>',
                         marker_color='orange', opacity=0.6))

    fig.add_trace(go.Scatter(x=years[:len(sales_data)], y=sales_data['Net_sales'],
                             mode='lines+markers', name='Historical Trend',
                             line=dict(color='rgba(221,170,221,0.7)'),
                             hovertemplate='Year: %{x}<br>Sales: $%{y:.2f} billion<extra></extra>'))

    fig.add_trace(go.Scatter(x=years[len(sales_data):], y=next_years,
                             mode='lines+markers', name='Predictions',
                             line=dict(dash='dash', color='red'),
                             hovertemplate='Year: %{x}<br>Predicted Sales: $%{y:.2f} billion<extra></extra>'))

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Net Sales (in billion U.S. dollars)',
        hovermode='x unified',
        barmode='overlay'
    )

    fig.add_vline(x=years[len(sales_data)-1] + 0.5, line_dash="dash", line_color="green")

    st.subheader("Sales Forecast Plot")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="Download Predicted Sales Data as CSV",
        data=pd.DataFrame({'Year': years, 'Net Sales': all_sales}).to_csv(index=False),
        file_name='predicted_sales.csv',
        mime='text/csv'
    )
