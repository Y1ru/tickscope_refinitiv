import eikon as ek
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # Add this import
from datetime import datetime
import pandas as pd
import os
from dotenv import load_dotenv
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import threading
import time
from collections import deque  # Add this import
from plotly.subplots import make_subplots

# Load environment variables and set up Eikon
load_dotenv()
ek.set_app_key(os.getenv('EIKON_API_KEY'))

# Initialize Dash app
app = dash.Dash(__name__)

# Define RICs
stock_ric = "BTC="

# Define constants
MAX_POINTS = 100  # Store last 100 data points

# Initialize StreamingPrices
streaming_prices = ek.StreamingPrices(
    instruments=[stock_ric],
    fields=['CF_LAST', 'BID', 'ASK', 'CF_VOLUME', 'CF_TIME']
)
streaming_prices.open()

# Create DataFrame to store historical data
stock_data = {
    'Time': deque(maxlen=MAX_POINTS),
    'CF_LAST': deque(maxlen=MAX_POINTS),
    'BID': deque(maxlen=MAX_POINTS),
    'ASK': deque(maxlen=MAX_POINTS),
    'CF_VOLUME': deque(maxlen=MAX_POINTS)
}

def update_data():
    while True:
        try:
            df = streaming_prices.get_snapshot()
            
            if stock_ric in df['Instrument'].values:
                stock_row = df[df['Instrument'] == stock_ric].iloc[0]
                try:
                    current_time = pd.to_datetime(stock_row['CF_TIME'])
                    # Only append if we have new data
                    if not stock_data['Time'] or current_time != stock_data['Time'][-1]:
                        stock_data['Time'].append(current_time)
                        stock_data['CF_LAST'].append(pd.to_numeric(stock_row['CF_LAST'], errors='coerce'))
                        stock_data['BID'].append(pd.to_numeric(stock_row['BID'], errors='coerce'))
                        stock_data['ASK'].append(pd.to_numeric(stock_row['ASK'], errors='coerce'))
                        stock_data['CF_VOLUME'].append(pd.to_numeric(stock_row['CF_VOLUME'], errors='coerce'))
                except (ValueError, KeyError) as e:
                    print(f"Error converting data: {e}")
                    continue
            
            time.sleep(1)
            
        except Exception as e:
            print(f"Error updating data: {e}")
            time.sleep(5)

@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('interval-selector', 'value')]
)
def update_graph(n, interval_ms):
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.1,
                       row_heights=[0.4, 0.3, 0.3])
    
    # Get the data and create pandas DataFrame for easier manipulation
    df = pd.DataFrame({
        'Time': list(stock_data['Time']),
        'Price': list(stock_data['CF_LAST']),
        'BID': list(stock_data['BID']),
        'ASK': list(stock_data['ASK']),
        'Volume': list(stock_data['CF_VOLUME'])
    })
    
    if not df.empty:
        # Resample data based on the selected interval
        interval_seconds = interval_ms / 1000
        freq = f'{int(interval_seconds)}S'
        
        # Resample price data
        price_resampled = df.groupby(df['Time'].dt.round(freq)).agg({
            'Price': 'last',
            'Time': 'first'
        }).dropna()
        
        # Resample bid/ask data
        bidask_resampled = df.groupby(df['Time'].dt.round(freq)).agg({
            'BID': 'last',
            'ASK': 'last',
            'Time': 'first'
        }).dropna()
        
        # Resample volume data
        volume_resampled = df.groupby(df['Time'].dt.round(freq)).agg({
            'Volume': 'sum',
            'Time': 'first'
        }).dropna()
        
        # Plot price density
        fig.add_trace(go.Scatter(
            x=price_resampled['Time'],
            y=price_resampled['Price'],
            mode='markers',
            marker=dict(size=10, color='navy'),
            showlegend=False
        ), row=1, col=1)
        
        # Plot Bid/Ask
        fig.add_trace(go.Scatter(
            x=bidask_resampled['Time'],
            y=bidask_resampled['BID'],
            name='Bid',
            mode='lines',
            line=dict(color='red')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=bidask_resampled['Time'],
            y=bidask_resampled['ASK'],
            name='Ask',
            mode='lines',
            line=dict(color='green')
        ), row=2, col=1)
        
        # Plot Volume
        fig.add_trace(go.Bar(
            x=volume_resampled['Time'],
            y=volume_resampled['Volume'],
            name='Volume',
            marker_color='purple'
        ), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title='Real-time BTC Trading Activity',
        showlegend=True,
        uirevision='constant',
        height=1000,
        yaxis3=dict(rangemode='nonnegative')
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Traded Price", row=1, col=1)
    fig.update_yaxes(title_text="Bid/Ask", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

# Start data collection in background
data_thread = threading.Thread(target=update_data, daemon=True)
data_thread.start()

# Dash layout
app.layout = html.Div([
    html.H1('Real-time BTC and Option Prices'),
    html.Div([
        html.Label('Update Interval: '),
        dcc.Dropdown(
            id='interval-selector',
            options=[
                {'label': '1 second', 'value': 1000},
                {'label': '5 seconds', 'value': 5000},
                {'label': '10 seconds', 'value': 10000},
                {'label': '30 seconds', 'value': 30000},
            ],
            value=1000,
            style={'width': '200px', 'display': 'inline-block'}
        ),
    ], style={'margin': '10px'}),
    dcc.Graph(id='live-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # This will be updated by the callback
        n_intervals=0
    )
])

# Add new callback to update interval
@app.callback(
    Output('interval-component', 'interval'),
    Input('interval-selector', 'value')
)
def update_interval(value):
    return value

if __name__ == '__main__':
    try:
        app.run_server(debug=False)
    finally:
        streaming_prices.close()