import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.dates as mdates

# Define functions

def calculate_markov_chain_probability(df, window_size=20):
    """Calculate the probability of mean reversion using a simple Markov Chain model"""
    probabilities = []
    num_windows = len(df) - window_size
    
    if num_windows <= 0:
        return [], []  # Not enough data to calculate probabilities
    
    for i in range(num_windows):
        window = df.iloc[i:i + window_size]
        mean = window['PercentageDifference'].mean()
        stdev = window['PercentageDifference'].std()
        
        # Probability of returning to the mean (mean reversion)
        within_one_std = (window['PercentageDifference'] >= mean - stdev) & (window['PercentageDifference'] <= mean + stdev)
        probability = within_one_std.sum() / len(window)
        probabilities.append(probability)
    
    prob_dates = df.index[window_size:]  # Dates corresponding to the probability calculation
    return probabilities, prob_dates

def plot_percentage_difference(df, ticker1, ticker2, average_percentage_difference):
    """Plot the percentage difference between two tickers"""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # Plot percentage difference
    ax.plot(df.index, df['PercentageDifference'], color='black', linestyle='-', linewidth=2)
    ax.axhline(y=average_percentage_difference, color='grey', linestyle='--', linewidth=1.5)
    ax.set_ylabel('Percentage Difference (%)', fontsize=16)
    ax.set_title(f'Percentage Difference Between {ticker1} and {ticker2}', fontsize=18, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Format x-axis to display dates properly
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Adjust layout to integrate into Streamlit
    plt.tight_layout()
    
    # Show plot
    st.pyplot(fig)

def plot_normalized_prices(df, ticker1, ticker2):
    """Plot normalized prices for two tickers"""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # Plot normalized prices
    ax.plot(df.index, df['NormalizedPrice1'], color='darkgrey', label=f'{ticker1} Price', linewidth=2)
    ax.plot(df.index, df['NormalizedPrice2'], color='lightgrey', label=f'{ticker2} Price', linewidth=2)
    ax.set_ylabel('Normalized Price (%)', fontsize=16)
    ax.set_title('Price Charts (Normalized to % Change)', fontsize=18, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=14)
    
    # Format x-axis to display dates properly
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Adjust layout to integrate into Streamlit
    plt.tight_layout()
    
    # Show plot
    st.pyplot(fig)

def plot_rolling_correlation(df, average_rolling_correlation):
    """Plot the rolling correlation between two tickers"""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # Plot rolling correlation
    ax.plot(df.index, df['RollingCorrelation'], color='black', linestyle='-', linewidth=2)
    ax.axhline(y=average_rolling_correlation, color='grey', linestyle='--', linewidth=1.5)
    ax.set_ylabel('Correlation (r)', fontsize=16)
    ax.set_title('Rolling Correlation (15-Day Window)', fontsize=18, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Format x-axis to display dates properly
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Adjust layout to integrate into Streamlit
    plt.tight_layout()
    
    # Show plot
    st.pyplot(fig)

def plot_mean_reversion_probability(interp_probabilities, interp_dates):
    """Plot the probability of mean reversion over time"""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # Plot the probability of mean reversion
    if len(interp_probabilities) > 0:
        ax.plot(interp_dates, interp_probabilities, color='black', linestyle='-', linewidth=2)
        ax.fill_between(interp_dates, 0.8, interp_probabilities, where=(np.array(interp_probabilities) > 0.8), color='grey', alpha=0.3)
    
    ax.set_ylabel('Probability (%)', fontsize=16)
    ax.set_title('Probability of Mean Reversion of % Difference', fontsize=18, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Format x-axis to display dates properly
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Adjust layout to integrate into Streamlit
    plt.tight_layout()
    
    # Show plot
    st.pyplot(fig)

def fetch_data(ticker1, ticker2, years):
    """Fetch data and prepare DataFrame"""
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years)
    
    # Fetch historical data for the specified time frame
    data1 = yf.download(ticker1, start=start_date, end=end_date)['Adj Close']
    data2 = yf.download(ticker2, start=start_date, end=end_date)['Adj Close']
    
    # Combine the data into a single DataFrame
    df = pd.DataFrame({'Price1': data1, 'Price2': data2}).dropna()
    
    # Normalize prices to start at 0% and show percentage changes
    df['NormalizedPrice1'] = ((df['Price1'] / df['Price1'].iloc[0]) - 1) * 100
    df['NormalizedPrice2'] = ((df['Price2'] / df['Price2'].iloc[0]) - 1) * 100
    
    # Calculate percentage difference based on normalized prices
    df['PercentageDifference'] = df['NormalizedPrice1'] - df['NormalizedPrice2']
    
    # Calculate rolling correlation with a 15-day window
    df['RollingCorrelation'] = df['NormalizedPrice1'].rolling(window=15).corr(df['NormalizedPrice2'])
    
    # Calculate average percentage difference and average rolling correlation
    average_percentage_difference = df['PercentageDifference'].mean()
    average_rolling_correlation = df['RollingCorrelation'].mean()
    
    # Calculate the probability of mean reversion
    probabilities, prob_dates = calculate_markov_chain_probability(df)
    
    # Smoothen the probability graph using cubic spline interpolation
    if probabilities:
        interp_func = interp1d(np.arange(len(probabilities)), probabilities, kind='cubic', fill_value="extrapolate")
        interp_x = np.linspace(0, len(probabilities) - 1, num=len(df))
        interp_probabilities = interp_func(interp_x)
        interp_dates = pd.date_range(start=df.index[0], end=df.index[-1], periods=len(df))
    else:
        interp_probabilities, interp_dates = [], []  # Ensure these variables are always initialized
    
    return df, average_percentage_difference, average_rolling_correlation, interp_probabilities, interp_dates

# Streamlit app

def main():
    st.title("Correlation Lab - Financial Analysis")
    
    # Create layout with columns for inputs and outputs
    col1, col2 = st.columns(2)

    with col1:
        ticker1 = st.text_input("Enter Ticker 1", value='AAPL')
        ticker2 = st.text_input("Enter Ticker 2", value='MSFT')

    with col2:
        years = st.number_input("Number of Years", min_value=1, max_value=10, value=1)
    
    # Use a button to trigger the analysis
    if st.button("Run Analysis"):
        st.markdown("---")
        
        st.subheader("Analysis Results")
        
        # Fetch the data
        df, average_percentage_difference, average_rolling_correlation, interp_probabilities, interp_dates = fetch_data(ticker1, ticker2, years)
        
        # Show average results before plots
        st.markdown(f"**Average Percentage Difference**: {average_percentage_difference:.2f}%")
        st.markdown(f"**Average Rolling Correlation**: {average_rolling_correlation:.2f}")
        
        # Display individual plots
        plot_percentage_difference(df, ticker1, ticker2, average_percentage_difference)
        plot_normalized_prices(df, ticker1, ticker2)
        plot_rolling_correlation(df, average_rolling_correlation)

        st.markdown("---")
        
        st.subheader("Backtesting Results")
        plot_mean_reversion_probability(interp_probabilities, interp_dates)

if __name__ == "__main__":
    main()
