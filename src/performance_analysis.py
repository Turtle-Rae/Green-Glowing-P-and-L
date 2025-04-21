import pandas as pd
import numpy as np

def calculate_overall_performance(trade_analysis_df):
    """
    Calculate overall trading performance metrics
    
    Args:
        trade_analysis_df (pd.DataFrame): Trade analysis dataframe
    
    Returns:
        dict: Performance metrics
    """
    performance_metrics = {
        'Total_Trades': len(trade_analysis_df),
        'Distinct_Tickers': trade_analysis_df['Symbol'].nunique(),
        'Total_Wins': (trade_analysis_df['Trade_Outcome'] == 'Win').sum(),
        'Total_Losses': (trade_analysis_df['Trade_Outcome'] == 'Loss').sum(),
        'Total_Break_Even': (trade_analysis_df['Trade_Outcome'] == 'Break Even').sum(),
        'Win_Rate': (trade_analysis_df['Trade_Outcome'] == 'Win').mean() * 100,
        'Loss_Rate': (trade_analysis_df['Trade_Outcome'] == 'Loss').mean() * 100,
        'Average_Percent_Gain_Loss': trade_analysis_df['Percent_Gain_Loss'].mean(),
        'Trading_Days': trade_analysis_df['Date'].nunique()
    }
    
    return performance_metrics

def performance_by_dimension(trade_analysis_df, dimension):
    """
    Analyze performance by specific dimension
    
    Args:
        trade_analysis_df (pd.DataFrame): Trade analysis dataframe
        dimension (str): Dimension to analyze (Price_Range, Market_Hour_Category, etc.)
    
    Returns:
        pd.DataFrame: Performance breakdown by dimension
    """
    performance_breakdown = trade_analysis_df.groupby(dimension).agg({
        'Symbol': 'count',  # Number of trades
        'Profit_Loss': 'sum',  # Total profit/loss
        'Percent_Gain_Loss': 'mean',  # Average percent gain/loss
        'Trade_Outcome': lambda x: (x == 'Win').mean() * 100  # Win rate
    }).rename(columns={
        'Symbol': 'Number_of_Trades',
        'Profit_Loss': 'Total_Profit_Loss',
        'Percent_Gain_Loss': 'Average_Percent_Gain_Loss',
        'Trade_Outcome': 'Win_Rate_Percent'
    }).reset_index()
    
    return performance_breakdown

def detailed_trade_statistics(trade_analysis_df):
    """
    Generate detailed trade statistics
    
    Args:
        trade_analysis_df (pd.DataFrame): Trade analysis dataframe
    
    Returns:
        dict: Detailed trade statistics
    """
    # Descriptive statistics
    trade_stats = {
        'Trade_Quantity': {
            'Mean_Buy_Quantity': trade_analysis_df['Buy_Quantity'].mean(),
            'Median_Buy_Quantity': trade_analysis_df['Buy_Quantity'].median(),
            'Total_Buy_Quantity': trade_analysis_df['Buy_Quantity'].sum()
        },
        'Price_Stats': {
            'Mean_Buy_Price': trade_analysis_df['Avg_Buy_Price'].mean(),
            'Mean_Sell_Price': trade_analysis_df['Avg_Sell_Price'].mean(),
            'Median_Buy_Price': trade_analysis_df['Avg_Buy_Price'].median(),
            'Median_Sell_Price': trade_analysis_df['Avg_Sell_Price'].median()
        },
        'Profit_Stats': {
            'Max_Profit_Trade': trade_analysis_df['Profit_Loss'].max(),
            'Max_Loss_Trade': trade_analysis_df['Profit_Loss'].min(),
            'Largest_Percent_Gain': trade_analysis_df['Percent_Gain_Loss'].max(),
            'Largest_Percent_Loss': trade_analysis_df['Percent_Gain_Loss'].min()
        }
    }
    
    return trade_stats