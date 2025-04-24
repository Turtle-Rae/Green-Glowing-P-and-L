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
    # Basic performance metrics
    performance_metrics = {
        'Total_Trades': len(trade_analysis_df),
        'Distinct_Tickers': trade_analysis_df['Symbol'].nunique(),
        'Total_Wins': (trade_analysis_df['Trade_Outcome'] == 'Win').sum(),
        'Total_Losses': (trade_analysis_df['Trade_Outcome'] == 'Loss').sum(),
        'Total_Break_Even': (trade_analysis_df['Trade_Outcome'] == 'Break Even').sum(),
        'Win_Rate': (trade_analysis_df['Trade_Outcome'] == 'Win').mean() * 100,
        'Loss_Rate': (trade_analysis_df['Trade_Outcome'] == 'Loss').mean() * 100,
        'Average_Percent_Gain_Loss': trade_analysis_df['Percent_Gain_Loss'].mean(),  # Keep the original metric
        'Trading_Days': trade_analysis_df['Date'].nunique()
    }
    
    # Add PnL_% average too for use in UI
    if 'PnL_%' in trade_analysis_df.columns:
        performance_metrics['Average_PnL_Percent'] = trade_analysis_df['PnL_%'].mean()
    
    # Filter for wins and losses
    win_trades = trade_analysis_df[trade_analysis_df['Trade_Outcome'] == 'Win']
    loss_trades = trade_analysis_df[trade_analysis_df['Trade_Outcome'] == 'Loss']
    
    # Calculate new risk-reward metrics
    if not win_trades.empty:
        performance_metrics['Reward_%'] = win_trades['Percent_Gain_Loss'].mean()
        performance_metrics['Reward_$'] = win_trades['PnL_Per_Share'].mean()
    else:
        performance_metrics['Reward_%'] = 0
        performance_metrics['Reward_$'] = 0
    
    if not loss_trades.empty:
        # Use absolute values for risk metrics for easier comparison
        performance_metrics['Risk_%'] = abs(loss_trades['Percent_Gain_Loss'].mean())
        performance_metrics['Risk_$'] = abs(loss_trades['PnL_Per_Share'].mean())
    else:
        performance_metrics['Risk_%'] = 0
        performance_metrics['Risk_$'] = 0
    
    # Calculate risk-reward ratio
    if performance_metrics['Risk_%'] > 0:
        performance_metrics['Risk_Reward_Ratio'] = performance_metrics['Reward_%'] / performance_metrics['Risk_%']
    else:
        performance_metrics['Risk_Reward_Ratio'] = 0
    
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
    # Determine which profit/loss field to use
    profit_loss_field = 'Total_Profit_Loss' if 'Total_Profit_Loss' in trade_analysis_df.columns else 'Profit_Loss'
    
    performance_breakdown = trade_analysis_df.groupby(dimension).agg({
        'Symbol': 'count',  # Number of trades
        profit_loss_field: 'sum',  # Total profit/loss
        'Percent_Gain_Loss': 'mean',  # Average percent gain/loss
        'Trade_Outcome': lambda x: (x == 'Win').mean() * 100  # Win rate
    }).rename(columns={
        'Symbol': 'Number_of_Trades',
        profit_loss_field: 'Total_Profit_Loss',
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
    # Determine which profit/loss field to use
    profit_loss_field = 'Total_Profit_Loss' if 'Total_Profit_Loss' in trade_analysis_df.columns else 'Profit_Loss'
    
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
            'Max_Profit_Trade': trade_analysis_df[profit_loss_field].max(),
            'Max_Loss_Trade': trade_analysis_df[profit_loss_field].min(),
            'Largest_Percent_Gain': trade_analysis_df['Percent_Gain_Loss'].max(),
            'Largest_Percent_Loss': trade_analysis_df['Percent_Gain_Loss'].min()
        }
    }
    
    return trade_stats