import pandas as pd
import numpy as np
from datetime import datetime
import re
import warnings
import os
import io

def load_trading_data_from_bytes(file_bytes, file_name):
    """
    Load and preprocess trading data from bytes (in-memory)
    
    Args:
        file_bytes (bytes): The file content as bytes
        file_name (str): Original file name (used for format detection)
    
    Returns:
        pd.DataFrame: Processed trading dataframe
    """
    # Determine file type from extension
    file_extension = os.path.splitext(file_name)[1].lower()
    
    # Create a BytesIO object
    bytes_io = io.BytesIO(file_bytes)
    
    # Read the file from memory
    if file_extension == '.csv':
        df = pd.read_csv(bytes_io)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(bytes_io)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Check if this is Rachel's Trading Log format or original format
    if 'Name' not in df.columns and (
            'Filled Time' in df.columns or 
            file_name.lower().startswith('rachel')
        ):
        # Convert Rachel's format to match expected format
        return convert_rachel_format(df)
    else:
        # Process original format
        return process_original_format(df)

def load_trading_data(file_path):
    """
    Load and preprocess trading data from CSV or Excel file path
    (Maintained for backward compatibility)
    
    Args:
        file_path (str): Path to the CSV or Excel file
    
    Returns:
        pd.DataFrame: Processed trading dataframe
    """
    # Determine file type from extension
    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    
    # Read the file
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Check if this is Rachel's Trading Log format or original format
    if 'Name' not in df.columns and (
            'Filled Time' in df.columns or 
            file_name.lower().startswith('rachel')
        ):
        # Convert Rachel's format to match expected format
        return convert_rachel_format(df)
    else:
        # Process original format
        return process_original_format(df)

def convert_rachel_format(df):
    """
    Convert Rachel's Trading Log format to match the expected format
    
    Args:
        df (pd.DataFrame): DataFrame in Rachel's format
    
    Returns:
        pd.DataFrame: Processed trading dataframe in the expected format
    """
    # Make a copy to avoid modifying original
    processed_df = df.copy()
    
    # Handle missing columns
    if 'Status' not in processed_df.columns:
        # All trades in Rachel's format are implicitly filled
        processed_df['Status'] = 'Filled'
    
    # Rename columns to match expected format
    column_mapping = {
        'Filled Time': 'Placed Time',
        'Quantity': 'Filled'
    }
    processed_df = processed_df.rename(columns=column_mapping)
    
    # Handle timezone warning by preprocessing the datetime strings
    # Define a function to remove timezone abbreviations
    def clean_datetime(dt_string):
        if isinstance(dt_string, str):
            # Use regex to remove timezone abbreviations like EDT, EST, etc.
            return re.sub(r'\s+[A-Z]{3}$', '', dt_string)
        return dt_string
    
    # First, clean the datetime strings to remove timezone abbreviations
    processed_df['Placed Time'] = processed_df['Placed Time'].apply(clean_datetime)
    
    # Now convert to datetime with warnings suppressed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        processed_df['Placed Time'] = pd.to_datetime(processed_df['Placed Time'], errors='coerce')
    
    # Add time-based columns
    processed_df = add_time_columns(processed_df)
    
    # Apply price range categorization to Avg Price
    processed_df['Price_Range'] = processed_df['Avg Price'].apply(categorize_price_range)
    
    return processed_df

def process_original_format(df):
    """
    Process trading data in the original format
    
    Args:
        df (pd.DataFrame): DataFrame in the original format
    
    Returns:
        pd.DataFrame: Processed trading dataframe
    """
    # Filter only Filled status
    df = df[df['Status'] == 'Filled'].copy()
    
    # Handle timezone warning by preprocessing the datetime strings
    def clean_datetime(dt_string):
        if isinstance(dt_string, str):
            # Use regex to remove timezone abbreviations like EDT, EST, etc.
            return re.sub(r'\s+[A-Z]{3}$', '', dt_string)
        return dt_string
    
    # Create a tzinfos dictionary for common timezone abbreviations
    tzinfos = {
        'EDT': -4 * 3600,  # Eastern Daylight Time (UTC-4)
        'EST': -5 * 3600,  # Eastern Standard Time (UTC-5)
        'CDT': -5 * 3600,  # Central Daylight Time (UTC-5)
        'CST': -6 * 3600,  # Central Standard Time (UTC-6)
        'MDT': -6 * 3600,  # Mountain Daylight Time (UTC-6)
        'MST': -7 * 3600,  # Mountain Standard Time (UTC-7)
        'PDT': -7 * 3600,  # Pacific Daylight Time (UTC-7)
        'PST': -8 * 3600   # Pacific Standard Time (UTC-8)
    }
    
    # First, clean the datetime strings to remove timezone abbreviations
    df['Placed Time'] = df['Placed Time'].apply(clean_datetime)
    
    # Now convert to datetime with warnings suppressed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df['Placed Time'] = pd.to_datetime(df['Placed Time'], errors='coerce')
    
    # Add time-based columns
    df = add_time_columns(df)
    
    # Apply price range categorization to Avg Price
    df['Price_Range'] = df['Avg Price'].apply(categorize_price_range)
    
    return df

def add_time_columns(df):
    """
    Add time-based columns to the dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with added time columns
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Categorize Market Hours
    def categorize_market_hours(hour):
        if hour < 9:
            return 'Pre-Market Hour'
        elif 9 <= hour < 16:
            return 'Regular Hour'
        else:
            return 'Post-Market Hour'
    
    # Add additional time-based columns
    df['Year'] = df['Placed Time'].dt.year
    df['Month'] = df['Placed Time'].dt.month_name()
    df['Day_of_Week'] = df['Placed Time'].dt.day_name()
    df['Date'] = df['Placed Time'].dt.date
    df['Market_Hour'] = df['Placed Time'].dt.hour
    df['DateTime'] = df['Placed Time']
    df['Market_Hour_Category'] = df['Market_Hour'].apply(categorize_market_hours)
    
    return df

def categorize_price_range(price):
    """
    Categorize price ranges
    
    Args:
        price: Price value to categorize
        
    Returns:
        str: Price range category
    """
    # Handle both string and numeric prices
    if isinstance(price, str):
        # Remove '$' and ',' from string prices
        price = price.replace('$', '').replace(',', '')
    
    # Convert to float
    try:
        price = float(price)
    except (ValueError, TypeError):
        return 'Unknown'
    
    # Categorize price ranges
    if price <= 2:
        return '0-2'
    elif price <= 4:
        return '2-4'
    elif price <= 10:
        return '4-10'
    elif price <= 15:
        return '10-15'
    elif price <= 20:
        return '15-20'
    elif price <= 30:
        return '20-30'
    elif price <= 40:
        return '30-40'
    elif price <= 60:
        return '40-60'
    else:
        return '>60'

def prepare_trade_analysis(df):
    """
    Prepare trade analysis by carefully matching buy and sell orders
    
    Args:
        df (pd.DataFrame): Processed trading dataframe
    
    Returns:
        pd.DataFrame: Trade analysis dataframe
    """
    # Check if input has already pre-calculated P&L data (Rachel's format)
    if 'Gain/Loss' in df.columns and 'Profit or Loss' in df.columns:
        return prepare_trade_analysis_from_rachel(df)
    
    # Original logic for matching buy and sell orders
    # Separate buy and sell orders
    buy_orders = df[df['Side'] == 'Buy'].copy()
    sell_orders = df[df['Side'] == 'Sell'].copy()
    
    # Initialize trade analysis list
    trade_analysis = []
    
    # Group buys and sells by symbol and date for precise matching
    for (symbol, date), buy_group in buy_orders.groupby(['Symbol', 'Date']):
        # Find sell orders for this symbol and date
        symbol_date_sells = sell_orders[
            (sell_orders['Symbol'] == symbol) & 
            (sell_orders['Date'] == date)
        ].copy()
        
        # If sells exist for this symbol and date
        if not symbol_date_sells.empty:
            # Sort both buy and sell groups by time to ensure correct matching
            buy_group_sorted = buy_group.sort_values('Placed Time')
            sell_group_sorted = symbol_date_sells.sort_values('Placed Time')
            
            # Track used sell orders to avoid double-counting
            used_sell_indices = set()
            
            # Iterate through buy orders
            for _, buy_row in buy_group_sorted.iterrows():
                buy_qty = buy_row['Filled']
                buy_price = buy_row['Avg Price']
                
                # Find matching sell orders
                remaining_qty = buy_qty
                total_sell_qty = 0
                total_sell_revenue = 0
                used_current_sell_indices = []
                
                # Try to match sell orders
                for sell_idx, sell_row in sell_group_sorted.iterrows():
                    # Skip if this sell order already used
                    if sell_idx in used_sell_indices:
                        continue
                    
                    # Determine sell quantity
                    sell_qty = min(sell_row['Filled'], remaining_qty)
                    
                    # Calculate sell details
                    total_sell_qty += sell_qty
                    total_sell_revenue += sell_qty * sell_row['Avg Price']
                    remaining_qty -= sell_qty
                    used_current_sell_indices.append(sell_idx)
                    
                    # Check if we've matched the full buy quantity
                    if remaining_qty <= 0:
                        break
                
                # If we've sold some or all of the buy order
                if total_sell_qty > 0:
                    # Mark used sell indices
                    used_sell_indices.update(used_current_sell_indices)
                    
                    # Calculate trade performance
                    total_cost = buy_qty * buy_price
                    total_revenue = total_sell_revenue
                    total_profit_loss = total_revenue - total_cost
                    
                    # PnL per share calculation
                    avg_sell_price = total_sell_revenue / total_sell_qty if total_sell_qty > 0 else 0
                    pnl_per_share = avg_sell_price - buy_price
                    
                    # Calculate percent gain/loss safely
                    try:
                        percent_gain_loss = (total_profit_loss / total_cost) * 100 if total_cost != 0 else 0
                    except Exception:
                        percent_gain_loss = 0
                    
                    # Determine trade outcome
                    trade_outcome = 'Win' if total_profit_loss > 0 else 'Loss' if total_profit_loss < 0 else 'Break Even'
                    
                    # Construct trade entry
                    trade_entry = {
                        'Symbol': symbol,
                        'Buy_Quantity': buy_qty,
                        'Sell_Quantity': total_sell_qty,
                        'Avg_Buy_Price': buy_price,
                        'Avg_Sell_Price': avg_sell_price,
                        'PnL_Per_Share': pnl_per_share,
                        'Total_Cost': total_cost,
                        'Total_Revenue': total_revenue,
                        'Total_Profit_Loss': total_profit_loss,
                        'Percent_Gain_Loss': percent_gain_loss,  # Keep old name for backward compatibility
                        'Trade_Outcome': trade_outcome,
                        'Price_Range': buy_row['Price_Range'],
                        'Market_Hour_Category': buy_row['Market_Hour_Category'],
                        'Year': buy_row['Year'],
                        'Month': buy_row['Month'],
                        'Day_of_Week': buy_row['Day_of_Week'],
                        'Date': date,
                        'DateTime': buy_row['DateTime']
                    }
                    
                    trade_analysis.append(trade_entry)
    
    # Create dataframe from trade analysis list
    result_df = pd.DataFrame(trade_analysis)
    
    # Add PnL_% column as a copy of Percent_Gain_Loss
    if not result_df.empty:
        result_df['PnL_%'] = result_df['Percent_Gain_Loss']
    
    return result_df

def prepare_trade_analysis_from_rachel(df):
    """
    Prepare trade analysis from Rachel's Trading Log format
    which already has calculated P&L data
    
    Args:
        df (pd.DataFrame): DataFrame in Rachel's format
    
    Returns:
        pd.DataFrame: Trade analysis dataframe in the expected format
    """
    # Focus on Buy orders as they contain all the P&L data in Rachel's format
    buy_orders = df[df['Side'] == 'Buy'].copy()
    
    # Initialize trade analysis list
    trade_analysis = []
    
    # Iterate through buy orders
    for _, buy_row in buy_orders.iterrows():
        # Skip rows with no P&L data
        if pd.isna(buy_row['Gain/Loss']):
            continue
            
        # Calculate quantities from the data we have
        symbol = buy_row['Symbol']
        buy_qty = buy_row['Filled']
        buy_price = buy_row['Avg Price']
        date = buy_row['Date']
        
        # Get P&L data
        total_profit_loss = buy_row['Gain/Loss']
        percent_gain_loss = buy_row['% Gain/Loss']
        
        # Calculate Total Cost
        total_cost = buy_qty * buy_price
        
        # Calculate Total Revenue
        total_revenue = total_cost + total_profit_loss
        
        # Calculate Avg Sell Price
        avg_sell_price = total_revenue / buy_qty if buy_qty > 0 else 0
        
        # Calculate PnL per share
        pnl_per_share = avg_sell_price - buy_price
        
        # Determine trade outcome
        if 'Profit or Loss' in buy_row:
            # Use the existing outcome if available
            trade_outcome = 'Win' if buy_row['Profit or Loss'] == 'Profit' else 'Loss'
        else:
            # Otherwise calculate it from the P&L
            trade_outcome = 'Win' if total_profit_loss > 0 else 'Loss' if total_profit_loss < 0 else 'Break Even'
        
        # Construct trade entry
        trade_entry = {
            'Symbol': symbol,
            'Buy_Quantity': buy_qty,
            'Sell_Quantity': buy_qty,  # Assume all bought shares were sold
            'Avg_Buy_Price': buy_price,
            'Avg_Sell_Price': avg_sell_price,
            'PnL_Per_Share': pnl_per_share,
            'Total_Cost': total_cost,
            'Total_Revenue': total_revenue,
            'Total_Profit_Loss': total_profit_loss,
            'Percent_Gain_Loss': percent_gain_loss,  # Keep old name for backward compatibility
            'Trade_Outcome': trade_outcome,
            'Price_Range': buy_row['Price_Range'],
            'Market_Hour_Category': buy_row['Market_Hour_Category'],
            'Year': buy_row['Year'],
            'Month': buy_row['Month'],
            'Day_of_Week': buy_row['Day_of_Week'],
            'Date': date,
            'DateTime': buy_row['DateTime']
        }
        
        trade_analysis.append(trade_entry)
    
    # Create dataframe from trade analysis list
    result_df = pd.DataFrame(trade_analysis)
    
    # Add PnL_% column as a copy of Percent_Gain_Loss
    if not result_df.empty:
        result_df['PnL_%'] = result_df['Percent_Gain_Loss']
    
    return result_df