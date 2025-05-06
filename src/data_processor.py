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
    LEGACY: Prepare trade analysis by matching buy and sell orders using the original algorithm
    
    NOTE: This function is kept for backward compatibility but may not handle
    complex trading patterns correctly. Use prepare_trade_analysis_enhanced instead.
    
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


# This is a partial modification of the prepare_trade_analysis_enhanced function
# to fix the issue with sell price matching

def prepare_trade_analysis_enhanced(df):
    """
    Enhanced trade analysis to handle complex buy-sell patterns including partial positions
    
    This function handles:
    - 1 buy (full position) 1 sell (full position)
    - 1 buy (full position) 2 sells (2 half positions)
    - 2 buys (half and half) 1 sell
    - 1 buy (full) 1 sell (half) 1 buy 1 sell (half) 1 buy 1 sell (full)
    
    Args:
        df (pd.DataFrame): Processed trading dataframe
    
    Returns:
        pd.DataFrame: Trade analysis dataframe
    """
    # Check if input has already pre-calculated P&L data (Rachel's format)
    if 'Gain/Loss' in df.columns and 'Profit or Loss' in df.columns:
        return prepare_trade_analysis_from_rachel(df)
    
    # Sort dataframe by symbol, date and time to ensure chronological processing
    # Make sure datetime is used for sorting to handle intraday timing correctly
    df_sorted = df.sort_values(['Symbol', 'DateTime']).copy()
    
    # Initialize trade analysis list
    trade_analysis = []
    
    # Initialize position tracker dictionary by symbol
    positions = {}
    
    # Debug flag for troubleshooting specific symbols
    debug_mode = True  # Set to True to enable debugging
    debug_symbols = ['FOXO']  # Add symbols to debug here
    
    # For debugging
    def debug_print(symbol, message):
        if debug_mode and symbol in debug_symbols:
            print(f"DEBUG [{symbol}]: {message}")
    
    # Process orders chronologically
    for idx, row in df_sorted.iterrows():
        symbol = row['Symbol']
        side = row['Side']
        date = row['Date']
        qty = row['Filled']
        price = row['Avg Price']
        timestamp = row['DateTime']
        
        # Skip non-filled orders
        if row['Status'] != 'Filled':
            continue
        
        # Initialize position tracking for this symbol if not exists
        if symbol not in positions:
            positions[symbol] = {
                'position': 0,
                'buys': [],
                'partial_sells': []
            }
        
        position_data = positions[symbol]
        
        debug_print(symbol, f"Processing {side} order for {qty} shares at ${price} at {timestamp}")
        debug_print(symbol, f"Before: Position={position_data['position']}, Buys={[(b['qty'], b['price']) for b in position_data['buys']]}")
        
        # Handle Buy order
        if side == 'Buy':
            # Add to current position
            position_data['position'] += qty
            
            # Store buy information with complete row data
            position_data['buys'].append({
                'qty': qty,
                'price': price,
                'date': date,
                'timestamp': timestamp,
                'row_data': row.to_dict()
            })
            
            debug_print(symbol, f"After Buy: Position={position_data['position']}, Buys={[(b['qty'], b['price']) for b in position_data['buys']]}")
        
        # Handle Sell order
        elif side == 'Sell':
            # Skip if no position to sell
            if position_data['position'] <= 0:
                debug_print(symbol, f"Skipping sell - no position available")
                continue
                
            # Determine actual sell quantity (can't sell more than position)
            actual_sell_qty = min(qty, position_data['position'])
            
            # Update remaining position
            position_data['position'] -= actual_sell_qty
            
            # IMPORTANT CHANGE: Store original sell data before processing any matches
            # This ensures we use the correct sell price for all matches from this sell order
            original_sell_data = {
                'sell_qty': actual_sell_qty,
                'sell_price': price,
                'sell_date': date,
                'sell_timestamp': timestamp,
                'sell_row_data': row.to_dict()
            }
            
            # Track this sell order
            position_data['partial_sells'].append(original_sell_data)
            
            debug_print(symbol, f"Sell order: {actual_sell_qty} shares at ${price}")
            
            # Process completed trades while we have sells to allocate
            remaining_sell_qty = actual_sell_qty
            
            # Process buys from oldest to newest (FIFO - First In, First Out)
            while remaining_sell_qty > 0 and position_data['buys']:
                oldest_buy = position_data['buys'][0]
                
                # Calculate how much of this buy can be sold
                match_qty = min(oldest_buy['qty'], remaining_sell_qty)
                
                debug_print(symbol, f"Matching {match_qty} shares from buy at ${oldest_buy['price']}")
                
                # Update remaining quantities
                oldest_buy['qty'] -= match_qty
                remaining_sell_qty -= match_qty
                
                # Generate trade entry
                buy_price = oldest_buy['price']
                
                # IMPORTANT CHANGE: Always use the original sell price from this sell transaction
                # Not a price from a different sell transaction
                weighted_sell_price = original_sell_data['sell_price']
                
                # Calculate trade performance
                total_cost = match_qty * buy_price
                total_revenue = match_qty * weighted_sell_price
                total_profit_loss = total_revenue - total_cost
                pnl_per_share = weighted_sell_price - buy_price
                
                # Calculate percent gain/loss safely
                try:
                    percent_gain_loss = (total_profit_loss / total_cost) * 100 if total_cost != 0 else 0
                except Exception:
                    percent_gain_loss = 0
                
                # Determine trade outcome
                trade_outcome = 'Win' if total_profit_loss > 0 else 'Loss' if total_profit_loss < 0 else 'Break Even'
                
                # Get buy data
                buy_row_data = oldest_buy['row_data']
                
                # For some fields, fallback to safe defaults if data is missing
                price_range = buy_row_data.get('Price_Range', 'Unknown')
                if price_range == 'Unknown' and isinstance(buy_price, (int, float)):
                    price_range = categorize_price_range(buy_price)
                
                # Construct trade entry with detailed information
                trade_entry = {
                    'Symbol': symbol,
                    'Buy_Quantity': match_qty,
                    'Sell_Quantity': match_qty,
                    'Avg_Buy_Price': buy_price,
                    'Avg_Sell_Price': weighted_sell_price,
                    'PnL_Per_Share': pnl_per_share,
                    'Total_Cost': total_cost,
                    'Total_Revenue': total_revenue,
                    'Total_Profit_Loss': total_profit_loss,
                    'Percent_Gain_Loss': percent_gain_loss,
                    'PnL_%': percent_gain_loss,
                    'Trade_Outcome': trade_outcome,
                    'Price_Range': price_range,
                    'Market_Hour_Category': buy_row_data.get('Market_Hour_Category', 'Unknown'),
                    'Year': buy_row_data.get('Year', date.year if hasattr(date, 'year') else None),
                    'Month': buy_row_data.get('Month', None),
                    'Day_of_Week': buy_row_data.get('Day_of_Week', None),
                    'Date': date,
                    'DateTime': buy_row_data.get('DateTime', None),
                    'Buy_Time': oldest_buy['timestamp'],
                    'Sell_Time': timestamp
                }
                
                debug_print(symbol, f"Created trade entry: {match_qty} shares, P&L: ${total_profit_loss:.2f}, {percent_gain_loss:.2f}%")
                
                trade_analysis.append(trade_entry)
                
                # If buy is fully matched, remove it from the list
                if oldest_buy['qty'] <= 0:
                    position_data['buys'].pop(0)
                    debug_print(symbol, f"Removed fully matched buy")
            
            debug_print(symbol, f"After Sell: Position={position_data['position']}, Buys={[(b['qty'], b['price']) for b in position_data['buys']]}")
    
    # Create dataframe from trade analysis list
    result_df = pd.DataFrame(trade_analysis)
    
    # IMPORTANT CHANGE: Add validation check for unmatched positions
    for symbol, position_data in positions.items():
        if position_data['position'] != 0 or position_data['buys']:
            print(f"Warning: Unmatched positions for {symbol}: {position_data['position']} shares remaining")
            if position_data['buys']:
                print(f"  Unmatched buys: {[(b['qty'], b['price']) for b in position_data['buys']]}")
    
    # If DataFrame is empty, return an empty DataFrame with the expected columns
    if result_df.empty:
        columns = ['Symbol', 'Buy_Quantity', 'Sell_Quantity', 'Avg_Buy_Price', 'Avg_Sell_Price',
                  'PnL_Per_Share', 'Total_Cost', 'Total_Revenue', 'Total_Profit_Loss',
                  'Percent_Gain_Loss', 'PnL_%', 'Trade_Outcome', 'Price_Range',
                  'Market_Hour_Category', 'Year', 'Month', 'Day_of_Week', 'Date', 'DateTime']
        return pd.DataFrame(columns=columns)
    
    return result_df

def prepare_position_based_trade_analysis(df):
    """
    Prepare trade analysis based on positions - considering all activity within a position as one trade
    
    A "trade" is defined as the entire buy/sell activity from position open to position close
    (when position quantity returns to zero)
    
    Args:
        df (pd.DataFrame): Processed trading dataframe
        
    Returns:
        pd.DataFrame: Position-based trade analysis dataframe
    """
    # Sort dataframe by symbol, date and time to ensure chronological processing
    df_sorted = df.sort_values(['Symbol', 'DateTime']).copy()
    
    # Initialize position-based trade analysis list
    position_trade_analysis = []
    
    # Initialize position tracker dictionary by symbol
    positions = {}
    
    # Debug flag for troubleshooting specific symbols
    debug_mode = True
    debug_symbols = ['FOXO']
    
    # For debugging
    def debug_print(symbol, message):
        if debug_mode and symbol in debug_symbols:
            print(f"DEBUG [{symbol}]: {message}")
    
    # Initialize trade ID counter
    trade_id_counter = 1
    
    # Process orders chronologically
    for idx, row in df_sorted.iterrows():
        symbol = row['Symbol']
        side = row['Side']
        date = row['Date']
        qty = row['Filled']
        price = row['Avg Price']
        timestamp = row['DateTime']
        
        # Skip non-filled orders
        if row['Status'] != 'Filled':
            continue
        
        # Initialize position tracking for this symbol if not exists
        if symbol not in positions:
            positions[symbol] = {
                'position': 0,
                'current_trade_id': trade_id_counter,
                'buys': [],
                'sells': [],
                'trade_data': {
                    'first_buy_date': None,
                    'last_sell_date': None,
                    'first_buy_price': None,
                    'market_hour_category': None,
                    'price_range': None,
                    'year': None,
                    'month': None,
                    'day_of_week': None
                }
            }
            trade_id_counter += 1
        
        position_data = positions[symbol]
        
        debug_print(symbol, f"Processing {side} order for {qty} shares at ${price} at {timestamp}")
        debug_print(symbol, f"Before: Position={position_data['position']}")
        
        # If position was zero and this is a buy, start a new trade
        if position_data['position'] == 0 and side == 'Buy':
            position_data['current_trade_id'] = trade_id_counter
            trade_id_counter += 1
            
            # Reset trade data for the new trade
            position_data['buys'] = []
            position_data['sells'] = []
            position_data['trade_data'] = {
                'first_buy_date': date,
                'last_sell_date': None,
                'first_buy_price': price,
                'market_hour_category': row.get('Market_Hour_Category', 'Unknown'),
                'price_range': row.get('Price_Range', 'Unknown'),
                'year': row.get('Year', None),
                'month': row.get('Month', None),
                'day_of_week': row.get('Day_of_Week', None)
            }
            
            debug_print(symbol, f"Starting new trade with ID {position_data['current_trade_id']}")
        
        # Handle Buy order
        if side == 'Buy':
            # Add to current position
            position_data['position'] += qty
            
            # Store buy information
            position_data['buys'].append({
                'qty': qty,
                'price': price,
                'date': date,
                'timestamp': timestamp,
                'row_data': row.to_dict()
            })
            
            debug_print(symbol, f"After Buy: Position={position_data['position']}")
        
        # Handle Sell order
        elif side == 'Sell':
            # Skip if no position to sell
            if position_data['position'] <= 0:
                debug_print(symbol, f"Skipping sell - no position available")
                continue
                
            # Determine actual sell quantity (can't sell more than position)
            actual_sell_qty = min(qty, position_data['position'])
            
            # Update remaining position
            position_data['position'] -= actual_sell_qty
            
            # Store sell information
            position_data['sells'].append({
                'qty': actual_sell_qty,
                'price': price,
                'date': date,
                'timestamp': timestamp,
                'row_data': row.to_dict()
            })
            
            # Update last sell date
            position_data['trade_data']['last_sell_date'] = date
            
            debug_print(symbol, f"After Sell: Position={position_data['position']}")
            
            # If position is now zero, calculate trade performance and record it
            if position_data['position'] == 0:
                debug_print(symbol, f"Position closed - calculating trade performance")
                
                # Calculate FIFO-based P&L for this trade
                buys = position_data['buys'].copy()
                sells = position_data['sells'].copy()
                
                # Initialize trade performance metrics
                total_buy_qty = sum(b['qty'] for b in buys)
                total_sell_qty = sum(s['qty'] for s in sells)
                total_cost = sum(b['qty'] * b['price'] for b in buys)
                total_revenue = sum(s['qty'] * s['price'] for s in sells)
                
                # Calculate overall P&L
                total_profit_loss = total_revenue - total_cost
                
                # Safety check for division by zero
                if total_buy_qty > 0 and total_cost > 0:
                    avg_buy_price = total_cost / total_buy_qty
                    avg_sell_price = total_revenue / total_sell_qty
                    pnl_per_share = avg_sell_price - avg_buy_price
                    percent_gain_loss = (total_profit_loss / total_cost) * 100
                else:
                    avg_buy_price = avg_sell_price = pnl_per_share = percent_gain_loss = 0
                
                # Determine trade outcome
                trade_outcome = 'Win' if total_profit_loss > 0 else 'Loss' if total_profit_loss < 0 else 'Break Even'
                
                # Get first buy and last sell info
                first_buy = buys[0]
                last_sell = sells[-1]
                
                # Construct trade entry
                trade_entry = {
                    'Trade_ID': position_data['current_trade_id'],
                    'Symbol': symbol,
                    'Total_Buy_Quantity': total_buy_qty,
                    'Total_Sell_Quantity': total_sell_qty,
                    'Avg_Buy_Price': avg_buy_price,
                    'Avg_Sell_Price': avg_sell_price,
                    'PnL_Per_Share': pnl_per_share,
                    'Total_Cost': total_cost,
                    'Total_Revenue': total_revenue,
                    'Total_Profit_Loss': total_profit_loss,
                    'Percent_Gain_Loss': percent_gain_loss,
                    'PnL_%': percent_gain_loss,
                    'Trade_Outcome': trade_outcome,
                    'Num_Buys': len(buys),
                    'Num_Sells': len(sells),
                    'First_Buy_Time': first_buy['timestamp'],
                    'Last_Sell_Time': last_sell['timestamp'],
                    'Trade_Duration_Minutes': (last_sell['timestamp'] - first_buy['timestamp']).total_seconds() / 60,
                    'Price_Range': position_data['trade_data']['price_range'],
                    'Market_Hour_Category': position_data['trade_data']['market_hour_category'],
                    'Year': position_data['trade_data']['year'],
                    'Month': position_data['trade_data']['month'],
                    'Day_of_Week': position_data['trade_data']['day_of_week'],
                    'Start_Date': position_data['trade_data']['first_buy_date'],
                    'End_Date': position_data['trade_data']['last_sell_date']
                }
                
                # Add additional details like buy/sell details if needed
                buy_prices = [f"${b['price']:.2f} x {b['qty']}" for b in buys]
                sell_prices = [f"${s['price']:.2f} x {s['qty']}" for s in sells]
                
                trade_entry['Buy_Details'] = ", ".join(buy_prices)
                trade_entry['Sell_Details'] = ", ".join(sell_prices)
                
                debug_print(symbol, f"Trade completed - P&L: ${total_profit_loss:.2f}, {percent_gain_loss:.2f}%")
                position_trade_analysis.append(trade_entry)
    
    # Process any remaining open positions (for reporting purposes)
    for symbol, position_data in positions.items():
        if position_data['position'] > 0:
            debug_print(symbol, f"Warning: Open position {position_data['position']} shares at end of data")
            
            # Can optionally add these as incomplete trades if desired
    
    # Create dataframe from position-based trade analysis list
    result_df = pd.DataFrame(position_trade_analysis)
    
    # If DataFrame is empty, return an empty DataFrame with the expected columns
    if result_df.empty:
        columns = ['Trade_ID', 'Symbol', 'Total_Buy_Quantity', 'Total_Sell_Quantity', 
                  'Avg_Buy_Price', 'Avg_Sell_Price', 'PnL_Per_Share', 'Total_Cost', 
                  'Total_Revenue', 'Total_Profit_Loss', 'Percent_Gain_Loss', 'PnL_%', 
                  'Trade_Outcome', 'Num_Buys', 'Num_Sells', 'First_Buy_Time', 
                  'Last_Sell_Time', 'Trade_Duration_Minutes']
        return pd.DataFrame(columns=columns)
    
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
        if pd.isna(buy_row.get('Gain/Loss')):
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