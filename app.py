import streamlit as st
import pandas as pd
import io


# Import custom modules
from src.data_processor import load_trading_data_from_bytes, prepare_trade_analysis
from src.performance_analysis import (
    calculate_overall_performance, 
    performance_by_dimension, 
    detailed_trade_statistics
)
from src.visualizations import (
    create_performance_overview, 
    create_dimensional_analysis, 
    create_detailed_trade_table
)
from src.calendar_view import create_calendar_view  # Add import for new calendar view

def main():
    """
    Main Streamlit application for trading performance dashboard
    """
    # Set page configuration
    st.set_page_config(
        page_title="Trading Performance Dashboard", 
        page_icon=":chart_with_upwards_trend:", 
        layout="wide"
    )
    
    # Title and introduction
    st.title("📊 Trading Performance Dashboard")
    st.write("Comprehensive analysis of your trading performance")
    
    # File uploader - support both CSV and Excel files
    uploaded_file = st.file_uploader(
        "Upload your Trading Records (CSV or Excel)", 
        type=['csv', 'xlsx', 'xls'], 
        help="Please upload your trading records file (Webull CSV or Rachel's Trading Log Excel)"
    )
    
    if uploaded_file is not None:
        # Process trading data directly from uploaded file
        try:
            # Get file name and extension for format detection
            file_name = uploaded_file.name
            
            # Load the trading data directly from memory
            raw_df = load_trading_data_from_bytes(uploaded_file.getvalue(), file_name)
            
            # Add sidebar filters
            st.sidebar.header("Filter Options")
            
            # Prepare trade analysis first to ensure Percent_Gain_Loss is calculated
            full_trade_analysis_df = prepare_trade_analysis(raw_df)
            
            # Date Range Filter
            min_date = full_trade_analysis_df['DateTime'].min().date()
            max_date = full_trade_analysis_df['DateTime'].max().date()
            
            date_range = st.sidebar.date_input(
                "Select Date Range", 
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # Market Hour Category Filter
            market_hour_categories = full_trade_analysis_df['Market_Hour_Category'].unique().tolist()
            selected_market_hours = st.sidebar.multiselect(
                "Select Market Hour Categories",
                options=market_hour_categories,
                default=market_hour_categories
            )
            
            # Percent Gain/Loss Filter - Use Percent_Gain_Loss for compatibility
            # We'll keep using the original name in filters to ensure backward compatibility
            min_gain_loss = full_trade_analysis_df['Percent_Gain_Loss'].min()
            max_gain_loss = full_trade_analysis_df['Percent_Gain_Loss'].max()
            
            gain_loss_range = st.sidebar.slider(
                "Percent Gain/Loss Range",
                min_value=float(min_gain_loss),
                max_value=float(max_gain_loss),
                value=(float(min_gain_loss), float(max_gain_loss))
            )
            
            # Filter trade analysis dataframe
            filtered_trade_analysis_df = full_trade_analysis_df[
                (full_trade_analysis_df['DateTime'].dt.date >= date_range[0]) & 
                (full_trade_analysis_df['DateTime'].dt.date <= date_range[1]) &
                (full_trade_analysis_df['Market_Hour_Category'].isin(selected_market_hours)) &
                (full_trade_analysis_df['Percent_Gain_Loss'] >= gain_loss_range[0]) &
                (full_trade_analysis_df['Percent_Gain_Loss'] <= gain_loss_range[1])
            ]
            
            # Calculate overall performance metrics
            performance_metrics = calculate_overall_performance(filtered_trade_analysis_df)
            
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["Performance Overview", "Calendar View", "Detailed Trades"])
            
            # Tab 1: Performance Overview
            with tab1:
                # Performance Overview Section
                create_performance_overview(filtered_trade_analysis_df, performance_metrics)
                
                # Dimensional Analysis Section
                create_dimensional_analysis(filtered_trade_analysis_df)
            
            # Tab 2: Calendar View (New)
            with tab2:
                create_calendar_view(filtered_trade_analysis_df)
            
            # Tab 3: Detailed Trade Table
            with tab3:
                create_detailed_trade_table(filtered_trade_analysis_df)
        
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            st.exception(e)  # Show detailed error information
    else:
        st.info("Please upload a trading records file (CSV or Excel) to begin analysis.")

if __name__ == "__main__":
    main()