import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import streamlit as st
import plotly.subplots as sp

def create_performance_overview(trade_analysis_df, performance_metrics):
    """
    Create performance overview visualizations
    
    Args:
        trade_analysis_df (pd.DataFrame): Trade analysis dataframe
        performance_metrics (dict): Overall performance metrics
    """
    # Retrieve color configuration from session state or set defaults
    if 'win_color' not in st.session_state:
        st.session_state.win_color = "#4CAF50"
    if 'loss_color' not in st.session_state:
        st.session_state.loss_color = "#F44336"
    if 'break_even_color' not in st.session_state:
        st.session_state.break_even_color = "#9E9E9E"
    
    # Create a softer theme configuration
    theme_config = {
        'font': {'family': 'Arial, sans-serif', 'color': '#333'},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'titlefont': {'size': 16, 'color': '#333'},
    }
    
    # Create columns for basic metrics with a softer look - MODIFIED ORDER
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(label="Trading Days", value=performance_metrics['Trading_Days'])
    
    with col2:
        st.metric(label="Total Trades", value=performance_metrics['Total_Trades'])
    
    with col3:
        st.metric(label="Wins", value=performance_metrics['Total_Wins'])
    
    with col4:
        st.metric(label="Losses", value=performance_metrics['Total_Losses'])
    
    with col5:
        st.metric(label="Win Rate", value=f"{performance_metrics['Win_Rate']:.1f}%")
    
    # Add Risk-Reward metrics section
    st.subheader("Risk-Reward Analysis")
    
    # Create columns for Risk-Reward metrics - MODIFIED TO 5 COLUMNS PER ROW
    rr_col1, rr_col2, rr_col3, rr_col4, rr_col5 = st.columns(5)
    
    with rr_col1:
        st.metric(
            label="Reward %",
            value=f"{performance_metrics['Reward_%']:.2f}%",
            help="Average percentage gain on winning trades"
        )
    
    with rr_col2:
        st.metric(
            label="Risk %",
            value=f"{performance_metrics['Risk_%']:.2f}%",
            help="Average percentage loss on losing trades (absolute value)"
        )
    
    with rr_col3:
        st.metric(
            label="Reward $",
            value=f"${performance_metrics['Reward_$']:.2f}",
            help="Average $ gain per share on winning trades"
        )
    
    with rr_col4:
        st.metric(
            label="Risk $",
            value=f"${performance_metrics['Risk_$']:.2f}",
            help="Average $ loss per share on losing trades (absolute value)"
        )
    
    # Risk-Reward Ratio
    with rr_col5:
        st.metric(
            label="Risk-Reward Ratio",
            value=f"1:{performance_metrics['Risk_Reward_Ratio']:.2f}",
            help="Ratio of reward % to risk % (higher is better)"
        )
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    # Trade Outcome Distribution (Pie Chart)
    outcome_dist = trade_analysis_df['Trade_Outcome'].value_counts()
    
    with col1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=outcome_dist.index, 
            values=outcome_dist.values,
            marker_colors=[
                st.session_state.win_color if outcome == 'Win' else 
                st.session_state.loss_color if outcome == 'Loss' else 
                st.session_state.break_even_color
                for outcome in outcome_dist.index
            ]
        )])
        
        fig_pie.update_layout(
            title="Trade Outcome Distribution",
            **theme_config
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Daily Trade Outcomes (Bar Chart)
    with col2:
        # Group by date and count trades
        daily_trades = trade_analysis_df.groupby('Date')['Trade_Outcome'].value_counts().unstack(fill_value=0)
        
        # Remove dates with no trades
        daily_trades = daily_trades[(daily_trades.T != 0).any()]
        
        # Ensure all columns exist
        for outcome in ['Win', 'Loss', 'Break Even']:
            if outcome not in daily_trades.columns:
                daily_trades[outcome] = 0
        
        # Create stacked bar chart
        fig_daily = go.Figure(data=[
            go.Bar(name='Wins', x=daily_trades.index.astype(str), y=daily_trades['Win'], marker_color=st.session_state.win_color),
            go.Bar(name='Losses', x=daily_trades.index.astype(str), y=daily_trades['Loss'], marker_color=st.session_state.loss_color),
            go.Bar(name='Break Even', x=daily_trades.index.astype(str), y=daily_trades['Break Even'], marker_color=st.session_state.break_even_color)
        ])
        
        fig_daily.update_layout(
            barmode='stack', 
            title='Daily Trade Outcomes',
            xaxis_title='Date',
            yaxis_title='Number of Trades',
            xaxis_type='category',
            **theme_config
        )
        
        st.plotly_chart(fig_daily, use_container_width=True)

def create_dimensional_analysis(trade_analysis_df):
    """
    Create dimensional performance analysis visualizations
    
    Args:
        trade_analysis_df (pd.DataFrame): Trade analysis dataframe
    """
    # Add color configuration to sidebar
    st.sidebar.header("Chart Color Customization")
    st.session_state.win_color = st.sidebar.color_picker(
        "Win Color", 
        value=st.session_state.win_color, 
        key="win_color_picker"
    )
    st.session_state.loss_color = st.sidebar.color_picker(
        "Loss Color", 
        value=st.session_state.loss_color, 
        key="loss_color_picker"
    )
    st.session_state.break_even_color = st.sidebar.color_picker(
        "Break Even Color", 
        value=st.session_state.break_even_color, 
        key="break_even_color_picker"
    )
    
    # Theme configuration
    theme_config = {
        'font': {'family': 'Arial, sans-serif', 'color': '#333'},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'titlefont': {'size': 16, 'color': '#333'},
    }
    
    # Prepare plot configurations with win/loss counts
    plot_configurations = [
        {
            'dimension': 'Price_Range',
            'title': 'Trades by Price Range',
            'sort_func': 'sort_index',
            'order': [
                '0-2', '2-4', '4-10', '10-15', '15-20', 
                '20-30', '30-40', '40-60', '>60'
            ]
        },
        {
            'dimension': 'Market_Hour_Category',
            'title': 'Trades by Market Hour Category',
            'sort_func': None,
            'order': ['Pre-Market Hour', 'Regular Hour', 'Post-Market Hour']
        },
        {
            'dimension': 'Month',
            'title': 'Trades by Month',
            'sort_func': 'sort_index',
            'order': [
                'January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
        },
        {
            'dimension': 'Day_of_Week',
            'title': 'Trades by Day of Week',
            'sort_func': 'sort_index',
            'order': [
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
            ]
        }
    ]
    
    # Create columns for charts
    columns = st.columns(2)
    
    # Create and display plots for each dimension
    for i, config in enumerate(plot_configurations):
        # Determine which column to use
        col = columns[i % 2]
        
        # Group by dimension and count wins/losses
        trades_by_dimension = trade_analysis_df.groupby([config['dimension'], 'Trade_Outcome']).size().unstack(fill_value=0)
        
        # Ensure all outcome columns exist
        for outcome in ['Win', 'Loss', 'Break Even']:
            if outcome not in trades_by_dimension.columns:
                trades_by_dimension[outcome] = 0
        
        # Apply sorting if specified
        if config['order']:
            # Reindex with the specified order, filling missing values with 0
            trades_by_dimension = trades_by_dimension.reindex(config['order'], fill_value=0)
        
        # Create stacked bar plot
        with col:
            fig = go.Figure(data=[
                go.Bar(name='Wins', x=trades_by_dimension.index, y=trades_by_dimension['Win'], marker_color=st.session_state.win_color),
                go.Bar(name='Losses', x=trades_by_dimension.index, y=trades_by_dimension['Loss'], marker_color=st.session_state.loss_color),
                go.Bar(name='Break Even', x=trades_by_dimension.index, y=trades_by_dimension['Break Even'], marker_color=st.session_state.break_even_color)
            ])
            
            fig.update_layout(
                barmode='stack', 
                title=config['title'],
                xaxis_title=config['dimension'],
                yaxis_title='Number of Trades',
                **theme_config
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)

def create_detailed_trade_table(trade_analysis_df):
    """
    Create a detailed trade performance table
    
    Args:
        trade_analysis_df (pd.DataFrame): Trade analysis dataframe
    """
    # Check if this is position-based analysis (it has Trade_ID and additional columns)
    is_position_based = 'Trade_ID' in trade_analysis_df.columns and 'Buy_Details' in trade_analysis_df.columns
    
    st.subheader("Detailed Trade Performance")
    
    # Determine which field names to use for display based on availability
    profit_loss_field = 'Total_Profit_Loss' if 'Total_Profit_Loss' in trade_analysis_df.columns else 'Profit_Loss'
    percent_gain_loss_field = 'PnL_%' if 'PnL_%' in trade_analysis_df.columns else 'Percent_Gain_Loss'
    
    # Columns to display - adapt based on analysis type
    if is_position_based:
        display_columns = [
            'Trade_ID', 'Symbol', 'Buy_Quantity', 'Avg_Buy_Price', 'Avg_Sell_Price', 
            'PnL_Per_Share', percent_gain_loss_field,
            'Total_Cost', 'Total_Revenue', profit_loss_field,
            'Trade_Outcome', 'Num_Buys', 'Num_Sells', 'Trade_Duration_Minutes',
            'Price_Range', 'Market_Hour_Category', 'Month', 'Day_of_Week', 
            'Buy_Details', 'Sell_Details'
        ]
    else:
        display_columns = [
            'Symbol', 'Buy_Quantity', 'Avg_Buy_Price', 'Avg_Sell_Price', 
            'PnL_Per_Share', percent_gain_loss_field,
            'Total_Cost', 'Total_Revenue', profit_loss_field,
            'Trade_Outcome', 'Price_Range', 
            'Market_Hour_Category', 'Month', 'Day_of_Week', 'DateTime'
        ]
    
    # Filter to only include columns that actually exist in the dataframe
    display_columns = [col for col in display_columns if col in trade_analysis_df.columns]
    
    # Format the dataframe for display
    display_df = trade_analysis_df[display_columns].copy()
    
    # Store original numeric values for styling
    numeric_pnl_per_share = trade_analysis_df['PnL_Per_Share'].copy()
    numeric_profit_loss = trade_analysis_df[profit_loss_field].copy()
    numeric_percent_gain_loss = trade_analysis_df[percent_gain_loss_field].copy()
    
    # Format numeric columns
    display_df['Avg_Buy_Price'] = display_df['Avg_Buy_Price'].apply(lambda x: f"${x:.2f}")
    display_df['Avg_Sell_Price'] = display_df['Avg_Sell_Price'].apply(lambda x: f"${x:.2f}")
    display_df['Total_Cost'] = display_df['Total_Cost'].apply(lambda x: f"${x:.2f}")
    display_df['Total_Revenue'] = display_df['Total_Revenue'].apply(lambda x: f"${x:.2f}")
    
    # Format P&L columns with dollar signs and percent signs
    display_df['PnL_Per_Share'] = display_df['PnL_Per_Share'].apply(
        lambda x: f"${x:.2f}" if x >= 0 else f"-${abs(x):.2f}"
    )
    display_df[profit_loss_field] = display_df[profit_loss_field].apply(
        lambda x: f"${x:.2f}" if x >= 0 else f"-${abs(x):.2f}"
    )
    display_df[percent_gain_loss_field] = display_df[percent_gain_loss_field].apply(
        lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
    )
    
    # Format additional position-based columns if they exist
    if is_position_based and 'Trade_Duration_Minutes' in display_df.columns:
        # Format duration as hours:minutes
        display_df['Trade_Duration_Minutes'] = display_df['Trade_Duration_Minutes'].apply(
            lambda x: f"{int(x // 60)}h {int(x % 60)}m" if x >= 60 else f"{int(x)}m"
        )
    
    # Create a function to apply styling based on value
    def style_dataframe(df):
        # Get the win/loss colors from session state
        win_color = getattr(st.session_state, 'win_color', "#4CAF50")
        loss_color = getattr(st.session_state, 'loss_color', "#F44336")
        
        # Create a styler with empty strings
        styler = pd.DataFrame('', index=df.index, columns=df.columns)
        
        # Apply color to PnL_Per_Share column based on values
        for i, val in enumerate(numeric_pnl_per_share):
            if val > 0:
                styler.iloc[i, df.columns.get_loc('PnL_Per_Share')] = f'color: {win_color}'
            elif val < 0:
                styler.iloc[i, df.columns.get_loc('PnL_Per_Share')] = f'color: {loss_color}'
        
        # Apply color to Total_Profit_Loss/Profit_Loss column based on values
        for i, val in enumerate(numeric_profit_loss):
            if val > 0:
                styler.iloc[i, df.columns.get_loc(profit_loss_field)] = f'color: {win_color}'
            elif val < 0:
                styler.iloc[i, df.columns.get_loc(profit_loss_field)] = f'color: {loss_color}'
        
        # Apply color to PnL_%/Percent_Gain_Loss column based on values
        for i, val in enumerate(numeric_percent_gain_loss):
            if val > 0:
                styler.iloc[i, df.columns.get_loc(percent_gain_loss_field)] = f'color: {win_color}'
            elif val < 0:
                styler.iloc[i, df.columns.get_loc(percent_gain_loss_field)] = f'color: {loss_color}'
        
        return styler
    
    # Apply styling and display the dataframe
    st.dataframe(
        display_df.style.apply(style_dataframe, axis=None),
        use_container_width=True
    )