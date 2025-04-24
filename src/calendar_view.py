import pandas as pd
import streamlit as st
import calendar
import plotly.graph_objs as go
from datetime import datetime
import numpy as np

def create_calendar_view(trade_analysis_df):
    """
    Create a calendar view of trading performance
    
    Args:
        trade_analysis_df (pd.DataFrame): Trade analysis dataframe
    """
    st.subheader("📅 Calendar View")
    
    # Determine which field names to use for calculations based on availability
    profit_loss_field = 'Total_Profit_Loss' if 'Total_Profit_Loss' in trade_analysis_df.columns else 'Profit_Loss'
    percent_gain_loss_field = 'PnL_%' if 'PnL_%' in trade_analysis_df.columns else 'Percent_Gain_Loss'
    
    # Get unique months and years in the data
    trade_analysis_df['Year_Month'] = trade_analysis_df['DateTime'].dt.strftime('%Y-%m')
    unique_year_months = sorted(trade_analysis_df['Year_Month'].unique())
    
    # Create a dropdown for selecting month-year
    selected_year_month = st.selectbox(
        "Select Month",
        options=unique_year_months,
        index=len(unique_year_months)-1  # Default to most recent month
    )
    
    # Filter data for selected month
    year, month = map(int, selected_year_month.split('-'))
    month_data = trade_analysis_df[
        (trade_analysis_df['DateTime'].dt.year == year) &
        (trade_analysis_df['DateTime'].dt.month == month)
    ]
    
    # Calculate monthly total
    monthly_profit_loss = month_data[profit_loss_field].sum()
    monthly_trades = len(month_data)
    monthly_win_rate = (month_data['Trade_Outcome'] == 'Win').mean() * 100 if monthly_trades > 0 else 0
    
    # Calculate monthly risk-reward metrics
    monthly_win_trades = month_data[month_data['Trade_Outcome'] == 'Win']
    monthly_loss_trades = month_data[month_data['Trade_Outcome'] == 'Loss']
    
    if not monthly_win_trades.empty:
        monthly_reward_percent = monthly_win_trades[percent_gain_loss_field].mean()
        monthly_reward_per_share = monthly_win_trades['PnL_Per_Share'].mean()
    else:
        monthly_reward_percent = 0
        monthly_reward_per_share = 0
        
    if not monthly_loss_trades.empty:
        monthly_risk_percent = abs(monthly_loss_trades[percent_gain_loss_field].mean())
        monthly_risk_per_share = abs(monthly_loss_trades['PnL_Per_Share'].mean())
    else:
        monthly_risk_percent = 0
        monthly_risk_per_share = 0
    
    # Calculate risk-reward ratio
    monthly_rr_ratio = monthly_reward_percent / monthly_risk_percent if monthly_risk_percent > 0 else 0
    
    # Display monthly summary
    st.markdown(f"### {calendar.month_name[month]} {year}")
    
    monthly_pl_color = "green" if monthly_profit_loss >= 0 else "red"
    monthly_pl_formatted = f"+${monthly_profit_loss:.2f}" if monthly_profit_loss >= 0 else f"-${abs(monthly_profit_loss):.2f}"
    
    st.markdown(f"<h4 style='color:{monthly_pl_color};'>Monthly P&L: {monthly_pl_formatted}</h4>", unsafe_allow_html=True)
    
    # Display monthly risk-reward metrics
    monthly_metrics_col1, monthly_metrics_col2, monthly_metrics_col3 = st.columns(3)
    
    with monthly_metrics_col1:
        st.metric("Monthly Win Rate", f"{monthly_win_rate:.1f}%")
    
    with monthly_metrics_col2:
        st.metric("Reward:Risk Ratio", f"1:{monthly_rr_ratio:.2f}" if monthly_rr_ratio > 0 else "N/A")
    
    with monthly_metrics_col3:
        # Format as a table for reward/risk metrics
        metrics_html = f"""
        <div style="border:1px solid #ddd; border-radius:5px; padding:10px;">
            <div style="display:flex; justify-content:space-between;">
                <div><strong>Reward %:</strong> {monthly_reward_percent:.2f}%</div>
                <div><strong>Risk %:</strong> {monthly_risk_percent:.2f}%</div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:5px;">
                <div><strong>Reward $:</strong> ${monthly_reward_per_share:.2f}</div>
                <div><strong>Risk $:</strong> ${monthly_risk_per_share:.2f}</div>
            </div>
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)
    
    # Create dictionary for daily data
    days_in_month = calendar.monthrange(year, month)[1]
    day_data = {}
    
    for day in range(1, days_in_month + 1):
        date_str = f"{year}-{month:02d}-{day:02d}"
        date_obj = pd.to_datetime(date_str).date()
        
        # Get data for this day
        day_df = month_data[month_data['DateTime'].dt.date == date_obj]
        
        # Calculate metrics
        num_trades = len(day_df)
        profit_loss = day_df[profit_loss_field].sum() if num_trades > 0 else 0
        win_rate = (day_df['Trade_Outcome'] == 'Win').mean() * 100 if num_trades > 0 else 0
        
        # Calculate daily risk-reward metrics
        win_trades = day_df[day_df['Trade_Outcome'] == 'Win']
        loss_trades = day_df[day_df['Trade_Outcome'] == 'Loss']
        
        reward_percent = win_trades[percent_gain_loss_field].mean() if not win_trades.empty else 0
        risk_percent = abs(loss_trades[percent_gain_loss_field].mean()) if not loss_trades.empty else 0
        
        # Risk-reward ratio
        rr_ratio = reward_percent / risk_percent if risk_percent > 0 else 0
        
        day_data[day] = {
            'date': date_obj,
            'num_trades': num_trades,
            'profit_loss': profit_loss,
            'win_rate': win_rate,
            'rr_ratio': rr_ratio,  # Use R:R instead of avg_gain
            'weekday': date_obj.strftime('%A')
        }
    
    # Get the first day of the month
    first_day = datetime(year, month, 1).weekday()
    
    # Create calendar layout 
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Create weekly data for display
    weekly_data = {}
    for week_num in range(6):  # Maximum 6 weeks in a month view
        weekly_data[week_num] = {
            'num_trades': 0,
            'profit_loss': 0,
        }
    
    # Create calendar using Streamlit columns
    st.markdown("---")
    
    # Create header row with day names and week summary header
    cols = st.columns(8)  # 7 days + 1 column for week summary
    for i, day_name in enumerate(weekdays):
        cols[i].markdown(f"<div style='text-align: center; font-weight: bold;'>{day_name}</div>", unsafe_allow_html=True)
    cols[7].markdown(f"<div style='text-align: center; font-weight: bold;'>Week Summary</div>", unsafe_allow_html=True)
    
    # Create calendar grid
    day_counter = 1
    for week in range(6):  # Maximum 6 weeks in a month view
        cols = st.columns(8)  # 7 days + 1 column for week summary
        week_trades = 0
        week_pl = 0
        
        for weekday in range(7):
            if (week == 0 and weekday < first_day) or (day_counter > days_in_month):
                # Empty cell
                cols[weekday].markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
            else:
                # Get day's data
                data = day_data[day_counter]
                num_trades = data['num_trades']
                profit_loss = data['profit_loss']
                win_rate = data['win_rate']
                rr_ratio = data['rr_ratio']  # Use R:R instead of avg_gain
                
                # Add to weekly totals
                week_trades += num_trades
                week_pl += profit_loss
                
                # Format the profit/loss with color and sign
                if profit_loss == 0:
                    pl_text = "$0"
                    pl_color = "gray"
                elif profit_loss > 0:
                    pl_text = f"+${profit_loss:.2f}"
                    pl_color = "green"
                else:
                    pl_text = f"-${abs(profit_loss):.2f}"
                    pl_color = "red"
                
                # Format win rate and R:R
                win_rate_text = f"{win_rate:.1f}%" if num_trades > 0 else "N/A"
                rr_ratio_text = f"1:{rr_ratio:.2f}" if rr_ratio > 0 and num_trades > 0 else "N/A"
                
                # Create cell content with all metrics
                cell_content = f"""
                <div style='border: 1px solid #ddd; border-radius: 5px; padding: 8px; height: 120px;'>
                    <div style='font-weight: bold;'>{day_counter}</div>
                    <div style='color: {pl_color}; font-weight: bold;'>{pl_text}</div>
                    <div style='font-size: 0.8em;'>{num_trades} trades</div>
                    <div style='font-size: 0.8em;'>WR: {win_rate_text}</div>
                    <div style='font-size: 0.8em;'>R:R: {rr_ratio_text}</div>
                </div>
                """
                cols[weekday].markdown(cell_content, unsafe_allow_html=True)
                
                day_counter += 1
        
        # Update weekly data
        weekly_data[week]['num_trades'] = week_trades
        weekly_data[week]['profit_loss'] = week_pl
        
        # Add weekly summary in the last column
        # Calculate week number regardless of whether there are trades
        if week == 0:
            week_label = "Week 1"
        elif week == 1:
            week_label = "Week 2"
        elif week == 2:
            week_label = "Week 3"
        elif week == 3:
            week_label = "Week 4"
        elif week == 4:
            week_label = "Week 5"
        else:
            week_label = "Week 6"
        
        # Check if we have days rendered in this week
        has_days_in_week = False
        if week == 0:
            has_days_in_week = (day_counter <= days_in_month)
        else:
            has_days_in_week = (day_counter - 1 <= days_in_month) and (day_counter > 1)
        
        if has_days_in_week:
            # Format weekly P&L
            if week_pl >= 0:
                week_pl_text = f"+${week_pl:.2f}"
                week_pl_color = "green"
            else:
                week_pl_text = f"-${abs(week_pl):.2f}"
                week_pl_color = "red"
            
            if week_pl == 0:
                week_pl_text = "$0"
                week_pl_color = "gray"
            
            # Display weekly summary in the last column
            weekly_summary = f"""
            <div style='border: 1px solid #ddd; border-radius: 5px; padding: 8px; height: 120px;'>
                <div style='font-weight: bold;'>{week_label}</div>
                <div style='color: {week_pl_color}; font-weight: bold;'>{week_pl_text}</div>
                <div style='font-size: 0.8em;'>{week_trades} trades</div>
            </div>
            """
            cols[7].markdown(weekly_summary, unsafe_allow_html=True)
        else:
            # Empty cell for week with no days
            cols[7].markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
        
        st.markdown("---")