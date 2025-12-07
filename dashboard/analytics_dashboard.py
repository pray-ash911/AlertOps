# dashboard.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json

# Set page config
st.set_page_config(
    page_title="AI Surveillance Analyticss Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #374151;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3B82F6;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .weapon-card {
        border-left: 5px solid #DC2626;
    }
    .crowd-card {
        border-left: 5px solid #10B981;
    }
    .total-card {
        border-left: 5px solid #8B5CF6;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1F2937;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
        margin-top: 0.5rem;
    }
    .alert-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .weapon-alert {
        background-color: #FEE2E2;
        color: #DC2626;
        border: 1px solid #FCA5A5;
    }
    .crowd-alert {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #6EE7B7;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .left-align {
        text-align: left !important;
    }
    .right-align {
        text-align: right !important;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    .hour-stat-box {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        border: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
LOCAL_URL = "http://127.0.0.1:8000"
ANALYTICS_URL = f"{LOCAL_URL}/api/analytics/"

# Title
st.markdown('<h1 class="main-header">Weapon & Overcrowding Detection Analytics Dashboard</h1>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("Dashboard Configuration")
st.sidebar.markdown("---")

# Data source selection
data_source = st.sidebar.radio(
    "Select Data Source:",
    ["Live API", "Sample Data"],
    index=0
)

# Date range in sidebar
st.sidebar.header("Analysis Period")
days_back = st.sidebar.slider(
    "Days to analyze:",
    min_value=7,
    max_value=90,
    value=30,
    step=1
)

# Event type filter
st.sidebar.header("Event Type Filter")
show_weapons = st.sidebar.checkbox("Show Weapon Detections", value=True)
show_crowd = st.sidebar.checkbox("Show Overcrowding Events", value=True)

# Chart settings
st.sidebar.header("Chart Settings")
chart_height = st.sidebar.slider("Chart Height", 300, 600, 400)

# Refresh button
st.sidebar.markdown("---")
if st.sidebar.button("Refresh Data", width='stretch'):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Dashboard Information\n\n• Shows weapon and overcrowding detection analytics\n• Data updates automatically\n• Click refresh to get latest data")

# Fetch data function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_analytics_data():
    try:
        response = requests.get(ANALYTICS_URL, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None

# Load data
if data_source == "Live API":
    with st.spinner("Fetching data from API..."):
        data = fetch_analytics_data()
    
    if data is None:
        st.warning("Could not connect to API. Using sample data.")
        data_source = "Sample Data"

if data_source == "Sample Data" or data is None:
    # Generate comprehensive sample data
    dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
    analytics_data = []
    
    for i, date in enumerate(dates):
        # Create realistic patterns
        day_of_week = date.weekday()
        
        # Weekends have more events
        if day_of_week >= 5:  # Saturday, Sunday
            weapon = np.random.poisson(lam=2.5)
            crowd = np.random.poisson(lam=8)
        # Fridays moderate-high
        elif day_of_week == 4:  # Friday
            weapon = np.random.poisson(lam=2.0)
            crowd = np.random.poisson(lam=6)
        # Weekdays
        else:
            weapon = np.random.poisson(lam=1.2)
            crowd = np.random.poisson(lam=4)
        
        # Add some trend/pattern
        if i > days_back * 0.7:  # Recent days have more activity
            weapon = min(weapon + np.random.randint(0, 2), 5)
            crowd = min(crowd + np.random.randint(0, 3), 12)
        
        analytics_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'weapon': int(weapon),
            'overcrowding': int(crowd),
            'total_detections': int(weapon + crowd)
        })
    
    # Create sample data structure
    data = {
        'daily_analytics': analytics_data,
        'hourly_analytics': [
            {
                'hour': hour,
                'weapon': max(0, int(np.random.normal(loc=1.5 if 9 <= hour <= 21 else 0.3, scale=0.5))),
                'overcrowding': max(0, int(np.random.normal(loc=4 if 9 <= hour <= 21 else 1, scale=1.5))),
                'total': 0
            }
            for hour in range(24)
        ],
        'summary': {
            'total_weapons': sum(item['weapon'] for item in analytics_data),
            'total_overcrowding': sum(item['overcrowding'] for item in analytics_data),
            'total_all': sum(item['weapon'] + item['overcrowding'] for item in analytics_data),
            'avg_daily_weapons': round(sum(item['weapon'] for item in analytics_data) / len(analytics_data), 1),
            'avg_daily_crowd': round(sum(item['overcrowding'] for item in analytics_data) / len(analytics_data), 1),
            'today_weapon': analytics_data[-1]['weapon'],
            'today_crowd': analytics_data[-1]['overcrowding'],
            'peak_hour': "14:00",
            'peak_hour_weapon': 3,
            'peak_hour_crowd': 9
        }
    }

# Convert to DataFrames
daily_df = pd.DataFrame(data['daily_analytics'])
daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df['day_of_week'] = daily_df['date'].dt.day_name()
daily_df['week_number'] = daily_df['date'].dt.isocalendar().week
daily_df['month'] = daily_df['date'].dt.strftime('%Y-%m')

hourly_df = pd.DataFrame(data['hourly_analytics'])

# Display data source info
source_info = "Sample Data" if data_source == "Sample Data" else "Live API Data"
st.info(f"Data Source: {source_info} | Period: Last {days_back} days | Last Updated: {datetime.now().strftime('%H:%M:%S')}")

# Row 1: Key Metrics
st.markdown('<h2 class="sub-header">Key Performance Indicators</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    total_weapons = data['summary']['total_weapons']
    st.markdown(f"""
    <div class="metric-card weapon-card">
        <div class="metric-value">{total_weapons}</div>
        <div class="metric-label">Total Weapons Detected</div>
        <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 0.5rem;">
            Avg: {data['summary']['avg_daily_weapons']}/day
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_crowd = data['summary']['total_overcrowding']
    st.markdown(f"""
    <div class="metric-card crowd-card">
        <div class="metric-value">{total_crowd}</div>
        <div class="metric-label">Overcrowding Events</div>
        <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 0.5rem;">
            Avg: {data['summary']['avg_daily_crowd']}/day
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    total_all = data['summary']['total_all']
    st.markdown(f"""
    <div class="metric-card total-card">
        <div class="metric-value">{total_all}</div>
        <div class="metric-label">Total Security Events</div>
        <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 0.5rem;">
            Combined detection count
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    today_weapon = data['summary']['today_weapon']
    st.markdown(f"""
    <div class="metric-card weapon-card">
        <div class="metric-value">{today_weapon}</div>
        <div class="metric-label">Today's Weapons</div>
        <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 0.5rem;">
            Current day detection
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    today_crowd = data['summary']['today_crowd']
    st.markdown(f"""
    <div class="metric-card crowd-card">
        <div class="metric-value">{today_crowd}</div>
        <div class="metric-label">Today's Crowd Events</div>
        <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 0.5rem;">
            Current day events
        </div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    peak_hour = data['summary']['peak_hour']
    st.markdown(f"""
    <div class="metric-card total-card">
        <div class="metric-value">{peak_hour}</div>
        <div class="metric-label">Peak Activity Hour</div>
        <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 0.5rem;">
            Weapons: {data['summary'].get('peak_hour_weapon', 0)} | Crowd: {data['summary'].get('peak_hour_crowd', 0)}
        </div>
    </div>
    """, unsafe_allow_html=True)

# Row 2: Daily Trends
st.markdown('<h2 class="sub-header">Daily Trends Analysis</h2>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Daily Chart", "Comparison View", "Statistics"])

with tab1:
    # Bar chart for daily trends
    fig_daily = go.Figure()

    if show_weapons and 'weapon' in daily_df.columns:
        fig_daily.add_trace(go.Bar(
            x=daily_df['date'],
            y=daily_df['weapon'],
            name='Weapon Detections',
            marker_color='red',
            opacity=0.7,
            hovertemplate='Date: %{x}<br>Weapons: %{y}<extra></extra>'
        ))

    if show_crowd and 'overcrowding' in daily_df.columns:
        fig_daily.add_trace(go.Bar(
            x=daily_df['date'],
            y=daily_df['overcrowding'],
            name='Overcrowding Events',
            marker_color='blue',
            opacity=0.7,
            hovertemplate='Date: %{x}<br>Crowd Events: %{y}<extra></extra>'
        ))

    fig_daily.update_layout(
        title=f"Daily Detection Trends (Last {days_back} Days)",
        xaxis_title="Date",
        yaxis_title="Number of Events",
        template="plotly_white",
        height=chart_height,
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='rgba(255, 255, 255, 0.9)',
        font=dict(family="Arial, sans-serif")
    )

    # Add grid
    fig_daily.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig_daily.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    st.plotly_chart(fig_daily, width='stretch')

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Stacked area chart
        fig_area = go.Figure()
        
        if show_weapons and 'weapon' in daily_df.columns:
            fig_area.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['weapon'],
                mode='lines',
                name='Weapons',
                stackgroup='one',
                line=dict(color='red', width=0),
                fillcolor='rgba(255, 0, 0, 0.4)',
                hovertemplate='Date: %{x}<br>Weapons: %{y}<extra></extra>'
            ))
        
        if show_crowd and 'overcrowding' in daily_df.columns:
            fig_area.add_trace(go.Scatter(
                x=daily_df['date'],
                y=daily_df['overcrowding'],
                mode='lines',
                name='Overcrowding',
                stackgroup='one',
                line=dict(color='blue', width=0),
                fillcolor='rgba(0, 0, 255, 0.4)',
                hovertemplate='Date: %{x}<br>Crowd Events: %{y}<extra></extra>'
            ))
        
        fig_area.update_layout(
            title="Stacked Daily Events",
            xaxis_title="Date",
            yaxis_title="Number of Events",
            template="plotly_white",
            height=chart_height - 50,
            showlegend=True
        )

        st.plotly_chart(fig_area, width='stretch')
    
    with col2:
        # Bar chart for comparison
        fig_bar = go.Figure()
        
        # Show last 14 days for better visibility
        recent_df = daily_df.tail(14)
        
        if show_weapons:
            fig_bar.add_trace(go.Bar(
                x=recent_df['date'],
                y=recent_df['weapon'],
                name='Weapons',
                marker_color='red',
                opacity=0.8,
                hovertemplate='Date: %{x}<br>Weapons: %{y}<extra></extra>'
            ))
        
        if show_crowd:
            fig_bar.add_trace(go.Bar(
                x=recent_df['date'],
                y=recent_df['overcrowding'],
                name='Overcrowding',
                marker_color='blue',
                opacity=0.8,
                hovertemplate='Date: %{x}<br>Crowd Events: %{y}<extra></extra>'
            ))
        
        fig_bar.update_layout(
            title="Recent 14 Days Comparison",
            xaxis_title="Date",
            yaxis_title="Count",
            barmode='group',
            template="plotly_white",
            height=chart_height - 50,
            showlegend=True
        )

        st.plotly_chart(fig_bar, width='stretch')

with tab3:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Weapon Statistics")
        if 'weapon' in daily_df.columns:
            st.metric("Average Daily", f"{daily_df['weapon'].mean():.2f}")
            st.metric("Maximum Daily", f"{daily_df['weapon'].max():.0f}")
            st.metric("Minimum Daily", f"{daily_df['weapon'].min():.0f}")
            st.metric("Standard Deviation", f"{daily_df['weapon'].std():.2f}")
            st.metric("Days with Weapons", f"{(daily_df['weapon'] > 0).sum():.0f}")
    
    with col2:
        st.markdown("### Crowd Statistics")
        if 'overcrowding' in daily_df.columns:
            st.metric("Average Daily", f"{daily_df['overcrowding'].mean():.2f}")
            st.metric("Maximum Daily", f"{daily_df['overcrowding'].max():.0f}")
            st.metric("Minimum Daily", f"{daily_df['overcrowding'].min():.0f}")
            st.metric("Standard Deviation", f"{daily_df['overcrowding'].std():.2f}")
            st.metric("Days with Crowd Events", f"{(daily_df['overcrowding'] > 0).sum():.0f}")
    
    with col3:
        st.markdown("### Combined Statistics")
        if 'total_detections' in daily_df.columns:
            st.metric("Total Events", f"{daily_df['total_detections'].sum():.0f}")
            st.metric("Average Total Daily", f"{daily_df['total_detections'].mean():.2f}")
            correlation = daily_df['weapon'].corr(daily_df['overcrowding']) if 'weapon' in daily_df.columns and 'overcrowding' in daily_df.columns else 0
            st.metric("Weapon-Crowd Correlation", f"{correlation:.3f}")
            st.metric("Highest Alert Day", f"{daily_df['total_detections'].max():.0f}")
            st.metric("Alert-Free Days", f"{(daily_df['total_detections'] == 0).sum():.0f}")

# Row 3: Hourly Patterns
st.markdown('<h2 class="sub-header">Hourly Activity Patterns</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    # Hourly bar chart with dual y-axis
    fig_hourly = go.Figure()
    
    # Create secondary y-axis for weapons
    fig_hourly = make_subplots(specs=[[{"secondary_y": True}]])
    
    if show_weapons:
        fig_hourly.add_trace(
            go.Bar(
                x=hourly_df['hour'],
                y=hourly_df['weapon'],
                name='Weapons',
                marker_color='red',
                opacity=0.7,
                hovertemplate='Hour: %{x}:00<br>Weapons: %{y}<extra></extra>'
            ),
            secondary_y=False,
        )
    
    if show_crowd:
        fig_hourly.add_trace(
            go.Bar(
                x=hourly_df['hour'],
                y=hourly_df['overcrowding'],
                name='Overcrowding',
                marker_color='blue',
                opacity=0.7,
                hovertemplate='Hour: %{x}:00<br>Crowd Events: %{y}<extra></extra>'
            ),
            secondary_y=True,
        )
    
    # Set x-axis properties
    fig_hourly.update_xaxes(
        title_text="Hour of Day",
        tickmode='array',
        tickvals=list(range(0, 24, 2)),
        ticktext=[f"{h:02d}:00" for h in range(0, 24, 2)]
    )
    
    # Set y-axes properties
    fig_hourly.update_yaxes(
        title_text="Weapon Detections",
        secondary_y=False,
        title_font=dict(color="red")
    )
    
    fig_hourly.update_yaxes(
        title_text="Overcrowding Events",
        secondary_y=True,
        title_font=dict(color="blue")
    )
    
    fig_hourly.update_layout(
        title="Hourly Distribution of Events",
        barmode='overlay',
        height=chart_height,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )

    st.plotly_chart(fig_hourly, width='stretch')

with col2:
    # Top Weapon Hours - Left aligned
    if show_weapons and 'weapon' in hourly_df.columns:
        st.markdown('<h3 class="left-align">Top Weapon Hours</h3>', unsafe_allow_html=True)
        top_weapon_hours = hourly_df.nlargest(3, 'weapon')[['hour', 'weapon']]
        for idx, row in top_weapon_hours.iterrows():
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.markdown(f"<p style='font-weight: 600; color: #DC2626; margin: 0;'>{int(row['hour']):02d}:00</p>", unsafe_allow_html=True)
            with col_right:
                st.markdown(f"<p style='font-weight: 700; color: #DC2626; text-align: right; margin: 0;'>{int(row['weapon'])}</p>", unsafe_allow_html=True)
            st.markdown(f"<div style='height: 1px; background-color: #E5E7EB; margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
    
    # Top Crowd Hours - Left aligned
    if show_crowd and 'overcrowding' in hourly_df.columns:
        st.markdown('<h3 class="left-align">Top Crowd Hours</h3>', unsafe_allow_html=True)
        top_crowd_hours = hourly_df.nlargest(3, 'overcrowding')[['hour', 'overcrowding']]
        for idx, row in top_crowd_hours.iterrows():
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.markdown(f"<p style='font-weight: 600; color: #065F46; margin: 0;'>{int(row['hour']):02d}:00</p>", unsafe_allow_html=True)
            with col_right:
                st.markdown(f"<p style='font-weight: 700; color: #065F46; text-align: right; margin: 0;'>{int(row['overcrowding'])}</p>", unsafe_allow_html=True)
            st.markdown(f"<div style='height: 1px; background-color: #E5E7EB; margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
    
    # Hourly Summary - Aligned left/right
    st.markdown('<h3 class="left-align">Hourly Summary</h3>', unsafe_allow_html=True)
    
    if show_weapons:
        avg_weapons_hourly = hourly_df['weapon'].mean()
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("<p style='margin: 0; color: #374151;'>Avg Weapons/Hour</p>", unsafe_allow_html=True)
        with col_right:
            st.markdown(f"<p style='margin: 0; font-weight: 600; text-align: right;'>{avg_weapons_hourly:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<div style='height: 1px; background-color: #E5E7EB; margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
    
    if show_crowd:
        avg_crowd_hourly = hourly_df['overcrowding'].mean()
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("<p style='margin: 0; color: #374151;'>Avg Crowd Events/Hour</p>", unsafe_allow_html=True)
        with col_right:
            st.markdown(f"<p style='margin: 0; font-weight: 600; text-align: right;'>{avg_crowd_hourly:.2f}</p>", unsafe_allow_html=True)
        st.markdown(f"<div style='height: 1px; background-color: #E5E7EB; margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

# Row 4: Weekly & Monthly Analysis
st.markdown('<h2 class="sub-header">Weekly & Monthly Patterns</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Weekly aggregation
    weekly_stats = daily_df.groupby('day_of_week').agg({
        'weapon': 'mean',
        'overcrowding': 'mean',
        'total_detections': 'mean'
    }).reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    fig_weekly = go.Figure()
    
    if show_weapons:
        fig_weekly.add_trace(go.Bar(
            x=weekly_stats.index,
            y=weekly_stats['weapon'],
            name='Avg Weapons',
            marker_color='red',
            opacity=0.7,
            hovertemplate='Day: %{x}<br>Avg Weapons: %{y:.2f}<extra></extra>'
        ))
    
    if show_crowd:
        fig_weekly.add_trace(go.Bar(
            x=weekly_stats.index,
            y=weekly_stats['overcrowding'],
            name='Avg Crowd Events',
            marker_color='blue',
            opacity=0.7,
            hovertemplate='Day: %{x}<br>Avg Crowd Events: %{y:.2f}<extra></extra>'
        ))
    
    fig_weekly.update_layout(
        title="Average Events by Day of Week",
        xaxis_title="Day",
        yaxis_title="Average Count",
        barmode='group',
        template="plotly_white",
        height=chart_height - 100
    )
    
    st.plotly_chart(fig_weekly, use_container_width=True)

with col2:
    # Monthly aggregation (if we have enough data)
    if len(daily_df) > 30:
        monthly_stats = daily_df.groupby('month').agg({
            'weapon': 'sum',
            'overcrowding': 'sum',
            'total_detections': 'sum'
        }).reset_index()
        
        fig_monthly = go.Figure()
        
        if show_weapons:
            fig_monthly.add_trace(go.Bar(
                x=monthly_stats['month'],
                y=monthly_stats['weapon'],
                name='Total Weapons',
                marker_color='red',
                opacity=0.7,
                hovertemplate='Month: %{x}<br>Total Weapons: %{y}<extra></extra>'
            ))
        
        if show_crowd:
            fig_monthly.add_trace(go.Bar(
                x=monthly_stats['month'],
                y=monthly_stats['overcrowding'],
                name='Total Crowd Events',
                marker_color='blue',
                opacity=0.7,
                hovertemplate='Month: %{x}<br>Total Crowd Events: %{y}<extra></extra>'
            ))
        
        fig_monthly.update_layout(
            title="Monthly Totals",
            xaxis_title="Month",
            yaxis_title="Total Count",
            barmode='group',
            template="plotly_white",
            height=chart_height - 100
        )
        
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        # Show distribution pie chart
        st.markdown("### Event Distribution")
        
        # Calculate percentages
        total_weapons = data['summary']['total_weapons']
        total_crowd = data['summary']['total_overcrowding']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Weapon Detections', 'Overcrowding Events'],
            values=[total_weapons, total_crowd],
            hole=.3,
            marker_colors=['red', 'blue'],
            textinfo='label+percent',
            hovertemplate='%{label}: %{value} events<extra></extra>'
        )])
        
        fig_pie.update_layout(
            height=chart_height - 100,
            showlegend=False
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

# Row 5: Recent Events & Data Export
st.markdown('<h2 class="sub-header">Recent Events & Data</h2>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Recent Events", "Export Data"])

with tab1:
    # Display recent events if available
    if 'recent_events' in data and data['recent_events']:
        recent_events_df = pd.DataFrame(data['recent_events'])
        
        # Format the DataFrame for display
        display_df = recent_events_df.copy()
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp'])
            display_df['Date'] = display_df['timestamp'].dt.strftime('%Y-%m-%d')
            display_df['Time'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
        
        # Select and order columns
        columns_to_show = []
        if 'Date' in display_df.columns:
            columns_to_show.append('Date')
        if 'Time' in display_df.columns:
            columns_to_show.append('Time')
        if 'type__name' in display_df.columns:
            columns_to_show.append('type__name')
        if 'confidence_value' in display_df.columns:
            columns_to_show.append('confidence_value')
        if 'status' in display_df.columns:
            columns_to_show.append('status')
        
        if columns_to_show:
            display_df = display_df[columns_to_show]
            display_df.columns = [col.replace('type__name', 'Event Type')
                                 .replace('confidence_value', 'Confidence/Count')
                                 .replace('status', 'Status') for col in display_df.columns]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=300,
                column_config={
                    "Event Type": st.column_config.TextColumn(
                        "Event Type",
                        help="Type of security event"
                    ),
                    "Confidence/Count": st.column_config.NumberColumn(
                        "Confidence/Count",
                        format="%.2f",
                        help="Confidence score for weapons or count for overcrowding"
                    )
                }
            )
    else:
        st.info("No recent events data available. Events will appear here as they are detected.")

with tab2:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Raw Data Preview")
        st.dataframe(
            daily_df[['date', 'weapon', 'overcrowding', 'total_detections', 'day_of_week']],
            use_container_width=True,
            height=250
        )
    
    with col2:
        st.markdown("### Export Options")
        
        # Convert DataFrames to CSV
        daily_csv = daily_df.to_csv(index=False)
        hourly_csv = hourly_df.to_csv(index=False)
        
        st.download_button(
            label="Download Daily Data (CSV)",
            data=daily_csv,
            file_name=f"surveillance_daily_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.download_button(
            label="Download Hourly Data (CSV)",
            data=hourly_csv,
            file_name=f"surveillance_hourly_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Export as JSON
        json_data = json.dumps(data, indent=2)
        st.download_button(
            label="Download Full Data (JSON)",
            data=json_data,
            file_name=f"surveillance_full_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
            use_container_width=True
        )

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

with col2:
    st.markdown("**AI Surveillance Analytics Dashboard** - Real-time weapon and overcrowding detection monitoring")

with col3:
    if st.button("Refresh Now", use_container_width=True):
        st.rerun()

# Add some spacing
st.markdown("<br><br>", unsafe_allow_html=True)