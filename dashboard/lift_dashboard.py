# lift_dashboard.py
import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time
from PIL import Image
import io
import pytz

# Set your timezone here
YOUR_TIMEZONE = 'Asia/Kathmandu'

# Page config
st.set_page_config(
    page_title="Lift Capacity Monitor",
    layout="wide",
    page_icon="🏢",
    initial_sidebar_state="expanded"
)

# Custom CSS - Clean Theme
st.markdown("""
<style>
    .stApp {
        background: #0a0e27;
        color: #ffffff;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #0066ff, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(0, 102, 255, 0.3);
    }

    .sub-header {
        font-size: 1.8rem;
        color: #00ccff;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid rgba(0, 102, 255, 0.3);
        padding-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(0, 102, 255, 0.2);
    }

    .status-ok {
        border-left: 5px solid #10B981;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(16, 185, 129, 0.2));
    }

    .status-warning {
        border-left: 5px solid #eab308;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(234, 179, 8, 0.2));
    }

    .status-danger {
        border-left: 5px solid #ff3333;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(163, 0, 0, 0.2));
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 0 10px rgba(0, 102, 255, 0.3);
    }

    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-box {
        background: rgba(0, 102, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #0066ff;
        margin: 1rem 0;
        color: #cbd5e1;
    }

    .status-box {
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.2em;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .status-box-ok {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 2px solid #10B981;
        color: #6ee7b7;
    }

    .status-box-warning {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.2) 0%, rgba(202, 138, 4, 0.3) 100%);
        border: 2px solid #eab308;
        color: #fde047;
    }

    .status-box-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.3) 100%);
        border: 2px solid #ef4444;
        color: #fca5a5;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0066ff, #00ccff);
        color: white !important;
        border: none !important;
        border-radius: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0.5rem 2rem;
    }

    .upload-box {
        border: 3px dashed #0066ff;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background: rgba(15, 23, 42, 0.6);
        margin: 20px 0;
    }

    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 2px solid rgba(0, 102, 255, 0.3);
        background: rgba(15, 23, 42, 0.8);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = "http://127.0.0.1:8000"
LOCAL_URL = "http://127.0.0.1:8000"

# Initialize session state
if 'lifts' not in st.session_state:
    st.session_state.lifts = {}
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Title with Logo
st.markdown(f"""
<div style="margin-bottom: 2rem;">
    <div style="display: flex; align-items: center; gap: 15px;">
        <img src="{LOCAL_URL}/static/images/img_5.png" 
             style="width: 130px; height: 130px; border-radius: 10px; 
                    border: 1px solid rgba(0, 102, 255, 0.2);">
        <h1 class="main-header">
            Lift Capacity Monitoring System
        </h1>
    </div>
</div>
""", unsafe_allow_html=True)


def create_metric_card(label, value, delta=None, status_class=""):
    """
        Render a styled metric card in the Streamlit UI

        Query Parameters:
        - label: The descriptive text for the metric
        - value: The primary numerical or text value to display
        - delta: Optional secondary information or trend text (default: None)
        - status_class: CSS class for conditional styling (e.g., status-danger)
        """
    delta_html = f'<div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="metric-card {status_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.markdown("Configuration Panel")
    st.markdown("---")

    # Fetch lifts from API
    try:
        response = requests.get(f"{API_URL}/api/lift/list/", timeout=3)
        if response.status_code == 200:
            lifts_data = response.json().get('lifts', [])
            if lifts_data:
                st.session_state.lifts = {
                    f"{l['name']} (Max: {l['max_capacity']})": l['lift_id']
                    for l in lifts_data
                }
            else:
                st.session_state.lifts = {"Main Lift (Max: 8)": 1}
        else:
            st.session_state.lifts = {"Main Lift (Max: 8)": 1}
    except:
        st.session_state.lifts = {"Main Lift (Max: 8)": 1}

    # Lift selection
    if st.session_state.lifts:
        selected_lift_name = st.selectbox(
            "Select Lift:",
            list(st.session_state.lifts.keys())
        )
        lift_id = st.session_state.lifts[selected_lift_name]
    else:
        selected_lift_name = "Main Lift (Max: 8)"
        lift_id = 1

    st.markdown("---")

    # Quick stats
    st.markdown("Today's Statistics")
    try:
        response = requests.get(
            f"{API_URL}/api/lift/usage-stats/?lift_id={lift_id}&days=1",
            timeout=3
        )
        if response.status_code == 200:
            stats_data = response.json()
            if stats_data.get('stats'):
                today = stats_data['stats'][0].get('today')
                if today:
                    st.metric("Total Uses", today['usage_count'])
                    st.metric("Total People", today['total_people'])
                    st.metric("Overcrowding Events", today['overcrowding_count'])
                    st.metric("Peak Occupancy", today['max_people'])
    except:
        st.info("Stats loading...")

    st.markdown("---")

    # Instructions
    st.markdown("""
    <div class="info-box">
        <strong>System Instructions</strong><br><br>
        1. Select target lift<br>
        2. Upload clear image<br>
        3. Process detection<br>
        4. Review capacity status
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Refresh Dashboard", width='stretch'):
        st.rerun()

# Main content tabs
tab1, tab2, tab3 = st.tabs(["DETECTION SYSTEM", "ANALYTICS DASHBOARD", "ACTIVITY LOG"])

with tab1:
    st.markdown('<h2 class="sub-header">Real-Time Detection System</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("Upload Lift Image")

        uploaded_file = st.file_uploader(
            "Select image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload clear lift interior image",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # File info
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.markdown(f"**File:** {uploaded_file.name}")
            with col_info2:
                st.markdown(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            with col_info3:
                st.markdown(f"**Type:** {uploaded_file.type}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Process button
            col_btn = st.columns([1, 2, 1])
            with col_btn[1]:
                if st.button("PROCESS IMAGE", type="primary", width='stretch'):
                    st.session_state.processing = True

            if st.session_state.processing:
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("Uploading image...")
                    progress_bar.progress(20)

                    uploaded_file.seek(0)
                    file_bytes = uploaded_file.read()

                    if len(file_bytes) == 0:
                        st.error("File is empty! Please select a valid image.")
                        st.session_state.processing = False
                    else:
                        status_text.text("Running AI detection...")
                        progress_bar.progress(50)

                        files = {
                            'file': (uploaded_file.name, file_bytes, uploaded_file.type or 'image/jpeg')
                        }
                        data = {'lift_id': lift_id}

                        response = requests.post(
                            f"{API_URL}/api/lift/process-image/",
                            files=files,
                            data=data,
                            timeout=30
                        )

                        status_text.text("Analyzing results...")
                        progress_bar.progress(80)

                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.last_result = result
                            progress_bar.progress(100)
                            status_text.text("Complete!")
                            time.sleep(0.5)
                            st.session_state.processing = False
                            st.rerun()
                        else:
                            st.error(f"API Error {response.status_code}")
                            st.session_state.processing = False

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Ensure server is running at http://127.0.0.1:8000")
                    st.session_state.processing = False
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.processing = False
        else:
            st.markdown("""
            <div class="upload-box">
                <div style="font-size: 3em; margin-bottom: 20px; color: #0066ff;">▲</div>
                <h3 style="color: #cbd5e1;">Upload Lift Image</h3>
                <p style="color: #94a3b8; margin-top: 10px;">
                    Drag and drop or click to browse
                </p>
                <p style="color: #64748b; font-size: 0.9em; margin-top: 15px;">
                    Supported: JPG, JPEG, PNG
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("Detection Results")

        if st.session_state.last_result:
            result = st.session_state.last_result
            people_count = result['results']['people_count']
            max_capacity = result['lift']['max_capacity']
            is_overcrowded = result['results']['is_overcrowded']
            confidence = result['results']['confidence']
            warning_threshold = result['lift'].get('warning_threshold', max_capacity - 2)

            # Status banner
            if is_overcrowded:
                st.markdown("""
                <div class="status-box status-box-danger">
                    OVERLOADED<br>
                    <span style="font-size: 0.7em;">Immediate Action Required</span>
                </div>
                """, unsafe_allow_html=True)
            elif people_count >= warning_threshold:
                st.markdown("""
                <div class="status-box status-box-warning">
                    WARNING<br>
                    <span style="font-size: 0.7em;">Approaching Capacity</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="status-box status-box-ok">
                    NORMAL<br>
                    <span style="font-size: 0.7em;">Operating Normally</span>
                </div>
                """, unsafe_allow_html=True)

            # Metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                create_metric_card(
                    "People Count",
                    people_count,
                    f"of {max_capacity} maximum",
                    "status-danger" if is_overcrowded else "status-ok"
                )
            with col_m2:
                create_metric_card(
                    "Confidence",
                    f"{confidence * 100:.1f}%",
                    "Detection accuracy"
                )

            # Occupancy gauge
            st.markdown("Capacity Analysis")
            occupancy_rate = min(people_count / max_capacity, 1.0) * 100

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=occupancy_rate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Occupancy %", 'font': {'size': 20, 'color': '#cbd5e1'}},
                delta={'reference': warning_threshold / max_capacity * 100, 'increasing': {'color': "#ef4444"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                    'bar': {'color': "#0066ff"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(0, 102, 255, 0.3)",
                    'steps': [
                        {'range': [0, 60], 'color': 'rgba(16, 185, 129, 0.3)'},
                        {'range': [60, 80], 'color': 'rgba(234, 179, 8, 0.3)'},
                        {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "#ef4444", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#cbd5e1", 'family': "Arial"},
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

            # Progress bar
            st.progress(
                min(people_count / max_capacity, 1.0),
                text=f"**{people_count}/{max_capacity} people** ({occupancy_rate:.0f}%)"
            )

            # Performance metrics
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Processing Time", f"{result['results']['processing_time']:.2f}s")
            with col_p2:
                st.metric("Detection ID", f"#{result['detection_id']}")
        else:
            st.markdown("""
            <div class="info-box">
                <strong>Awaiting Analysis</strong><br><br>
                Upload an image and click "PROCESS IMAGE" to begin detection.<br><br>
                <strong>Results will include:</strong><br>
                • People count<br>
                • Occupancy level<br>
                • Detection confidence<br>
                • Annotated image
            </div>
            """, unsafe_allow_html=True)

    # Show processed image ONLY (not uploaded)
    if st.session_state.last_result and st.session_state.last_result.get('images', {}).get('processed'):
        st.markdown('<h2 class="sub-header">Processed Detection Image</h2>', unsafe_allow_html=True)
        processed_url = f"{API_URL}{st.session_state.last_result['images']['processed']}"
        try:
            response_img = requests.get(processed_url, timeout=5)
            if response_img.status_code == 200:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                img = Image.open(io.BytesIO(response_img.content))
                st.image(img, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.info("Processed image not available")

with tab2:
    st.markdown('<h2 class="sub-header">Usage Analytics Dashboard</h2>', unsafe_allow_html=True)

    # Date range selector
    col_date = st.columns([3, 1])
    with col_date[0]:
        days = st.selectbox("Analysis Period", [1, 7, 14, 30], index=1, format_func=lambda x: f"Last {x} days")

    try:
        response = requests.get(
            f"{API_URL}/api/lift/usage-stats/?lift_id={lift_id}&days={days}",
            timeout=5
        )

        if response.status_code == 200:
            stats_data = response.json()

            if stats_data.get('stats'):
                lift_stats = stats_data['stats'][0]
                period_stats = lift_stats.get('period_stats', {})

                # Summary KPIs
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    create_metric_card(
                        "Total Uses",
                        period_stats.get('total_uses', 0),
                        f"in {days} days"
                    )

                with col2:
                    create_metric_card(
                        "Total People",
                        period_stats.get('total_people', 0),
                        f"avg {period_stats.get('avg_people_per_use', 0):.1f}/use"
                    )

                with col3:
                    create_metric_card(
                        "Overcrowding Events",
                        period_stats.get('total_overcrowding', 0),
                        f"{period_stats.get('overcrowding_rate', 0):.1f}% rate",
                        "status-danger" if period_stats.get('total_overcrowding', 0) > 0 else "status-ok"
                    )

                with col4:
                    create_metric_card(
                        "Avg Daily Uses",
                        f"{period_stats.get('avg_uses_per_day', 0):.1f}",
                        "per day"
                    )

                # Today's stats
                st.markdown('<h3 class="sub-header" style="font-size: 1.4rem;">Today\'s Overview</h3>',
                            unsafe_allow_html=True)

                today_stats = lift_stats.get('today')
                if today_stats:
                    col_today1, col_today2, col_today3, col_today4 = st.columns(4)
                    with col_today1:
                        st.metric("Uses Today", today_stats.get('usage_count', 0))
                    with col_today2:
                        st.metric("People Today", today_stats.get('total_people', 0))
                    with col_today3:
                        st.metric("Overcrowding", today_stats.get('overcrowding_count', 0))
                    with col_today4:
                        st.metric("Peak Today", today_stats.get('max_people', 0))

                # Charts
                st.markdown('<h3 class="sub-header" style="font-size: 1.4rem;">Detection Trends</h3>',
                            unsafe_allow_html=True)

                recent = lift_stats.get('recent_detections', [])

                if recent and len(recent) > 1:
                    # Prepare data for charts with PROPER TIME FORMATTING
                    df_recent = pd.DataFrame(recent)

                    # Convert timestamp to datetime with proper timezone handling
                    df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'], utc=True)

                    # Convert to local timezone (Asia/Kathmandu)
                    local_tz = pytz.timezone(YOUR_TIMEZONE)
                    df_recent['local_time'] = df_recent['timestamp'].dt.tz_convert(local_tz).dt.tz_localize(None)

                    # Extract hour for grouping
                    df_recent['hour'] = df_recent['local_time'].dt.hour

                    # Format for display
                    df_recent['display_time'] = df_recent['local_time'].dt.strftime('%H:%M')
                    df_recent['display_date'] = df_recent['local_time'].dt.strftime('%Y-%m-%d %H:%M')

                    col_chart1, col_chart2 = st.columns(2)

                    with col_chart1:
                        # People count timeline
                        fig_timeline = go.Figure()

                        fig_timeline.add_trace(go.Scatter(
                            x=df_recent['local_time'],
                            y=df_recent['people_count'],
                            mode='lines+markers',
                            name='People Count',
                            line=dict(color='#0066ff', width=3),
                            marker=dict(size=8, color='#00ccff'),
                            hovertemplate='<b>Time</b>: %{x|%H:%M}<br><b>Date</b>: %{x|%Y-%m-%d}<br><b>Count</b>: %{y}<extra></extra>'
                        ))

                        # Add capacity line
                        max_capacity_val = today_stats.get('max_people', 8) if today_stats else 8
                        fig_timeline.add_hline(
                            y=max_capacity_val,
                            line_dash="dash",
                            line_color="#ef4444",
                            annotation_text=f"Max Capacity: {max_capacity_val}"
                        )

                        fig_timeline.update_layout(
                            title="People Count Timeline",
                            xaxis_title="Time",
                            yaxis_title="People Count",
                            template="plotly_dark",
                            height=300,
                            plot_bgcolor='rgba(15, 23, 42, 0.6)',
                            paper_bgcolor='rgba(15, 23, 42, 0.8)',
                            font=dict(color='#cbd5e1'),
                            title_font=dict(color='#00ccff'),
                            xaxis=dict(
                                tickformat='%H:%M\n%b %d',
                                tickangle=45
                            )
                        )

                        st.plotly_chart(fig_timeline, use_container_width=True)

                    with col_chart2:
                        # Average People by Hour
                        if len(df_recent['hour'].unique()) > 1:
                            # Group by hour and calculate statistics
                            hourly_stats = df_recent.groupby('hour').agg({
                                'people_count': ['mean', 'count', 'max', 'min']
                            }).round(1)
                            hourly_stats.columns = ['avg_people', 'count', 'max_people', 'min_people']
                            hourly_stats = hourly_stats.reset_index()

                            # Create hour labels (e.g., "08:00", "14:00")
                            hourly_stats['hour_label'] = hourly_stats['hour'].apply(lambda x: f"{x:02d}:00")

                            # Create the bar chart
                            fig_hourly = go.Figure()

                            fig_hourly.add_trace(go.Bar(
                                x=hourly_stats['hour_label'],
                                y=hourly_stats['avg_people'],
                                marker_color='#0066ff',
                                opacity=0.8,
                                hovertemplate='<b>Hour</b>: %{x}<br><b>Avg People</b>: %{y:.1f}<br><b>Samples</b>: ' + \
                                              hourly_stats['count'].astype(str) + '<extra></extra>',
                                name='Average People'
                            ))

                            fig_hourly.update_layout(
                                title="Average People by Hour",
                                xaxis_title="Hour of Day (24h)",
                                yaxis_title="Average People",
                                template="plotly_dark",
                                height=300,
                                plot_bgcolor='rgba(15, 23, 42, 0.6)',
                                paper_bgcolor='rgba(15, 23, 42, 0.8)',
                                font=dict(color='#cbd5e1'),
                                title_font=dict(color='#00ccff')
                            )

                            st.plotly_chart(fig_hourly, use_container_width=True)

                            # Show hourly statistics
                            st.markdown("Hourly Statistics:**")
                            col_h1, col_h2, col_h3 = st.columns(3)
                            with col_h1:
                                peak_hour = hourly_stats.loc[hourly_stats['avg_people'].idxmax()]
                                st.metric("Peak Hour", f"{peak_hour['hour_label']}")
                            with col_h2:
                                st.metric("Peak Avg", f"{peak_hour['avg_people']:.1f}")
                            with col_h3:
                                st.metric("Data Points", f"{len(df_recent)}")
                        else:
                            st.info("Not enough hourly data for analysis")

                # Recent detections table
                st.markdown('<h3 class="sub-header" style="font-size: 1.4rem;">Recent Detections</h3>',
                            unsafe_allow_html=True)

                if recent:
                    local_tz = pytz.timezone(YOUR_TIMEZONE)
                    for detection in recent[:5]:
                        status_color = detection['status_color']
                        status_icon = "OVERLOADED" if detection['is_overcrowded'] else "NORMAL"
                        status_class = "status-danger" if detection['is_overcrowded'] else "status-ok"

                        # Parse and format timestamp correctly
                        try:
                            # Parse timestamp with timezone
                            det_time = pd.to_datetime(detection['timestamp'], utc=True)
                            # Convert to local time
                            local_time = det_time.tz_convert(local_tz).tz_localize(None)
                            formatted_time = local_time.strftime('%I:%M %p')
                            formatted_date = local_time.strftime('%b %d')
                        except:
                            formatted_time = "Unknown"
                            formatted_date = ""

                        st.markdown(f"""
                        <div class="metric-card {status_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <strong style="font-size: 1.2em; color: #00ccff;">{detection['people_count']} people</strong>
                                    <span style="color: #94a3b8; margin-left: 15px;">
                                        {formatted_time} {formatted_date}
                                    </span>
                                </div>
                                <div>
                                    <span style="color: {status_color}; font-weight: bold; font-size: 1.1em;">
                                        {status_icon}
                                    </span>
                                    <span style="color: #94a3b8; margin-left: 10px;">
                                        {detection['confidence']:.1f}% conf
                                    </span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No recent detections available")
            else:
                st.info("No analytics data available yet")
    except Exception as e:
        st.error(f"Unable to load analytics data: {str(e)}")

with tab3:
    st.markdown('<h2 class="sub-header">Activity Log</h2>', unsafe_allow_html=True)

    try:
        response = requests.get(
            f"{API_URL}/api/lift/usage-stats/?lift_id={lift_id}&days=7",
            timeout=5
        )

        if response.status_code == 200:
            stats_data = response.json()

            if stats_data.get('stats'):
                recent_detections = stats_data['stats'][0].get('recent_detections', [])

                if recent_detections:
                    # Create DataFrame with proper time formatting
                    local_tz = pytz.timezone(YOUR_TIMEZONE)
                    df_data = []
                    for det in recent_detections:
                        try:
                            # Parse timestamp with timezone handling
                            det_time = pd.to_datetime(det['timestamp'], utc=True)
                            # Convert to local time (Asia/Kathmandu)
                            local_time = det_time.tz_convert(local_tz).tz_localize(None)
                            date_str = local_time.strftime('%Y-%m-%d')
                            time_str = local_time.strftime('%I:%M %p')
                        except:
                            date_str = "Unknown"
                            time_str = "Unknown"

                        df_data.append({
                            'Date': date_str,
                            'Time': time_str,
                            'People': det['people_count'],
                            'Status': 'OVERLOADED' if det['is_overcrowded'] else 'NORMAL',
                            'Confidence': f"{det['confidence']:.1f}%",
                            'ID': f"#{det['detection_id']}"
                        })

                    df = pd.DataFrame(df_data)


                    # Style the DataFrame
                    def color_status(val):
                        if 'OVERLOADED' in val:
                            return 'color: #ff3333; font-weight: bold'
                        elif 'NORMAL' in val:
                            return 'color: #10B981; font-weight: bold'
                        return ''


                    styled_df = df.style.applymap(color_status, subset=['Status'])

                    st.dataframe(styled_df, use_container_width=True, height=400, hide_index=True)

                    # Add some statistics
                    st.markdown('<h3 class="sub-header" style="font-size: 1.4rem;">Activity Statistics</h3>',
                                unsafe_allow_html=True)

                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Total Detections", len(df))
                    with col_stat2:
                        overloaded_count = len(df[df['Status'] == 'OVERLOADED'])
                        st.metric("Overloaded Events", overloaded_count)
                    with col_stat3:
                        avg_people = df['People'].mean()
                        st.metric("Avg People", f"{avg_people:.1f}")
                else:
                    st.info("No activity recorded yet")
    except Exception as e:
        st.error(f"Unable to load activity log: {str(e)}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown(
        f"<div style='color: #94a3b8;'><strong>Last Updated:</strong> {datetime.now().strftime('%H:%M:%S')}</div>",
        unsafe_allow_html=True)

with col2:
    st.markdown(
        "<div style='color: #94a3b8; text-align: center;'><strong>Lift Capacity Monitoring System</strong> - AI-powered occupancy detection</div>",
        unsafe_allow_html=True)

with col3:
    if st.button("Refresh Data", use_container_width=True):
        st.rerun()