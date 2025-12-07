import streamlit as st
import requests
import pandas as pd
import time
import os
from datetime import datetime

# --- Configuration (UPDATE) ---
# Local Django server URL for dashboard API calls
LOCAL_URL = "http://127.0.0.1:8000"
# Endpoint for video feed (should be at the app level, e.g., /video_feed/)
VIDEO_FEED_URL = f"{LOCAL_URL}/video_feed/"
# Endpoint for event logs (e.g., /api/logs/) - Updated to match your Django view
LOGS_URL = f"{LOCAL_URL}/api/logs/"
# NEW: Endpoint for the latest status (e.g., /api/latest_status/) - Updated to match your Django view
STATUS_API_URL = f"{LOCAL_URL}/api/latest_status/"

st.set_page_config(
    page_title="AI Surveillance System Dashboard (Weapon & Overcrowding)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Functions to Fetch Data ---

# Function to fetch latest system status
def fetch_system_status():
    """Fetches the latest alert status from the Django backend API."""
    try:
        response = requests.get(STATUS_API_URL, timeout=1) # Use a short timeout
        if response.status_code == 200:
            return response.json()
        else:
            return {'status_level': 'ERROR', 'message': f'Django Status API returned {response.status_code}'}
    except requests.exceptions.ConnectionError:
        return {'status_level': 'ERROR', 'message': 'Cannot connect to Django API. Server may be down.'}
    except requests.exceptions.Timeout:
        return {'status_level': 'ERROR', 'message': 'Django Status API connection timed out.'}
    except Exception as e:
        return {'status_level': 'ERROR', 'message': f'An unknown error occurred: {e}'}

# Function to fetch event logs (Updated for new schema)
def fetch_event_logs():
    """Fetches the latest events (weapon, overcrowding, etc.) from the Django backend."""
    try:
        response = requests.get(LOGS_URL)
        if response.status_code == 200:
            data = response.json()

            # If the response is an empty list, return an empty DataFrame immediately
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)

            # Format datetime
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Construct display_url locally from snapshot_path for dashboard display
            if 'snapshot_path' in df.columns:
                 df['display_url'] = df['snapshot_path'].apply(
                     lambda p: f"{LOCAL_URL}/snapshots/{p}" if pd.notna(p) and p else None
                 )
            else:
                 # If snapshot_path is not available, set display_url to None
                 df['display_url'] = None

            return df.sort_values(by='timestamp', ascending=False) # Ensure newest is first

        else:
            # Print status code for debugging if the API is returning an error
            st.error(f"Failed to fetch logs. Django Log API returned status code: {response.status_code}")
            return pd.DataFrame()
    except requests.exceptions.ConnectionError:
        # st.error("Cannot connect to Django API for logs.") # Uncomment for deeper debugging
        return pd.DataFrame()
    except Exception as e:
        # Catch unexpected errors during pandas processing
        st.error(f"Error processing log data in Streamlit: {e}")
        return pd.DataFrame()


# --- Dashboard Layout ---

st.title("🛡️ Real-Time AI Surveillance Dashboard (Weapon & Overcrowding)")

# 1. Status Banner Placeholder (Top priority)
status_placeholder = st.empty()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

# --- Column 1: Live Video Feed ---
with col1:
    st.header("Live Feed (YOLO Detection)")

    # MJPEG stream from Django
    st.markdown(
        f'<img src="{VIDEO_FEED_URL}" width="100%" style="border-radius: 10px;">',
        unsafe_allow_html=True
    )

# --- Column 2: Event Logs ---
with col2:
    st.header("Recent Event Logs")

    # Event Log Table Placeholder
    log_container = st.empty()


# --- Main Polling Loop for Dynamic Updates ---
if st.button("Start/Restart System Status Monitoring"):
    st.session_state['monitoring_active'] = True

if 'monitoring_active' not in st.session_state:
    st.session_state['monitoring_active'] = False

if st.session_state['monitoring_active']:
    while True:
        # --- A. Update System Status Banner (Polling every 2 seconds) ---
        status_data = fetch_system_status()

        with status_placeholder.container():

            if status_data.get('status_level') == 'ALERT':
                st.markdown(f"""
<div style='background-color: #A30000; color: white; padding: 25px; border-radius: 10px; font-size: 24px; font-weight: bold; text-align: center;'>
    🚨 💥 {status_data['message']} 💥 🚨
</div>
""", unsafe_allow_html=True)

            elif status_data.get('status_level') in ['OK', 'IDLE']:
                st.markdown(f"""
<div style='background-color: #2D4059; color: white; padding: 25px; border-radius: 10px; font-size: 20px; font-weight: bold; text-align: center;'>
    ✅ {status_data['message']}
</div>
""", unsafe_allow_html=True)

            else: # Error case
                st.error(f"⚠️ {status_data['message']}")

        # --- B. Update Event Logs (Polling every 5 seconds) ---
        logs_df = fetch_event_logs()

        with log_container.container():
            if not logs_df.empty:
                # Check if 'display_url' column exists before using it
                if 'display_url' in logs_df.columns:
                    snapshot_url_col = 'display_url'
                else:
                    snapshot_url_col = 'snapshot_path' # Fallback if display_url wasn't created

                st.dataframe(
                    logs_df,
                    column_config={
                        snapshot_url_col: st.column_config.LinkColumn(
                            "Snapshot Link",
                            display_text="View Snapshot",
                            help="Click to open the snapshot image" # Help text can be useful
                        ),
                        "timestamp": "Timestamp",
                        "confidence": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
                        "label": "Label", # This now comes from event.type.name
                        "snapshot_path": None, # Hide the raw path if we have display_url
                        "id": None, # Optionally hide the internal log ID
                        # Add other columns if needed, e.g., 'area_name' if linked in backend
                    },
                    column_order=['timestamp', 'label', 'confidence', snapshot_url_col], # Reorder as preferred
                    height=600,
                    hide_index=True
                )
            else:
                st.warning("No events logged yet, or API is unavailable.")

        # Sleep for the shorter of the two update intervals
        time.sleep(2) # Polls status every 2 seconds, logs effectively every 5 due to loop and sleep