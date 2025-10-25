import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta, date
from skyfield.api import load, EarthSatellite, wgs84, utc
import statistics

# Constants
ts = load.timescale()  # Initialize timescale globally
MU = 398600.4418  # km^3/s^2
R_EARTH = 6371.0  # km

NAVIK_SATS = {
    "IRNSS-1B": 39635,
    "IRNSS-1C": 40269,
    "IRNSS-1D": 40547,
    "IRNSS-1E": 41241,
    "IRNSS-1F": 41384,
    "IRNSS-1I": 43286,
    "NVS-01": 56759
}

INDIA_EXTREME_POINTS = {
    "Northernmost (Siachen Glacier)": (35.5, 77.0),
    "Southernmost (Indira Point)": (6.75, 93.85),
    "Easternmost (Kibithu)": (28.0, 97.0),
    "Westernmost (Guhar Moti)": (23.7, 68.1),
    "Capital (Delhi)": (28.7, 77.1)
}

NAVIK_SERVICE_REQUIREMENTS = {
    "IRNSS-1B": {"longitude": 55.0, "inclination": 29.0},
    "IRNSS-1C": {"longitude": 83.0, "inclination": 5.0},
    "IRNSS-1D": {"longitude": 111.75, "inclination": 30.0},
    "IRNSS-1E": {"longitude": 111.75, "inclination": 29.0},
    "IRNSS-1F": {"longitude": 32.5, "inclination": 5.0},
    "IRNSS-1I": {"longitude": 55.0, "inclination": 29.0},
    "NVS-01": {"longitude": 129.5, "inclination": 5.0}
}

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"

# Streamlit Configuration
st.set_page_config(page_title="NavIC Comprehensive Monitoring", layout="wide")
st.title("🛰️ NavIC (IRNSS/NVS) - Comprehensive Monitoring System")

# ==================== SPACE-TRACK FUNCTIONS ====================

@st.cache_resource
def get_spacetrack_session(username: str, password: str):
    """Returns a logged-in requests.Session cached as a resource."""
    s = requests.Session()
    resp = s.post(LOGIN_URL, data={'identity': username, 'password': password})
    if resp.status_code != 200:
        raise Exception(f"Space-Track login failed: HTTP {resp.status_code}")
    return s

@st.cache_data(ttl=3600)
def fetch_tle_json_cached(norad_id: int, start_date: str, end_date: str, username: str, password: str):
    """Cached fetch of the GP history JSON."""
    session = get_spacetrack_session(username, password)
    gp_url = (
        f"https://www.space-track.org/basicspacedata/query/class/gp_history/"
        f"EPOCH/{start_date}--{end_date}/NORAD_CAT_ID/{norad_id}/orderby/EPOCH asc/format/json"
    )
    resp = session.get(gp_url)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch GP data for {norad_id}: HTTP {resp.status_code}")
    return resp.json()

@st.cache_data(ttl=3600)
def fetch_multiple_tles(norad_ids, username: str, password: str):
    """Fetch latest TLE data for multiple satellites."""
    session = get_spacetrack_session(username, password)
    ids_str = ','.join(map(str, norad_ids))
    query_url = (
        f"https://www.space-track.org/basicspacedata/query/class/tle_latest/"
        f"NORAD_CAT_ID/{ids_str}/orderby/NORAD_CAT_ID,ORDINAL/format/3le"
    )
    resp = session.get(query_url)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch TLE data: HTTP {resp.status_code}")
    return resp.text

# ==================== MANEUVER DETECTION FUNCTIONS ====================

def rolling_median_safe(s, window=3):
    """Compute rolling median with fallback for edge cases."""
    return s.astype(float).rolling(window=window, min_periods=1, center=True).median()

def mad_zscore(x, threshold=1e-9):
    """Robust z-score using Median Absolute Deviation (MAD)."""
    x = np.array(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    
    if mad < threshold or np.isnan(mad):
        mean = np.nanmean(x)
        std = np.nanstd(x)
        if std < threshold or np.isnan(std):
            return np.zeros_like(x)
        return (x - mean) / std
    
    return 0.6745 * (x - med) / mad

def detect_navik_maneuvers(df, sma_col='SEMIMAJOR_AXIS', inc_col='INCLINATION',
                           z_thresh=3.5, sma_abs_thresh_km=0.5, inc_abs_thresh_deg=0.01,
                           persist_window=2):
    """Detects orbital maneuvers for NavIK satellites."""
    df2 = df.copy().reset_index(drop=True)
    
    for c in [sma_col, inc_col]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors='coerce')
    
    if sma_col in df2.columns:
        df2[sma_col + '_smooth'] = rolling_median_safe(df2[sma_col], window=3)
    if inc_col in df2.columns:
        df2[inc_col + '_smooth'] = rolling_median_safe(df2[inc_col], window=3)
    
    df2['dSMA'] = df2[sma_col + '_smooth'].diff() if sma_col + '_smooth' in df2 else np.nan
    df2['dINC'] = df2[inc_col + '_smooth'].diff() if inc_col + '_smooth' in df2 else np.nan
    
    df2['z_dSMA'] = mad_zscore(df2['dSMA'].fillna(0))
    df2['z_dINC'] = mad_zscore(df2['dINC'].fillna(0))
    
    df2['SMA_candidate'] = False
    if 'dSMA' in df2.columns and 'z_dSMA' in df2.columns:
        df2.loc[
            (df2['dSMA'].abs() >= sma_abs_thresh_km) & (df2['z_dSMA'].abs() >= z_thresh),
            'SMA_candidate'
        ] = True
    
    df2['INC_candidate'] = False
    if 'dINC' in df2.columns and 'z_dINC' in df2.columns:
        df2.loc[
            (df2['dINC'].abs() >= inc_abs_thresh_deg) & (df2['z_dINC'].abs() >= z_thresh),
            'INC_candidate'
        ] = True
    
    half = persist_window
    
    df2['pre_sma_med'] = df2[sma_col + '_smooth'].rolling(
        window=2*half+1, min_periods=1, center=False
    ).apply(
        lambda arr: np.median(arr[:-half]) if len(arr) > half else np.nan, raw=True
    )
    df2['post_sma_med'] = df2[sma_col + '_smooth'].shift(-half).rolling(
        window=2*half+1, min_periods=1, center=False
    ).apply(
        lambda arr: np.median(arr[half:]) if len(arr) > half else np.nan, raw=True
    )
    df2['sma_med_delta'] = (df2['post_sma_med'] - df2['pre_sma_med']).abs()
    
    df2['pre_inc_med'] = df2[inc_col + '_smooth'].rolling(
        window=2*half+1, min_periods=1, center=False
    ).apply(
        lambda arr: np.median(arr[:-half]) if len(arr) > half else np.nan, raw=True
    )
    df2['post_inc_med'] = df2[inc_col + '_smooth'].shift(-half).rolling(
        window=2*half+1, min_periods=1, center=False
    ).apply(
        lambda arr: np.median(arr[half:]) if len(arr) > half else np.nan, raw=True
    )
    df2['inc_med_delta'] = (df2['post_inc_med'] - df2['pre_inc_med']).abs()
    
    df2['SMA_MANEUVER'] = False
    df2.loc[
        (df2['SMA_candidate']) & (df2['sma_med_delta'] >= sma_abs_thresh_km),
        'SMA_MANEUVER'
    ] = True
    
    df2['INC_MANEUVER'] = False
    df2.loc[
        (df2['INC_candidate']) & (df2['inc_med_delta'] >= inc_abs_thresh_deg),
        'INC_MANEUVER'
    ] = True
    
    df2['MANEUVER'] = df2['SMA_MANEUVER'] | df2['INC_MANEUVER']
    
    return df2

# ==================== HEALTH ASSESSMENT FUNCTIONS ====================

def calculate_maneuver_uniformity(maneuver_dates):
    """Calculate coefficient of variation for maneuver spacing."""
    if len(maneuver_dates) < 2:
        return None
    
    maneuver_dates = sorted(maneuver_dates)
    intervals = [(maneuver_dates[i+1] - maneuver_dates[i]).days 
                 for i in range(len(maneuver_dates)-1)]
    
    if not intervals or np.mean(intervals) == 0:
        return None
    
    return np.std(intervals) / np.mean(intervals)

def assess_satellite_health(sat_name, sat_df, maneuver_events, inc_tolerance, 
                           min_man_per_month, max_man_per_month, uniformity_threshold):
    """Comprehensive health assessment for a satellite."""
    requirements = NAVIK_SERVICE_REQUIREMENTS.get(sat_name, {})
    target_inclination = requirements.get("inclination", None)
    
    mean_inclination = sat_df['INCLINATION'].mean()
    std_inclination = sat_df['INCLINATION'].std()
    
    observation_days = (sat_df['EPOCH'].max() - sat_df['EPOCH'].min()).days
    observation_months = observation_days / 30.0
    
    num_maneuvers = len(maneuver_events)
    maneuvers_per_month = num_maneuvers / observation_months if observation_months > 0 else 0
    
    if target_inclination is not None:
        inc_deviation = abs(mean_inclination - target_inclination)
        inc_score = max(0, 100 - (inc_deviation / inc_tolerance) * 100)
    else:
        inc_score = None
        inc_deviation = None
    
    if maneuvers_per_month < min_man_per_month:
        maintenance_score = 30
    elif maneuvers_per_month > max_man_per_month:
        maintenance_score = 60
    else:
        maintenance_score = 100
    
    if num_maneuvers >= 2:
        maneuver_dates = pd.to_datetime(maneuver_events['EPOCH']).tolist()
        uniformity_cov = calculate_maneuver_uniformity(maneuver_dates)
        
        if uniformity_cov is not None and uniformity_cov <= uniformity_threshold:
            uniformity_score = 100
        elif uniformity_cov is not None:
            uniformity_score = max(0, 100 - ((uniformity_cov - uniformity_threshold) / uniformity_threshold) * 50)
        else:
            uniformity_score = 50
    else:
        uniformity_score = 50 if num_maneuvers == 1 else 0
        uniformity_cov = None
    
    if inc_score is not None:
        overall_score = (inc_score * 0.5 + maintenance_score * 0.3 + uniformity_score * 0.2)
    else:
        overall_score = (maintenance_score * 0.6 + uniformity_score * 0.4)
    
    if overall_score >= 85:
        health_status = "Excellent"
        status_color = "🟢"
    elif overall_score >= 70:
        health_status = "Good"
        status_color = "🟡"
    elif overall_score >= 50:
        health_status = "Fair"
        status_color = "🟠"
    else:
        health_status = "Needs Attention"
        status_color = "🔴"
    
    remarks = []
    
    if inc_score is not None and inc_deviation is not None:
        if inc_deviation <= inc_tolerance * 0.3:
            remarks.append(f"Excellent inclination control (±{inc_deviation:.2f}°)")
        elif inc_deviation <= inc_tolerance:
            remarks.append(f"Inclination within tolerance (±{inc_deviation:.2f}°)")
        else:
            remarks.append(f"⚠️ Inclination deviation exceeds tolerance ({inc_deviation:.2f}°)")
    
    if maneuvers_per_month < min_man_per_month:
        remarks.append(f"⚠️ Low maintenance activity ({maneuvers_per_month:.1f}/month)")
    elif maneuvers_per_month > max_man_per_month:
        remarks.append(f"⚠️ High correction frequency ({maneuvers_per_month:.1f}/month)")
    else:
        remarks.append(f"Active maintenance ({maneuvers_per_month:.1f} maneuvers/month)")
    
    if uniformity_cov is not None:
        if uniformity_cov <= uniformity_threshold:
            remarks.append("Regular maneuver pattern detected")
        else:
            remarks.append("Irregular maneuver spacing")
    
    if std_inclination < 0.1:
        remarks.append("Stable orbital parameters")
    
    return {
        'Satellite': sat_name,
        'Health Status': f"{status_color} {health_status}",
        'Overall Score': round(overall_score, 1),
        'Target Incl. (°)': target_inclination if target_inclination else "N/A",
        'Mean Incl. (°)': round(mean_inclination, 3),
        'Incl. Dev. (°)': round(inc_deviation, 3) if inc_deviation else "N/A",
        'Maneuvers/Month': round(maneuvers_per_month, 2),
        'Uniformity (CoV)': round(uniformity_cov, 3) if uniformity_cov else "N/A",
        'Remarks': " | ".join(remarks)
    }

# ==================== DOP CALCULATION FUNCTIONS ====================

def parse_tle_data(tle_text, sat_dict):
    """Parse TLE text and create satellite objects"""
    ts = load.timescale()
    satellites = {}
    
    lines = tle_text.strip().split('\n')
    
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
        
        name = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        
        try:
            norad_id = int(line1[2:7])
            sat_name = None
            for s_name, s_id in sat_dict.items():
                if s_id == norad_id:
                    sat_name = s_name
                    break
            
            if sat_name:
                satellite = EarthSatellite(line1, line2, sat_name, ts)
                satellites[sat_name] = satellite
        except (ValueError, IndexError):
            continue
    
    return satellites

def calculate_satellite_position(satellite, time, observer_location):
    """Calculate satellite position relative to observer"""
    try:
        difference = satellite - observer_location
        topocentric = difference.at(time)
        alt, az, distance = topocentric.altaz()
        
        return {
            'altitude': alt.degrees,
            'azimuth': az.degrees,
            'distance': distance.km,
            'elevation': alt.degrees
        }
    except Exception:
        return None

def calculate_design_matrix(satellite_positions, observer_lat, observer_lon):
    """Calculate the geometry matrix (design matrix) for DOP calculation"""
    H = []
    
    for pos in satellite_positions:
        if pos is None:
            continue
            
        if pos['elevation'] > 5:
            az_rad = np.radians(pos['azimuth'])
            el_rad = np.radians(pos['elevation'])
            
            dx = np.cos(el_rad) * np.sin(az_rad)
            dy = np.cos(el_rad) * np.cos(az_rad)
            dz = np.sin(el_rad)
            
            H.append([dx, dy, dz, 1])
    
    return np.array(H) if H else np.array([]).reshape(0, 4)

def calculate_dop_values(H):
    """Calculate various DOP values from the design matrix"""
    if len(H) < 4:
        return None
    
    try:
        HTH = np.dot(H.T, H)
        
        if np.linalg.det(HTH) == 0:
            return None
            
        Q = np.linalg.inv(HTH)
        
        dop = {
            'GDOP': float(np.sqrt(np.trace(Q))),
            'PDOP': float(np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])),
            'HDOP': float(np.sqrt(Q[0,0] + Q[1,1])),
            'VDOP': float(np.sqrt(Q[2,2])),
            'TDOP': float(np.sqrt(Q[3,3])),
        }
        
        return dop
    except np.linalg.LinAlgError:
        return None

def calculate_dop_for_location(satellites_dict, lat, lon, time):
    """Calculate DOP for a specific location and time"""
    ts = load.timescale()
    t = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)
    
    observer = wgs84.latlon(lat, lon)
    
    satellite_positions = []
    visible_sats = []
    
    for sat_name, sat_obj in satellites_dict.items():
        pos = calculate_satellite_position(sat_obj, t, observer)
        if pos:
            satellite_positions.append(pos)
            if pos['elevation'] > 5:
                visible_sats.append(sat_name)
    
    H = calculate_design_matrix(satellite_positions, lat, lon)
    dop = calculate_dop_values(H)
    
    return dop, visible_sats, satellite_positions

# ==================== BOUNDING BOX CALCULATION FUNCTIONS ====================

def get_geo_box_vectorized(satellite, epoch, timestep_minutes, prop_duration_days):
    """Calculate geographic bounding box for satellite propagation."""
    # Ensure epoch is timezone-aware (UTC)
    if epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=utc)
    
    t1 = epoch
    t2 = t1 + timedelta(days=prop_duration_days)
    delta_t = (t2 - t1).seconds + 24*3600*(t2 - t1).days
    n_steps = int(delta_t / (timestep_minutes * 60)) + 1
    
    # Create time array
    time_offsets = np.arange(1, n_steps-1) * (60 * timestep_minutes)
    time_offsets = time_offsets.tolist()
    epochs = [t1 + timedelta(seconds=t) for t in time_offsets]
    
    # Convert epochs to Skyfield time objects
    ts_times = [ts.from_datetime(t) for t in epochs]
    
    # Get positions
    positions = [satellite.at(t) for t in ts_times]
    
    # Get lat/lon
    lat_lon = [wgs84.latlon_of(pos) for pos in positions]
    latitudes = [ll[0].degrees for ll in lat_lon]
    longitudes = [ll[1].degrees for ll in lat_lon]
    
    return {
        'epochs': epochs,
        'latitudes': latitudes,
        'longitudes': longitudes,
        'min_lon': min(longitudes),
        'max_lon': max(longitudes),
        'mean_lon': statistics.mean(longitudes),
        'min_lat': min(latitudes),
        'max_lat': max(latitudes),
        'mean_lat': statistics.mean(latitudes)
    }

def calculate_bounding_boxes(satellites_dict, reference_time, timestep_minutes=15, prop_duration_days=1.5):
    """Calculate bounding boxes for all satellites."""
    bounding_boxes = {}
    
    for sat_name, sat_obj in satellites_dict.items():
        try:
            box_data = get_geo_box_vectorized(sat_obj, reference_time, timestep_minutes, prop_duration_days)
            bounding_boxes[sat_name] = box_data
        except Exception as e:
            st.warning(f"Could not calculate bounding box for {sat_name}: {str(e)}")
            continue
    
    return bounding_boxes

# ==================== DATA FETCHING FUNCTION ====================

def fetch_and_classify_satellite(norad_id: int, start_date: str, end_date: str,
                                 username: str, password: str, igso_min=10, deviation_tol=0.3):
    """Fetches and classifies satellite data."""
    data = fetch_tle_json_cached(int(norad_id), start_date, end_date, username, password)

    if not data:
        raise ValueError(f"No GP data found for NORAD ID {norad_id} in given range")

    df = pd.DataFrame(data)

    if 'EPOCH' not in df.columns and 'epoch' in df.columns:
        df.rename(columns={'epoch': 'EPOCH'}, inplace=True)
    if 'INCLINATION' not in df.columns and 'inclination' in df.columns:
        df.rename(columns={'inclination': 'INCLINATION'}, inplace=True)
    if 'SEMIMAJOR_AXIS' not in df.columns and 'semimajor_axis' in df.columns:
        df.rename(columns={'semimajor_axis': 'SEMIMAJOR_AXIS'}, inplace=True)

    if 'EPOCH' not in df.columns or 'INCLINATION' not in df.columns:
        raise ValueError("GP JSON missing required fields 'EPOCH' or 'INCLINATION'")

    required_cols = ['EPOCH', 'INCLINATION']
    if 'SEMIMAJOR_AXIS' in df.columns:
        required_cols.append('SEMIMAJOR_AXIS')
    
    df = df[required_cols].copy()
    df['EPOCH'] = pd.to_datetime(df['EPOCH'])
    df['INCLINATION'] = df['INCLINATION'].astype(float)

    df['type'] = df['INCLINATION'].apply(
        lambda x: 'GSO' if (x > 0.0 and x < 10.0) else ('IGSO' if x >= igso_min else 'Unclassified')
    )

    mean_incl = df['INCLINATION'].mean()
    df['mean_inclination'] = mean_incl
    df['maintained'] = df['INCLINATION'].apply(lambda x: abs(x - mean_incl) <= deviation_tol)

    df = df.sort_values('EPOCH').reset_index(drop=True)

    if 'SEMIMAJOR_AXIS' in df.columns:
        df['SEMIMAJOR_AXIS'] = df['SEMIMAJOR_AXIS'].astype(float)
        df['altitude_km'] = df['SEMIMAJOR_AXIS'] - R_EARTH
    else:
        df['SEMIMAJOR_AXIS'] = np.nan
        df['altitude_km'] = np.nan

    return df

# ==================== STREAMLIT UI ====================

# Sidebar Configuration
st.sidebar.header("🔐 Space-Track Credentials")
username = st.sidebar.text_input("Username", value="", type="default")
password = st.sidebar.text_input("Password", value="", type="password")

st.sidebar.header("📅 Date Range")
start_date = st.sidebar.date_input("Start date", value=date(2025, 1, 1))
end_date = st.sidebar.date_input("End date", value=date(2025, 10, 1))
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

daily_only = st.sidebar.checkbox("Keep only one TLE per day (first entry)", value=True)

st.sidebar.header("⚙️ Analysis Parameters")
with st.sidebar.expander("Maneuver Detection", expanded=False):
    z_threshold = st.number_input("Z-Score Threshold", min_value=1.0, max_value=10.0, 
                                  value=3.5, step=0.5)
    sma_threshold = st.number_input("SMA Change Threshold (km)", min_value=0.1, 
                                    max_value=5.0, value=0.5, step=0.1)
    inc_threshold = st.number_input("Inclination Change Threshold (degrees)", 
                                    min_value=0.001, max_value=0.1, value=0.01, 
                                    step=0.001, format="%.3f")
    persist_window = st.number_input("Persistence Window", min_value=1, max_value=10, 
                                     value=2, step=1)

with st.sidebar.expander("Health Assessment", expanded=False):
    inclination_tolerance = st.number_input("Inclination Tolerance (degrees)", 
                                           min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    min_maneuvers_per_month = st.number_input("Min Maneuvers/Month", min_value=0, 
                                              max_value=10, value=1, step=1)
    max_maneuvers_per_month = st.number_input("Max Maneuvers/Month", min_value=1, 
                                              max_value=20, value=8, step=1)
    maneuver_uniformity_threshold = st.number_input("Maneuver Uniformity Threshold (CoV)", 
                                                   min_value=0.1, max_value=2.0, 
                                                   value=0.8, step=0.1)

# Main Analysis Button
if st.button("🚀 Fetch NavIC Data & Run Analysis", type="primary"):
    if not username or not password:
        st.error("❌ Please enter Space-Track username and password in the sidebar.")
    else:
        with st.spinner("🔄 Fetching satellite data..."):
            all_dfs = []
            errors = {}
            
            for sat_name, norad in NAVIK_SATS.items():
                try:
                    df = fetch_and_classify_satellite(
                        norad_id=int(norad),
                        start_date=start_date_str,
                        end_date=end_date_str,
                        username=username,
                        password=password,
                        igso_min=10,
                        deviation_tol=0.3
                    )

                    df['EPOCH'] = pd.to_datetime(df['EPOCH'])
                    df = df.sort_values('EPOCH').reset_index(drop=True)

                    if daily_only:
                        df['date'] = df['EPOCH'].dt.date
                        df = df.sort_values('EPOCH').groupby('date', as_index=False).first()
                        df['EPOCH'] = pd.to_datetime(df['EPOCH'])

                    df['satellite'] = sat_name

                    if 'mean_inclination' not in df.columns:
                        df['mean_inclination'] = df['INCLINATION'].mean()

                    all_dfs.append(df)

                except Exception as e:
                    errors[sat_name] = str(e)

            if errors:
                st.warning("⚠️ Some satellites failed to fetch:")
                for s, msg in errors.items():
                    st.write(f"- **{s}**: {msg}")

            if not all_dfs:
                st.error("❌ No data fetched for any satellite.")
            else:
                df_all = pd.concat(all_dfs, ignore_index=True, sort=False)
                
                # Store in session state
                st.session_state['df_all'] = df_all
                st.session_state['analysis_complete'] = True
                st.session_state['errors'] = errors
                
                st.success("✅ Data fetched successfully! Scroll down to see results.")

# Display results if analysis is complete
if st.session_state.get('analysis_complete', False):
    df_all = st.session_state['df_all']
    
    # ==================== HEALTH ASSESSMENT ====================
    st.header("🏥 Satellite Health Assessment")
    
    maneuver_summary = []
    all_maneuvers_df = pd.DataFrame()
    health_assessments = []
    
    for sat_name in sorted(df_all['satellite'].unique()):
        sat_df = df_all[df_all['satellite'] == sat_name].copy()
        
        sat_detected = detect_navik_maneuvers(
            sat_df,
            sma_col='SEMIMAJOR_AXIS',
            inc_col='INCLINATION',
            z_thresh=z_threshold,
            sma_abs_thresh_km=sma_threshold,
            inc_abs_thresh_deg=inc_threshold,
            persist_window=int(persist_window)
        )
        
        sma_maneuvers = int(sat_detected['SMA_MANEUVER'].sum())
        inc_maneuvers = int(sat_detected['INC_MANEUVER'].sum())
        
        maneuver_events = sat_detected[sat_detected['MANEUVER']].copy()
        maneuver_events['satellite'] = sat_name
        all_maneuvers_df = pd.concat([all_maneuvers_df, maneuver_events], ignore_index=True)
        
        maneuver_summary.append({
            'Satellite': sat_name,
            'SMA Maneuvers': sma_maneuvers,
            'Inclination Maneuvers': inc_maneuvers,
            'Observation Period (days)': (sat_df['EPOCH'].max() - sat_df['EPOCH'].min()).days
        })
        
        health_data = assess_satellite_health(
            sat_name, sat_df, maneuver_events,
            inclination_tolerance, min_maneuvers_per_month,
            max_maneuvers_per_month, maneuver_uniformity_threshold
        )
        health_assessments.append(health_data)
    
    health_df = pd.DataFrame(health_assessments)
    
    st.dataframe(
        health_df[[
            'Satellite', 'Health Status', 'Overall Score', 
            'Target Incl. (°)', 'Mean Incl. (°)', 'Incl. Dev. (°)',
            'Maneuvers/Month', 'Uniformity (CoV)'
        ]],
        hide_index=True,
        use_container_width=True
    )
    
    with st.expander("📋 View Detailed Health Remarks"):
        for _, row in health_df.iterrows():
            st.markdown(f"### {row['Satellite']} - {row['Health Status']}")
            st.markdown(f"**Overall Score:** {row['Overall Score']}/100")
            st.markdown("**Remarks:**")
            for remark in row['Remarks'].split(' | '):
                st.markdown(f"- {remark}")
            st.markdown("---")
    
    st.divider()
    
    # ==================== SATELLITE CLASSIFICATION ====================
    st.header("🔍 Satellite Classification")
    
    sat_summary = []
    for sat_name in sorted(df_all['satellite'].unique()):
        sub = df_all[df_all['satellite'] == sat_name]
        mean_incl = sub['INCLINATION'].mean() if not sub.empty else float('nan')
        mean_alt = sub['altitude_km'].mean() if not sub.empty and 'altitude_km' in sub.columns else float('nan')
        if 0.0 < mean_incl < 10.0:
            sat_type = 'GSO'
        elif mean_incl >= 10.0:
            sat_type = 'IGSO'
        else:
            sat_type = 'Unclassified'
        sat_summary.append({
            'Satellite': sat_name,
            'Mean Inclination (°)': round(mean_incl, 3) if not sub.empty else None,
            'Mean Altitude (km)': round(mean_alt, 2) if not pd.isna(mean_alt) else None,
            'Classified Type': sat_type
        })
    sat_summary_df = pd.DataFrame(sat_summary)
    st.dataframe(sat_summary_df, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # ==================== MANEUVER SUMMARY ====================
    st.header("🛠️ Maneuver Summary")
    
    maneuver_summary_df = pd.DataFrame(maneuver_summary)
    st.caption(f"Detection settings: Z-score ≥ {z_threshold}, SMA ≥ {sma_threshold} km, Inclination ≥ {inc_threshold}°, Window = {int(persist_window)}")
    st.dataframe(maneuver_summary_df, hide_index=True, use_container_width=True)
    
    st.divider()
    
    # ==================== DOP ANALYSIS ====================
    st.header("📡 Dilution of Precision (DOP) Analysis")
    
    with st.spinner("🔄 Fetching latest TLE data for DOP calculations..."):
        try:
            norad_ids = list(NAVIK_SATS.values())
            tle_data = fetch_multiple_tles(norad_ids, username, password)
            
            if not tle_data:
                st.error("❌ Failed to fetch TLE data for DOP calculations")
            else:
                satellites = parse_tle_data(tle_data, NAVIK_SATS)
                
                if len(satellites) == 0:
                    st.error("❌ No satellites parsed from TLE data")
                else:
                    st.success(f"✅ Successfully loaded {len(satellites)} satellites for DOP calculations")
                    
                    current_time = datetime.utcnow().replace(tzinfo=utc)
                    st.caption(f"Calculation Time (UTC): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    dop_results = []
                    
                    for location_name, (lat, lon) in INDIA_EXTREME_POINTS.items():
                        dop, visible_sats, sat_positions = calculate_dop_for_location(
                            satellites, lat, lon, current_time
                        )
                        
                        if dop:
                            gdop = dop['GDOP']
                            if gdop < 2:
                                quality = "Excellent"
                            elif gdop < 4:
                                quality = "Good"
                            elif gdop < 6:
                                quality = "Moderate"
                            elif gdop < 8:
                                quality = "Fair"
                            else:
                                quality = "Poor"
                            
                            dop_results.append({
                                'Location': location_name,
                                'Latitude': lat,
                                'Longitude': lon,
                                'Visible Sats': len(visible_sats),
                                'GDOP': round(dop['GDOP'], 2),
                                'PDOP': round(dop['PDOP'], 2),
                                'HDOP': round(dop['HDOP'], 2),
                                'VDOP': round(dop['VDOP'], 2),
                                'TDOP': round(dop['TDOP'], 2),
                                'Quality': quality
                            })
                        else:
                            dop_results.append({
                                'Location': location_name,
                                'Latitude': lat,
                                'Longitude': lon,
                                'Visible Sats': len(visible_sats),
                                'GDOP': None,
                                'PDOP': None,
                                'HDOP': None,
                                'VDOP': None,
                                'TDOP': None,
                                'Quality': 'N/A'
                            })
                    
                    dop_df = pd.DataFrame(dop_results)
                    st.dataframe(dop_df, hide_index=True, use_container_width=True)
                    
                    st.caption("**DOP Quality Guide:** Excellent: <2 | Good: 2-4 | Moderate: 4-6 | Fair: 6-8 | Poor: >8")
                    
                    # Store for plotting
                    st.session_state['satellites_dop'] = satellites
                    st.session_state['dop_results'] = dop_results
                    st.session_state['current_time'] = current_time
                    
        except Exception as e:
            st.error(f"❌ Error during DOP analysis: {str(e)}")
    
    st.divider()
    
    # ==================== PLOTTING SECTION ====================
    st.header("📊 Visualizations")
    
    show_plots = st.button("🎨 Generate All Plots", type="primary")
    
    if show_plots or st.session_state.get('show_plots', False):
        st.session_state['show_plots'] = True
        st.subheader("Individual Satellite Plots")
        
        for sat_name in sorted(df_all['satellite'].unique()):
            sat_df = df_all[df_all['satellite'] == sat_name].copy()
            
            st.markdown(f"### {sat_name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_incl = px.line(
                    sat_df,
                    x='EPOCH',
                    y='INCLINATION',
                    markers=True,
                    title=f"{sat_name} - Inclination Over Time",
                    labels={'EPOCH': 'Epoch', 'INCLINATION': 'Inclination (°)'},
                    hover_data=['INCLINATION', 'type']
                )
                fig_incl.update_traces(line_color='#636EFA')
                fig_incl.update_layout(hovermode='x unified', showlegend=False)
                st.plotly_chart(fig_incl, use_container_width=True)
            
            with col2:
                if 'altitude_km' in sat_df.columns and not sat_df['altitude_km'].isna().all():
                    fig_alt = px.line(
                        sat_df,
                        x='EPOCH',
                        y='altitude_km',
                        markers=True,
                        title=f"{sat_name} - Altitude Above Surface",
                        labels={'EPOCH': 'Epoch', 'altitude_km': 'Altitude (km)'}
                    )
                    fig_alt.update_traces(line_color='#EF553B')
                    fig_alt.update_layout(hovermode='x unified', showlegend=False)
                    st.plotly_chart(fig_alt, use_container_width=True)
                else:
                    st.info(f"No altitude data available for {sat_name}")
            
            st.markdown("---")
        
        # Satellite Bounding Box Plots
        st.subheader("🗺️ Satellite Ground Track Bounding Boxes")
        st.caption("Shows the geographic coverage area for each satellite over the next 1.5 days")
        
        if st.session_state.get('satellites_dop') and st.session_state.get('current_time'):
            satellites = st.session_state['satellites_dop']
            reference_time = st.session_state['current_time']
            
            with st.spinner("Calculating satellite ground tracks..."):
                # Calculate bounding boxes
                bounding_boxes = calculate_bounding_boxes(
                    satellites, 
                    reference_time, 
                    timestep_minutes=15, 
                    prop_duration_days=1.5
                )
                
                if bounding_boxes:
                    # Create individual plots for each satellite
                    for sat_name, box_data in bounding_boxes.items():
                        st.markdown(f"#### {sat_name} Ground Track")
                        
                        fig = go.Figure()
                        
                        # Plot ground track
                        fig.add_trace(go.Scattergeo(
                            lon=box_data['longitudes'],
                            lat=box_data['latitudes'],
                            mode='lines+markers',
                            name=sat_name,
                            marker=dict(size=3),
                            line=dict(width=2)
                        ))
                        
                        # Add bounding box corners
                        box_lons = [
                            box_data['min_lon'], box_data['max_lon'], 
                            box_data['max_lon'], box_data['min_lon'], 
                            box_data['min_lon']
                        ]
                        box_lats = [
                            box_data['min_lat'], box_data['min_lat'], 
                            box_data['max_lat'], box_data['max_lat'], 
                            box_data['min_lat']
                        ]
                        
                        fig.add_trace(go.Scattergeo(
                            lon=box_lons,
                            lat=box_lats,
                            mode='lines',
                            name='Bounding Box',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Add center point
                        fig.add_trace(go.Scattergeo(
                            lon=[box_data['mean_lon']],
                            lat=[box_data['mean_lat']],
                            mode='markers',
                            name='Center',
                            marker=dict(size=10, color='red', symbol='x')
                        ))
                        
                        fig.update_geos(
                            projection_type="natural earth",
                            showland=True,
                            landcolor="lightgray",
                            showocean=True,
                            oceancolor="lightblue",
                            showcountries=True,
                            countrycolor="white",
                            showlakes=True,
                            lakecolor="lightblue",
                            center=dict(lon=box_data['mean_lon'], lat=box_data['mean_lat']),
                            projection_scale=3
                        )
                        
                        fig.update_layout(
                            title=f"{sat_name} - Geographic Coverage (1.5 days)",
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display bounding box statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Longitude Range", 
                                     f"{box_data['min_lon']:.2f}° to {box_data['max_lon']:.2f}°")
                        with col2:
                            st.metric("Latitude Range", 
                                     f"{box_data['min_lat']:.2f}° to {box_data['max_lat']:.2f}°")
                        with col3:
                            st.metric("Center Position", 
                                     f"({box_data['mean_lat']:.2f}°, {box_data['mean_lon']:.2f}°)")
                        
                        st.markdown("---")
                    
                    # Combined view of all satellites
                    st.markdown("#### All Satellites - Combined Ground Tracks")
                    
                    fig_combined = go.Figure()
                    
                    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
                    
                    for idx, (sat_name, box_data) in enumerate(bounding_boxes.items()):
                        color = colors[idx % len(colors)]
                        
                        # Plot ground track
                        fig_combined.add_trace(go.Scattergeo(
                            lon=box_data['longitudes'],
                            lat=box_data['latitudes'],
                            mode='lines',
                            name=sat_name,
                            line=dict(width=2, color=color),
                            showlegend=True
                        ))
                    
                    # Add India boundary outline
                    fig_combined.update_geos(
                        projection_type="natural earth",
                        showland=True,
                        landcolor="lightgray",
                        showocean=True,
                        oceancolor="lightblue",
                        showcountries=True,
                        countrycolor="white",
                        showlakes=True,
                        lakecolor="lightblue",
                        center=dict(lon=80, lat=20),
                        projection_scale=2
                    )
                    
                    fig_combined.update_layout(
                        title="All NavIC Satellites - Combined Ground Tracks",
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_combined, use_container_width=True)
                else:
                    st.warning("No bounding box data available for plotting.")
        else:
            st.info("Bounding box plots require DOP analysis data. Please ensure DOP analysis completed successfully.")
        
        # DOP Over Time Plot
        if st.session_state.get('satellites_dop') and st.session_state.get('dop_results'):
            st.subheader("DOP Over Time (30 Days)")
            
            satellites = st.session_state['satellites_dop']
            
            # Select location for DOP time series
            location_options = list(INDIA_EXTREME_POINTS.keys())
            selected_location = st.selectbox("Select Location for DOP Time Series", location_options)
            
            if selected_location:
                lat, lon = INDIA_EXTREME_POINTS[selected_location]
                
                with st.spinner(f"Calculating DOP over time for {selected_location}..."):
                    current_time = datetime.utcnow()
                    time_points = []
                    gdop_values = []
                    pdop_values = []
                    hdop_values = []
                    vdop_values = []
                    visible_sat_counts = []
                    
                    # Calculate DOP for each hour over 30 days
                    for hours in range(0, 30*24, 6):  # Every 6 hours
                        calc_time = current_time + timedelta(hours=hours)
                        dop, visible_sats, _ = calculate_dop_for_location(
                            satellites, lat, lon, calc_time
                        )
                        
                        time_points.append(calc_time)
                        visible_sat_counts.append(len(visible_sats))
                        
                        if dop:
                            gdop_values.append(dop['GDOP'])
                            pdop_values.append(dop['PDOP'])
                            hdop_values.append(dop['HDOP'])
                            vdop_values.append(dop['VDOP'])
                        else:
                            gdop_values.append(None)
                            pdop_values.append(None)
                            hdop_values.append(None)
                            vdop_values.append(None)
                    
                    # Create DOP time series plot
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=(f'DOP Values Over Time - {selected_location}', 
                                      'Visible Satellites Count'),
                        vertical_spacing=0.15
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=time_points, y=gdop_values, name='GDOP', 
                                 line=dict(color='#636EFA')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_points, y=pdop_values, name='PDOP', 
                                 line=dict(color='#EF553B')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_points, y=hdop_values, name='HDOP', 
                                 line=dict(color='#00CC96')),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=time_points, y=vdop_values, name='VDOP', 
                                 line=dict(color='#AB63FA')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=time_points, y=visible_sat_counts, name='Visible Satellites',
                                 line=dict(color='#FFA15A'), fill='tozeroy'),
                        row=2, col=1
                    )
                    
                    fig.update_xaxes(title_text="Date", row=1, col=1)
                    fig.update_xaxes(title_text="Date", row=2, col=1)
                    fig.update_yaxes(title_text="DOP Value", row=1, col=1)
                    fig.update_yaxes(title_text="Count", row=2, col=1)
                    
                    fig.update_layout(height=800, showlegend=True, hovermode='x unified')
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Combined inclination plot
        st.subheader("All Satellites - Inclination Comparison")
        fig_all_incl = px.line(
            df_all,
            x='EPOCH',
            y='INCLINATION',
            color='satellite',
            markers=False,
            title="All NavIC Satellites - Inclination Over Time",
            labels={'EPOCH': 'Epoch', 'INCLINATION': 'Inclination (°)', 'satellite': 'Satellite'}
        )
        fig_all_incl.update_layout(hovermode='x unified', height=500)
        st.plotly_chart(fig_all_incl, use_container_width=True)
        
        # Combined altitude plot
        if 'altitude_km' in df_all.columns and not df_all['altitude_km'].isna().all():
            st.subheader("All Satellites - Altitude Comparison")
            fig_all_alt = px.line(
                df_all[df_all['altitude_km'].notna()],
                x='EPOCH',
                y='altitude_km',
                color='satellite',
                markers=False,
                title="All NavIC Satellites - Altitude Over Time",
                labels={'EPOCH': 'Epoch', 'altitude_km': 'Altitude (km)', 'satellite': 'Satellite'}
            )
            fig_all_alt.update_layout(hovermode='x unified', height=500)
            st.plotly_chart(fig_all_alt, use_container_width=True)
        
else:
    st.info("👆 Click the button in the sidebar to start the analysis")
    st.markdown("""
    ### Welcome to NavIC Comprehensive Monitoring System
    
    This application provides comprehensive monitoring and analysis of NavIC (IRNSS/NVS) satellites including:
    
    - **🏥 Health Assessment**: Comprehensive health scoring based on orbital parameters and maneuver patterns
    - **🔍 Satellite Classification**: GSO/IGSO classification based on inclination
    - **🛠️ Maneuver Detection**: Automated detection of orbital correction maneuvers
    - **📡 DOP Analysis**: Dilution of Precision calculations for key locations in India
    - **📊 Visualizations**: Time series plots for inclination, altitude, and DOP values
    
    **To get started:**
    1. Enter your Space-Track credentials in the sidebar
    2. Select the date range for analysis
    3. Adjust analysis parameters if needed
    4. Click "Fetch NavIC Data & Run Analysis"
    """)