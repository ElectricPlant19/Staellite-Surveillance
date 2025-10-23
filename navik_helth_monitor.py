import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
import numpy as np
import plotly.graph_objects as go

MU = 398600.4418  # km^3/s^2
R_EARTH = 6371.0  # km

st.title("NavIK (IRNSS/NVS) - Health Analysis")

# Space-Track credentials
username = st.text_input("Space-Track username", value="", type="default")
password = st.text_input("Space-Track password", value="", type="password")

# Date range
start_date = st.date_input("Start date", value=date(2025, 1, 1))
end_date = st.date_input("End date", value=date(2025, 10, 1))
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

# Option: keep only first TLE per day
daily_only = st.checkbox("Keep only one TLE per day (first entry)", value=True)

# Maneuver detection parameters
st.markdown("### Maneuver Detection Parameters")
with st.expander("Advanced Settings (click to expand)", expanded=False):
    col_param1, col_param2 = st.columns(2)
    
    with col_param1:
        z_threshold = st.number_input(
            "Z-Score Threshold",
            min_value=1.0,
            max_value=10.0,
            value=3.5,
            step=0.5,
            help="Statistical significance threshold. Higher values = fewer, more significant detections."
        )
        sma_threshold = st.number_input(
            "SMA Change Threshold (km)",
            min_value=0.1,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Minimum semi-major axis change to consider as maneuver (in kilometers)."
        )
    
    with col_param2:
        inc_threshold = st.number_input(
            "Inclination Change Threshold (degrees)",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Minimum inclination change to consider as maneuver (in degrees)."
        )
        persist_window = st.number_input(
            "Persistence Window",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Number of data points to check for sustained change. Higher = more conservative."
        )

# Health Assessment Parameters
st.markdown("### Health Assessment Parameters")
with st.expander("Health Tolerance Settings (click to expand)", expanded=False):
    col_health1, col_health2 = st.columns(2)
    
    with col_health1:
        inclination_tolerance = st.number_input(
            "Inclination Tolerance (degrees)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Acceptable deviation from target inclination"
        )
        min_maneuvers_per_month = st.number_input(
            "Min Maneuvers/Month (for active maintenance)",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            help="Minimum expected maneuvers per month for healthy satellite"
        )
    
    with col_health2:
        max_maneuvers_per_month = st.number_input(
            "Max Maneuvers/Month (normal operation)",
            min_value=1,
            max_value=20,
            value=8,
            step=1,
            help="Maximum expected maneuvers per month for normal operation"
        )
        maneuver_uniformity_threshold = st.number_input(
            "Maneuver Uniformity Threshold (CoV)",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Coefficient of Variation threshold for uniform maneuver spacing"
        )

# Service requirements for NavIC satellites
NAVIK_SERVICE_REQUIREMENTS = {
    "IRNSS-1B": {"longitude": 55.0, "inclination": 29.0},
    "IRNSS-1C": {"longitude": 83.0, "inclination": 5.0},
    "IRNSS-1D": {"longitude": 111.75, "inclination": 30.0},  # Using middle of 29-31 range
    "IRNSS-1E": {"longitude": 111.75, "inclination": 29.0},
    "IRNSS-1F": {"longitude": 32.5, "inclination": 5.0},
    "IRNSS-1I": {"longitude": 55.0, "inclination": 29.0},  # Assuming similar to 1B (replacement)
    "NVS-01": {"longitude": 129.5, "inclination": 5.0}
}


# MANEUVER DETECTION FUNCTIONS
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

def detect_navik_maneuvers(df,
                           sma_col='SEMIMAJOR_AXIS',
                           inc_col='INCLINATION',
                           z_thresh=3.5,
                           sma_abs_thresh_km=0.5,
                           inc_abs_thresh_deg=0.01,
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


# HEALTH ASSESSMENT FUNCTIONS
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

def assess_satellite_health(sat_name, sat_df, maneuver_events,
                           inc_tolerance, min_man_per_month, 
                           max_man_per_month, uniformity_threshold):
    """Comprehensive health assessment for a satellite."""
    
    # Get service requirements
    requirements = NAVIK_SERVICE_REQUIREMENTS.get(sat_name, {})
    target_inclination = requirements.get("inclination", None)
    
    # Calculate metrics
    mean_inclination = sat_df['INCLINATION'].mean()
    std_inclination = sat_df['INCLINATION'].std()
    
    observation_days = (sat_df['EPOCH'].max() - sat_df['EPOCH'].min()).days
    observation_months = observation_days / 30.0
    
    num_maneuvers = len(maneuver_events)
    maneuvers_per_month = num_maneuvers / observation_months if observation_months > 0 else 0
    
    # Inclination deviation score (0-100, higher is better)
    if target_inclination is not None:
        inc_deviation = abs(mean_inclination - target_inclination)
        inc_score = max(0, 100 - (inc_deviation / inc_tolerance) * 100)
    else:
        inc_score = None
        inc_deviation = None
    
    # Maintenance activity score (0-100)
    if maneuvers_per_month < min_man_per_month:
        maintenance_score = 30  # Insufficient maintenance
    elif maneuvers_per_month > max_man_per_month:
        maintenance_score = 60  # Over-correction, possible issues
    else:
        maintenance_score = 100  # Normal maintenance
    
    # Maneuver uniformity score
    if num_maneuvers >= 2:
        maneuver_dates = pd.to_datetime(maneuver_events['EPOCH']).tolist()
        uniformity_cov = calculate_maneuver_uniformity(maneuver_dates)
        
        if uniformity_cov is not None and uniformity_cov <= uniformity_threshold:
            uniformity_score = 100  # Uniform spacing
        elif uniformity_cov is not None:
            uniformity_score = max(0, 100 - ((uniformity_cov - uniformity_threshold) / uniformity_threshold) * 50)
        else:
            uniformity_score = 50
    else:
        uniformity_score = 50 if num_maneuvers == 1 else 0
        uniformity_cov = None
    
    # Overall health score (weighted average)
    if inc_score is not None:
        overall_score = (inc_score * 0.5 + maintenance_score * 0.3 + uniformity_score * 0.2)
    else:
        overall_score = (maintenance_score * 0.6 + uniformity_score * 0.4)
    
    # Health status determination
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
    
    # Generate remarks
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


# CACHING HELPERS
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"

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
        f"https://www.space-track.org/basicspacedata/query/class/gp_history/EPOCH/{start_date}--{end_date}/NORAD_CAT_ID/{norad_id}/orderby/EPOCH asc/format/json"
    )
    resp = session.get(gp_url)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch GP data for {norad_id}: HTTP {resp.status_code}")
    data = resp.json()
    return data

def fetch_and_classify_satellite(norad_id: int, start_date: str, end_date: str,
                                 username: str, password: str,
                                 gso_center=5.0, gso_tol=1, igso_min=10, deviation_tol=0.3):
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


# NavIK satellites
navik_sats = {
    "IRNSS-1B": 39635,
    "IRNSS-1C": 40269,
    "IRNSS-1D": 40547,
    "IRNSS-1E": 41241,
    "IRNSS-1F": 41384,
    "IRNSS-1I": 43286,
    "NVS-01": 56759
}

if st.button("Fetch NavIK & Analyze Health"):
    if not username or not password:
        st.error("Enter Space-Track username and password.")
    else:
        all_dfs = []
        errors = {}
        with st.spinner("Fetching GP data for NavIK satellites..."):
            for sat_name, norad in navik_sats.items():
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
            st.warning("Some satellites failed to fetch. See details below.")
            for s, msg in errors.items():
                st.write(f"- **{s}**: {msg}")

        if not all_dfs:
            st.error("No data fetched for any satellite.")
        else:
            df_all = pd.concat(all_dfs, ignore_index=True, sort=False)

            # MANEUVER DETECTION
            st.markdown("### Maneuver Detection Analysis")
            
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
                
                # HEALTH ASSESSMENT
                health_data = assess_satellite_health(
                    sat_name, sat_df, maneuver_events,
                    inclination_tolerance, min_maneuvers_per_month,
                    max_maneuvers_per_month, maneuver_uniformity_threshold
                )
                health_assessments.append(health_data)
            
            # Display maneuver summary (WITHOUT Total Maneuvers column)
            maneuver_summary_df = pd.DataFrame(maneuver_summary)
            st.markdown("#### Maneuver Summary by Satellite")
            st.caption(f"Detection settings: Z-score ≥ {z_threshold}, SMA ≥ {sma_threshold} km, Inclination ≥ {inc_threshold}°, Window = {int(persist_window)}")
            st.dataframe(maneuver_summary_df, hide_index=True, use_container_width=True)
            
            # Display detailed maneuver events
            if len(all_maneuvers_df) > 0:
                st.markdown("#### Detailed Maneuver Events")
                display_cols = ['satellite', 'EPOCH', 'dSMA', 'dINC', 'SMA_MANEUVER', 'INC_MANEUVER']
                maneuver_display = all_maneuvers_df[display_cols].copy()
                maneuver_display['EPOCH'] = pd.to_datetime(maneuver_display['EPOCH']).dt.strftime('%Y-%m-%d %H:%M:%S')
                maneuver_display['dSMA'] = maneuver_display['dSMA'].round(3)
                maneuver_display['dINC'] = maneuver_display['dINC'].round(5)
                st.dataframe(maneuver_display, hide_index=True, use_container_width=True)
            else:
                st.info("No maneuvers detected in the selected date range with current thresholds.")
            
            st.divider()
            
            # HEALTH ASSESSMENT TABLE
            st.markdown("### 🏥 Satellite Health Assessment")
            health_df = pd.DataFrame(health_assessments)
            
            # Display health summary
            st.dataframe(
                health_df[[
                    'Satellite', 'Health Status', 'Overall Score', 
                    'Target Incl. (°)', 'Mean Incl. (°)', 'Incl. Dev. (°)',
                    'Maneuvers/Month', 'Uniformity (CoV)'
                ]],
                hide_index=True,
                use_container_width=True
            )
            
            # Display detailed remarks
            st.markdown("#### Health Remarks")
            for _, row in health_df.iterrows():
                with st.expander(f"{row['Satellite']} - {row['Health Status']}"):
                    st.markdown(f"**Overall Score:** {row['Overall Score']}/100")
                    st.markdown(f"**Remarks:**")
                    for remark in row['Remarks'].split(' | '):
                        st.markdown(f"- {remark}")
            
            st.divider()

            # Satellite classification summary
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
            st.markdown("### Satellite Classification Summary")
            st.table(sat_summary_df)

            # Individual satellite plots
            st.markdown("### Individual Satellite Plots")
            
            for sat_name in sorted(df_all['satellite'].unique()):
                sat_df = df_all[df_all['satellite'] == sat_name].copy()
                
                st.markdown(f"#### {sat_name}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_incl = px.line(
                        sat_df,
                        x='EPOCH',
                        y='INCLINATION',
                        markers=True,
                        title=f"{sat_name} - Inclination",
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
                
                st.divider()
