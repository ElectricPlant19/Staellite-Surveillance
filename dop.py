import streamlit as st
import numpy as np
from skyfield.api import load, EarthSatellite, wgs84
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO

# IRNSS/NavIC satellite catalog numbers
NAVIK_SATS = {
    "IRNSS-1B": 39635,
    "IRNSS-1C": 40269,
    "IRNSS-1D": 40547,
    "IRNSS-1E": 41241,
    "IRNSS-1F": 41384,
    "IRNSS-1I": 43286,
    "NVS-01": 56759
}

# Extreme points of India (approximate)
INDIA_EXTREME_POINTS = {
    "Northernmost (Siachen Glacier)": (35.5, 77.0),
    "Southernmost (Indira Point)": (6.75, 93.85),
    "Easternmost (Kibithu)": (28.0, 97.0),
    "Westernmost (Guhar Moti)": (23.7, 68.1)
}

class SpaceTrackClient:
    """Client for Space-Track.org API"""
    
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.base_url = "https://www.space-track.org"
        self.session = None
    
    def login(self):
        """Login to Space-Track"""
        self.session = requests.Session()
        login_url = f"{self.base_url}/ajaxauth/login"
        data = {
            'identity': self.username,
            'password': self.password
        }
        try:
            response = self.session.post(login_url, data=data)
            response.raise_for_status()
            return True
        except Exception as e:
            st.error(f"Login failed: {str(e)}")
            return False
    
    def get_tle(self, norad_id):
        """Get TLE data for a specific NORAD ID"""
        if not self.session:
            if not self.login():
                return None
        
        query_url = f"{self.base_url}/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/{norad_id}/orderby/TLE_LINE1 ASC/format/3le"
        
        try:
            response = self.session.get(query_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            st.error(f"Error fetching TLE for {norad_id}: {str(e)}")
            return None
    
    def get_multiple_tles(self, norad_ids):
        """Get TLE data for multiple NORAD IDs"""
        if not self.session:
            if not self.login():
                return {}
        
        # Join NORAD IDs with comma
        ids_str = ','.join(map(str, norad_ids))
        query_url = f"{self.base_url}/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/{ids_str}/orderby/NORAD_CAT_ID,ORDINAL/format/3le"
        
        try:
            response = self.session.get(query_url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            st.error(f"Error fetching TLEs: {str(e)}")
            return ""

def parse_tle_data(tle_text, sat_dict):
    """Parse TLE text and create satellite objects"""
    ts = load.timescale()
    satellites = {}
    
    lines = tle_text.strip().split('\n')
    
    # Process TLE in groups of 3 (name, line1, line2)
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
        
        name = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()
        
        # Extract NORAD ID from line 1
        try:
            norad_id = int(line1[2:7])
            
            # Find satellite name from our dictionary
            sat_name = None
            for s_name, s_id in sat_dict.items():
                if s_id == norad_id:
                    sat_name = s_name
                    break
            
            if sat_name:
                satellite = EarthSatellite(line1, line2, sat_name, ts)
                satellites[sat_name] = satellite
        except (ValueError, IndexError) as e:
            st.warning(f"Error parsing TLE: {str(e)}")
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
    except Exception as e:
        st.warning(f"Error calculating position: {str(e)}")
        return None

def calculate_design_matrix(satellite_positions, observer_lat, observer_lon):
    """
    Calculate the geometry matrix (design matrix) for DOP calculation
    H matrix where each row represents the unit vector from receiver to satellite
    """
    H = []
    
    for pos in satellite_positions:
        if pos is None:
            continue
            
        # Only include satellites above horizon (elevation > 5 degrees for better geometry)
        if pos['elevation'] > 5:
            # Convert azimuth and elevation to direction cosines
            az_rad = np.radians(pos['azimuth'])
            el_rad = np.radians(pos['elevation'])
            
            # Direction cosines (East, North, Up)
            dx = np.cos(el_rad) * np.sin(az_rad)
            dy = np.cos(el_rad) * np.cos(az_rad)
            dz = np.sin(el_rad)
            
            # Add row: [dx, dy, dz, 1] - the 1 is for clock bias
            H.append([dx, dy, dz, 1])
    
    return np.array(H) if H else np.array([]).reshape(0, 4)

def calculate_dop_values(H):
    """
    Calculate various DOP values from the design matrix
    DOP = sqrt(trace(Q)) where Q = (H^T * H)^-1
    """
    if len(H) < 4:
        return None  # Need at least 4 satellites
    
    try:
        # Calculate (H^T * H)^-1
        HTH = np.dot(H.T, H)
        
        # Check if matrix is singular
        if np.linalg.det(HTH) == 0:
            return None
            
        Q = np.linalg.inv(HTH)
        
        # Extract DOP values
        dop = {
            'GDOP': float(np.sqrt(np.trace(Q))),  # Geometric DOP
            'PDOP': float(np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])),  # Position DOP
            'HDOP': float(np.sqrt(Q[0,0] + Q[1,1])),  # Horizontal DOP
            'VDOP': float(np.sqrt(Q[2,2])),  # Vertical DOP
            'TDOP': float(np.sqrt(Q[3,3])),  # Time DOP
        }
        
        return dop
    except np.linalg.LinAlgError:
        return None  # Singular matrix
    except Exception as e:
        st.warning(f"Error calculating DOP: {str(e)}")
        return None

def calculate_dop_for_location(satellites_dict, lat, lon, time):
    """Calculate DOP for a specific location and time"""
    ts = load.timescale()
    t = ts.utc(time.year, time.month, time.day, time.hour, time.minute, time.second)
    
    # Create observer location
    observer = wgs84.latlon(lat, lon)
    
    # Calculate positions for all satellites
    satellite_positions = []
    visible_sats = []
    
    for sat_name, sat_obj in satellites_dict.items():
        pos = calculate_satellite_position(sat_obj, t, observer)
        if pos:
            satellite_positions.append(pos)
            if pos['elevation'] > 5:  # 5 degree elevation mask
                visible_sats.append(sat_name)
    
    # Calculate design matrix
    H = calculate_design_matrix(satellite_positions, lat, lon)
    
    # Calculate DOP
    dop = calculate_dop_values(H)
    
    return dop, visible_sats, satellite_positions

def main():
    st.set_page_config(page_title="IRNSS NavIC DOP Calculator", layout="wide")
    
    st.title("🛰️ IRNSS/NavIC Constellation DOP Calculator")
    st.markdown("""
    This application calculates the Dilution of Precision (DOP) for India's Navigation 
    with Indian Constellation (NavIC) satellite system at extreme points of India.
    """)
    
    # Sidebar controls
    st.sidebar.header("Space-Track Credentials")
    
    # Check if credentials are in session state
    if 'spacetrack_logged_in' not in st.session_state:
        st.session_state.spacetrack_logged_in = False
    
    username = st.sidebar.text_input("Space-Track Username", type="default")
    password = st.sidebar.text_input("Space-Track Password", type="password")
    
    if not st.session_state.spacetrack_logged_in:
        if st.sidebar.button("Login to Space-Track"):
            if username and password:
                with st.spinner("Logging in to Space-Track..."):
                    client = SpaceTrackClient(username, password)
                    if client.login():
                        st.session_state.spacetrack_client = client
                        st.session_state.spacetrack_logged_in = True
                        st.sidebar.success("✅ Logged in successfully!")
                        st.rerun()
            else:
                st.sidebar.error("Please enter both username and password")
        
        st.sidebar.info("🔑 You need a Space-Track account to use this app. Register at space-track.org")
        return
    else:
        st.sidebar.success("✅ Logged in to Space-Track")
        if st.sidebar.button("Logout"):
            st.session_state.spacetrack_logged_in = False
            del st.session_state.spacetrack_client
            st.rerun()
    
    st.sidebar.header("Settings")
    
    # Date and time selection
    selected_date = st.sidebar.date_input("Select Date", datetime.now())
    selected_time = st.sidebar.time_input("Select Time (UTC)", datetime.now().time())
    
    # Combine date and time
    observation_datetime = datetime.combine(selected_date, selected_time)
    
    # Time range for analysis
    analyze_range = st.sidebar.checkbox("Analyze Time Range (24 hours)", value=False)
    
    if analyze_range:
        time_interval = st.sidebar.slider("Time Interval (minutes)", 10, 120, 60)
    
    # Elevation mask
    elevation_mask = st.sidebar.slider("Elevation Mask (degrees)", 0, 15, 5)
    
    # Load TLE data
    if st.sidebar.button("Load TLE Data", type="primary"):
        with st.spinner("Fetching TLE data from Space-Track..."):
            try:
                client = st.session_state.spacetrack_client
                
                # Get TLE data for all NavIC satellites
                norad_ids = list(NAVIK_SATS.values())
                tle_data = client.get_multiple_tles(norad_ids)
                
                if tle_data:
                    # Parse TLE data
                    navic_satellites = parse_tle_data(tle_data, NAVIK_SATS)
                    
                    if navic_satellites:
                        st.session_state.navic_satellites = navic_satellites
                        st.sidebar.success(f"✅ Loaded {len(navic_satellites)} NavIC satellites")
                        
                        # Show loaded satellites
                        with st.sidebar.expander("Loaded Satellites"):
                            for sat_name in navic_satellites.keys():
                                st.write(f"✓ {sat_name}")
                        
                        missing = set(NAVIK_SATS.keys()) - set(navic_satellites.keys())
                        if missing:
                            st.warning(f"⚠️ Could not load: {', '.join(missing)}")
                    else:
                        st.error("Failed to parse TLE data")
                else:
                    st.error("No TLE data received")
                    
            except Exception as e:
                st.error(f"Error loading TLE data: {str(e)}")
    
    # Check if satellites are loaded
    if 'navic_satellites' not in st.session_state:
        st.info("👆 Please load TLE data from Space-Track to continue")
        return
    
    navic_satellites = st.session_state.navic_satellites
    
    # Main calculation
    if st.button("Calculate DOP", type="primary"):
        
        if not analyze_range:
            # Single time point calculation
            st.header(f"DOP Analysis for {observation_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            results = []
            
            for location_name, (lat, lon) in INDIA_EXTREME_POINTS.items():
                with st.spinner(f"Calculating for {location_name}..."):
                    dop, visible_sats, sat_positions = calculate_dop_for_location(
                        navic_satellites, lat, lon, observation_datetime
                    )
                    
                    results.append({
                        'Location': location_name,
                        'Latitude': lat,
                        'Longitude': lon,
                        'Visible Satellites': len(visible_sats),
                        'Satellite Names': ', '.join(visible_sats) if visible_sats else 'None',
                        'GDOP': f"{dop['GDOP']:.2f}" if dop else 'N/A',
                        'PDOP': f"{dop['PDOP']:.2f}" if dop else 'N/A',
                        'HDOP': f"{dop['HDOP']:.2f}" if dop else 'N/A',
                        'VDOP': f"{dop['VDOP']:.2f}" if dop else 'N/A',
                        'TDOP': f"{dop['TDOP']:.2f}" if dop else 'N/A',
                    })
            
            # Display results
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            st.subheader("DOP Comparison")
            
            # Convert string values back to float for plotting
            plot_data = []
            for result in results:
                if result['GDOP'] != 'N/A':
                    plot_data.append({
                        'Location': result['Location'],
                        'GDOP': float(result['GDOP']),
                        'PDOP': float(result['PDOP']),
                        'HDOP': float(result['HDOP']),
                        'VDOP': float(result['VDOP']),
                    })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('GDOP', 'PDOP', 'HDOP', 'VDOP')
                )
                
                # GDOP
                fig.add_trace(
                    go.Bar(x=plot_df['Location'], y=plot_df['GDOP'], name='GDOP', marker_color='lightblue'),
                    row=1, col=1
                )
                
                # PDOP
                fig.add_trace(
                    go.Bar(x=plot_df['Location'], y=plot_df['PDOP'], name='PDOP', marker_color='lightgreen'),
                    row=1, col=2
                )
                
                # HDOP
                fig.add_trace(
                    go.Bar(x=plot_df['Location'], y=plot_df['HDOP'], name='HDOP', marker_color='lightyellow'),
                    row=2, col=1
                )
                
                # VDOP
                fig.add_trace(
                    go.Bar(x=plot_df['Location'], y=plot_df['VDOP'], name='VDOP', marker_color='lightcoral'),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid DOP data to plot. Need at least 4 visible satellites.")
            
            # Interpretation
            st.subheader("📊 DOP Interpretation")
            st.markdown("""
            **DOP Values Interpretation:**
            - **1-2**: Excellent
            - **2-5**: Good
            - **5-10**: Moderate
            - **10-20**: Fair
            - **>20**: Poor
            
            **DOP Types:**
            - **GDOP**: Geometric Dilution of Precision (overall quality)
            - **PDOP**: Position Dilution of Precision (3D position accuracy)
            - **HDOP**: Horizontal Dilution of Precision (latitude/longitude accuracy)
            - **VDOP**: Vertical Dilution of Precision (altitude accuracy)
            - **TDOP**: Time Dilution of Precision (clock accuracy)
            """)
        
        else:
            # Time range analysis
            st.header(f"24-Hour DOP Analysis")
            
            time_points = []
            current_time = observation_datetime
            end_time = observation_datetime + timedelta(hours=24)
            
            while current_time < end_time:
                time_points.append(current_time)
                current_time += timedelta(minutes=time_interval)
            
            # Select location for time analysis
            selected_location = st.selectbox(
                "Select Location for Time Analysis",
                list(INDIA_EXTREME_POINTS.keys())
            )
            
            lat, lon = INDIA_EXTREME_POINTS[selected_location]
            
            progress_bar = st.progress(0)
            time_series_data = []
            
            for i, time_point in enumerate(time_points):
                dop, visible_sats, _ = calculate_dop_for_location(
                    navic_satellites, lat, lon, time_point
                )
                
                if dop:
                    time_series_data.append({
                        'Time': time_point,
                        'Visible Satellites': len(visible_sats),
                        'GDOP': dop['GDOP'],
                        'PDOP': dop['PDOP'],
                        'HDOP': dop['HDOP'],
                        'VDOP': dop['VDOP'],
                    })
                
                progress_bar.progress((i + 1) / len(time_points))
            
            progress_bar.empty()
            
            if time_series_data:
                df_time = pd.DataFrame(time_series_data)
                
                # Plot time series
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(x=df_time['Time'], y=df_time['GDOP'], mode='lines+markers', name='GDOP'))
                fig.add_trace(go.Scatter(x=df_time['Time'], y=df_time['PDOP'], mode='lines+markers', name='PDOP'))
                fig.add_trace(go.Scatter(x=df_time['Time'], y=df_time['HDOP'], mode='lines+markers', name='HDOP'))
                fig.add_trace(go.Scatter(x=df_time['Time'], y=df_time['VDOP'], mode='lines+markers', name='VDOP'))
                
                fig.update_layout(
                    title=f'DOP Values Over 24 Hours - {selected_location}',
                    xaxis_title='Time (UTC)',
                    yaxis_title='DOP Value',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Satellite visibility plot
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df_time['Time'], 
                    y=df_time['Visible Satellites'], 
                    mode='lines+markers',
                    fill='tozeroy',
                    name='Visible Satellites'
                ))
                
                fig2.update_layout(
                    title='Visible Satellites Over Time',
                    xaxis_title='Time (UTC)',
                    yaxis_title='Number of Visible Satellites',
                    height=300
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Statistics
                st.subheader("Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg GDOP", f"{df_time['GDOP'].mean():.2f}")
                    st.metric("Min GDOP", f"{df_time['GDOP'].min():.2f}")
                    st.metric("Max GDOP", f"{df_time['GDOP'].max():.2f}")
                
                with col2:
                    st.metric("Avg PDOP", f"{df_time['PDOP'].mean():.2f}")
                    st.metric("Min PDOP", f"{df_time['PDOP'].min():.2f}")
                    st.metric("Max PDOP", f"{df_time['PDOP'].max():.2f}")
                
                with col3:
                    st.metric("Avg HDOP", f"{df_time['HDOP'].mean():.2f}")
                    st.metric("Min HDOP", f"{df_time['HDOP'].min():.2f}")
                    st.metric("Max HDOP", f"{df_time['HDOP'].max():.2f}")
                
                with col4:
                    st.metric("Avg Visible Sats", f"{df_time['Visible Satellites'].mean():.1f}")
                    st.metric("Min Visible Sats", f"{int(df_time['Visible Satellites'].min())}")
                    st.metric("Max Visible Sats", f"{int(df_time['Visible Satellites'].max())}")
                
                # Downloadable data
                csv = df_time.to_csv(index=False)
                st.download_button(
                    label="📥 Download Time Series Data (CSV)",
                    data=csv,
                    file_name=f"navic_dop_timeseries_{selected_location.replace(' ', '_').replace('(', '').replace(')', '')}_{observation_datetime.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid DOP data generated. Check satellite visibility.")

if __name__ == "__main__":
    main()