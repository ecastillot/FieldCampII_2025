import pandas as pd
import datetime
import matplotlib.pyplot as plt
import glob
import os
import obsplus
from obspy import read, Stream, UTCDateTime
from obspy import read_inventory
from pyproj import Transformer
import re
from typing import List
import numpy as np

def get_station_from_solo(folder_path: str) -> pd.DataFrame:
    """
    Reads all DigiSolo log files in a specified folder and extracts station information.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing DigiSolo log files.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing station names, latitude, longitude, elevation, and file paths.
    """
    data = []
    for filepath in glob.glob(os.path.join(folder_path, "**","*.LOG"),recursive=True):
        try:
            station_info = get_latlon_from_digisolo_log(filepath)
            data.append(station_info)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    if not data:
        raise ValueError("No valid DigiSolo log files found in the specified folder.")
    
    return pd.DataFrame(data)



def append_xy_coords(df, lat_col='latitude', lon_col='longitude', elev_col_in_km=None,epsg="epsg:32614"):
    """
    Appends UTM coordinates (x, y) and elevation in meters to a DataFrame based on latitude and longitude.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing latitude and longitude columns.
    lat_col : str, optional
        Name of the column containing latitude values (default is 'latitude').
    lon_col : str, optional
        Name of the column containing longitude values (default is 'longitude').
    elev_col_in_km : str, optional
        Name of the column containing elevation values in kilometers (default is None, which means no elevation column).
    epsg : str, optional
        EPSG code for the target coordinate system (default is "epsg:32614" for UTM zone 14N).
    """
    # Define transformer (WGS84 to UTM zone based on lat/lon in Texas, e.g., UTM zone 14N)
    transformer = Transformer.from_crs("epsg:4326", epsg, always_xy=True)

    # Apply transformation
    x, y = transformer.transform(df[lon_col].values, df[lat_col].values)

    # Create a new DataFrame with x, y, elevation
    df['x'] = x
    df['y'] = y
    
    if elev_col_in_km is not None:
        df['elevation_m'] = df[elev_col_in_km]*1e3  # in meters
    else:
        df['elevation_m'] = None

    return df

def get_latlon_from_digisolo_log(path: str) -> dict:
    """
    Extracts latitude and longitude from a DigiSolo log file.

    Parameters
    ----------
    path : str
        Path to the DigiSolo log file.

    Returns
    -------
    dict
        dictionary with station name, latitude, longitude, elevation, and file path.
    """
    df = parse_digisolo_log_to_dataframe(path)
    
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("Log file does not contain Latitude or Longitude fields.")
    
    station_names = df.copy()["Serial Number"].dropna().unique()

    if len(station_names) != 1:
        raise ValueError("Log file contains multiple or no unique station names.")
    else:
        station_name = station_names[0]
    
    df = df.dropna(subset=["Latitude", "Longitude"])
    if df.empty:
        raise ValueError("No valid latitude and longitude records found in the log file.")
    
    
    lat = df["Latitude"].astype(float).mean()
    lon = df["Longitude"].astype(float).mean()
    elev= df["Altitude"].astype(float).mean() if "Altitude" in df.columns else None
    
    return {"station": station_name, "latitude": lat, "longitude": lon,"elevation":elev,"path": path}

def parse_digisolo_log_to_dataframe(path: str) -> pd.DataFrame:
    """
    Parses a DigiSolo log file and extracts GPS, temperature, and battery data into a DataFrame.

    Parameters
    ----------
    path : str
        Path to the DigiSolo log file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing extracted records with UTC time, GPS status, latitude, longitude,
        altitude, temperature, voltage, and other fields.
    """
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    blocks = []
    current_block = {}
    current_section = None

    section_pattern = re.compile(r"\[(\w+)(\d+)\]")
    key_value_pattern = re.compile(r"([\w\s]+)=\s*(.+)")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        section_match = section_pattern.match(line)
        if section_match:
            # If there was a previous block, save it
            if current_block:
                blocks.append(current_block)
            current_block = {
                "section": section_match.group(1),
                "index": section_match.group(2)
            }
            continue

        key_value_match = key_value_pattern.match(line)
        if key_value_match:
            key = key_value_match.group(1).strip()
            value = key_value_match.group(2).strip().strip('"')
            current_block[key] = value

    # Append the last block
    if current_block:
        blocks.append(current_block)

    # Create DataFrame
    df = pd.DataFrame(blocks)

    # # Convert UTC Time to datetime if present
    # if "UTC Time" in df.columns:
    #     df["UTC Time"] = pd.to_datetime(df["UTC Time"], errors="coerce")

    # Convert numeric fields where applicable
    numeric_fields = ["Latitude", "Longitude", "Altitude", "Voltage", "Temperature"]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    return df

def read_waveforms(
    folder_path: str,
    station: str = "*",
    component: str = "*",
    starttime: UTCDateTime = None,
    endtime: UTCDateTime = None
) -> Stream:
    """
    Reads MiniSEED files using station/component wildcards and filters by time range.

    Parameters:
    - folder_path (str): Directory with MiniSEED files.
    - station (str): Wildcard for station code (e.g., "4530*", "*" for all).
    - component (str): Wildcard for component (e.g., "E", "Z", "*").
    - starttime (UTCDateTime, optional): Start of time window to include.
    - endtime (UTCDateTime, optional): End of time window to include.

    Returns:
    - stream (obspy.Stream): Combined Stream object with filtered traces.
    """
    pattern = os.path.join(
        folder_path,
        f"{station}.0001.*.*.*.*.*.*.*.{component}.miniseed"
    )

    matched_files = glob.glob(pattern)
    matched_files.sort()

    stream = Stream()
    for path in matched_files:
        try:
            st = read(path)
            if starttime or endtime:
                st = st.slice(starttime=starttime, endtime=endtime)
            stream += st
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return stream

def read_stations(folder_path: str):
    """ 
    Reads all XML files in a specified folder and concatenates them into a single DataFrame.
    Parameters
    ----------
    folder_path : str
        Path to the folder containing XML files.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the concatenated data from all XML files.
    """
    dfs = []
    for filepath in glob.glob(os.path.join(folder_path, "*.xml")):
        try:
            inv = read_inventory(filepath)
            df = inv.to_df()
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    if not dfs:
        raise ValueError("No valid XML files found in the specified folder.")
    # Concatenate all DataFrames into one
    data = pd.concat(dfs, ignore_index=True)
    
    return data

def read_shot_from_file(filepath, gps_start=None):
    """
    Parses a custom-formatted CSV file containing GPS time and location data for each shot.

    Parameters
    ----------
    filepath : str
        Path to the input CSV file.
    gps_start : datetime.datetime or None
        The GPS epoch start time. If None, defaults to January 6, 1980.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: shot, year, month, day, hour, minute, second, latitude, longitude.
    """
    
    
    if gps_start is None:
        gps_start = datetime.datetime(1980, 1, 6)
    
    shots = []
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Group lines into blocks of 2 lines each (ignoring empty lines)
    blocks = [lines[i:i+2] for i in range(0, len(lines), 3) if len(lines[i:i+2]) == 2]

    for i, block in enumerate(blocks):
        time_line, coord_line = block
        parts = time_line.strip().split(',')
        week = int(parts[0].split('=')[1])
        ms = int(parts[1].split('=')[1])
        subms = int(parts[2].split('=')[1])

        # Convert GPS week and ms to datetime
        total_seconds = (week * 7 * 24 * 3600) + (ms / 1000) + (subms / 1e6)

        week_start = gps_start + datetime.timedelta(weeks=week)
        timestamp = week_start + datetime.timedelta(milliseconds=ms, microseconds=subms)

        lat, lat_sign = coord_line.split("Latitude:")[1].strip().split(" ")
        lon, lon_sign = coord_line.split("Longitude:")[1].split("Latitude:")[0].strip().split(" ")
        
        if lat_sign == 'S':
            lat = -float(lat)
        else:
            lat = float(lat)
        if lon_sign == 'W':
            lon = -float(lon)
        else:
            lon = float(lon)
        
        
        # Append the shot information
        shots.append({
            "shot": i + 1,
            "year": timestamp.year,
            "month": timestamp.month,
            "day": timestamp.day,
            "hour": timestamp.hour,
            "minute": timestamp.minute,
            "second": round(timestamp.second + timestamp.microsecond / 1e6, 2),
            "latitude": round(lat, 5),
            "longitude": round(lon, 5)  # Convert W to positive if required
        })
    shots = pd.DataFrame(shots)
    shots['time'] = pd.to_datetime(
                                    shots[['year', 'month', 'day',
                                           'hour', 'minute']].assign(Second=shots['second']),
                                                                format='%Y-%m-%d %H:%M:%S.%f'
                                                                )
    # Assuming your DataFrame is called df and it's sorted by time
    shots = shots.sort_values('time').reset_index(drop=True)
    return shots

def get_shots_data(shots_labeled, source_geometry):
    """
    Merge and process shot and source geometry data for seismic acquisition.

    This function merges labeled shots with source geometry, computes shot 
    coordinates (sx, sy, selev)

    Parameters
    ----------
    shots_labeled : pd.DataFrame
        DataFrame containing shot labels and metadata.
    source_geometry : pd.DataFrame
        DataFrame containing source geometry information.

    Returns
    -------
    pd.DataFrame
        A DataFrame  with the following columns:
        - 'shot_group'
        - 'shot'
        - 'time'
        - 'time_from_group_lead'
        - 'Shot Location'
        - 'sx' (x-coordinate in m)
        - 'sy' (set to 0)
        - 'selev' (set to 0)
        - 'x-coodinate (m)'
    """
    # Merge shot metadata with source geometry using external function
    shots_geometry = merge_shots_and_geometry(shots=shots_labeled,
                                              geometry=source_geometry)

    # Convert x-coordinate from meters to centimeters and cast to integer
    shots_geometry["sx"] = (shots_geometry["x-coodinate (m)"]).astype(int)

    # Add default values for sy and selev
    shots_geometry["sy"] = 0
    shots_geometry["selev"] = 0

    # Select and reorder relevant columns
    shots_geometry = shots_geometry[
        ["shot_group", "shot", "time", 'time_from_group_lead',"Shot Location",
         "sx", "sy", "selev", "x-coodinate (m)"]
    ]

    # Set 'shot' as the index
    # shots_geometry.set_index("shot", inplace=True)

    return shots_geometry

def get_receiver_data(receiver_geometry):
    """
    Process receiver geometry DataFrame for seismic processing.

    This function computes geometry columns (gx, gy, gelev), converts
    x-coordinates from meters to centimeters

    Parameters
    ----------
    receiver_geometry : pd.DataFrame
        DataFrame containing the receiver geometry. Must include:
        - 'Receiver Number'
        - 'Node ID'
        - 'x-coordinate (m)'

    Returns
    -------
    pd.DataFrame
        A DataFrame   with the following columns:
        - 'Receiver Number'
        - 'Node ID'
        - 'gx' (x-coordinate in m)
        - 'gy' (set to 0)
        - 'gelev' (set to 0)
        - 'x-coordinate (m)'
    """
    # Convert x-coordinate from meters to centimeters and cast to integer
    receiver_geometry["gx"] = (receiver_geometry["x-coordinate (m)"]).astype(int)

    # Add default values for gy and gelev
    receiver_geometry["gy"] = 0
    receiver_geometry["gelev"] = 0

    # Select and reorder relevant columns
    receiver_geometry = receiver_geometry[
        ["Receiver Number", "Node ID", "gx", "gy", "gelev", "x-coordinate (m)"]
    ]

    # Set 'Node ID' as the index
    receiver_geometry.set_index("Node ID", inplace=True)

    return receiver_geometry

def read_shots(shots, gps_start=None, source_time_separation=1e6,debug=False):
    """
    Parses a custom-formatted CSV file containing GPS time and location data for each shot.

    Parameters
    ----------
    shots : pd.DataFrame
        A DataFrame containing shot data with a 'time' column.
    gps_start : datetime.datetime or None
        The GPS epoch start time. If None, defaults to January 6, 1980.
    source_time_separation : float
        The maximum time separation between shots to consider them part of the same group (in seconds).
    debug : bool
        If True, prints debug information during processing.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: shot, year, month, day, hour, minute, second, latitude, longitude.
    """
    
    
    if isinstance(source_time_separation, dict):
        # Assuming source_time_separation is a dictionary with keys as shot group names
        selected_shots = []
        i=0
        for group_name, time_sep in source_time_separation.items():
            
            start_time = shots.loc[i, 'time']
            mask = (shots['time'] - start_time).dt.total_seconds() <= time_sep
            group = shots[mask & (shots.index >= i)]
            group["time_from_group_lead"] = (group['time'] - group['time'].iloc[0]).dt.total_seconds()
            group["shot_group"] = group_name  # Assign the group name
            print(f"\nGroup {group_name} started at {group.iloc[0].time} with {len(group)} shots")
            selected_shots.append(group)
            
            # Move to the next unprocessed shot after this group
            i = group.index[-1] + 1
                       
            if debug:
                print(group[["shot","time","time_from_group_lead","shot_group"]])
                
        if not selected_shots:
            gd_shots = pd.DataFrame(columns=shots.columns)
            bad_shots = shots.copy()
        else:    
            # Concatenate all selected groups
            gd_shots = pd.concat(selected_shots).reset_index(drop=True)
            # print(gd_shots)
            bad_shots = shots[~shots.index.isin(gd_shots.index)]
        
        
        if not bad_shots.empty:
            bad_shots.sort_values('time', inplace=True)
            bad_shots.reset_index(drop=True, inplace=True)
            bad_shots["time_from_group_lead"] = (bad_shots['time'] - bad_shots['time'].iloc[0]).dt.total_seconds()
            bad_shots["shot_group"] = "last_group"  # Assign a default group name for bad shots
            
            print(f"\nGroup 'last_group' with {len(bad_shots)} shots")
            
            if debug:
                print(bad_shots[["shot","time","time_from_group_lead","shot_group"]])
            
            gd_shots = pd.concat([gd_shots, bad_shots]).reset_index(drop=True)
            # print(gd_shots)
            # exit()
    else:
        raise ValueError("source_time_separation must be a dictionary with group names as keys.")
            
    first_cols = ['shot_group', 'shot', 'time', 'time_from_group_lead']
    cols = first_cols +\
            [col for col in shots.columns if col not in first_cols]
    gd_shots = gd_shots[cols]
    return gd_shots

def merge_shots(shots_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges multiple DataFrames containing shot data into a single DataFrame.
    Parameters
    ----------
    shots_list : List[pd.DataFrame]
        List of DataFrames, each containing shot data with a 'time' column.
    Returns
    -------
    pd.DataFrame
        A single DataFrame containing all shots, sorted by time.
    """
    if not shots_list:
        raise ValueError("The shots_list is empty. Please provide at least one DataFrame.")
    merged_shots = pd.concat(shots_list, ignore_index=True)
    merged_shots.sort_values('time', inplace=True)
    merged_shots["shot"] = range(1,len(merged_shots)+1)  # Assign shot numbers starting from 1
    merged_shots.reset_index(drop=True, inplace=True)
    return merged_shots

def read_shots_from_folder(folder_path: str, gps_start=None, 
                           source_time_separation=None, 
                           bad_shots=None,debug=False) -> pd.DataFrame:
    """
    Reads all shot files in a specified folder and merges them into a single DataFrame.
    Parameters
    ----------
    folder_path : str
        Path to the folder containing shot files.
    gps_start : datetime.datetime or None
        The GPS epoch start time. If None, defaults to January 6, 1980.
    source_time_separation : dict
        The maximum time separation between shots to consider them part of the same group (in seconds).
        example: {"P1": 76.45, "S1_N": 21, ...}
    bad_shots : list or None
        A list of shot groups to exclude from the final DataFrame. If None, all shots are included.
    debug : bool
        If True, prints debug information during processing.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing all shots, sorted by time.
    """
    if source_time_separation is None:
        source_time_separation = {}
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided folder path '{folder_path}' is not a valid directory.")
    shot_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not shot_files:
        raise ValueError(f"No shot files found in the folder '{folder_path}'.")
    shots_list = []
    for _file in shot_files:
        try:
            shots = read_shot_from_file(_file, gps_start=gps_start)
            if not shots.empty:
                shots_list.append(shots)
        except Exception as e:
            print(f"Error reading file {_file}: {e}")
    if not shots_list:
        raise ValueError("No valid shot files found in the specified folder.")
    merged_shots = merge_shots(shots_list)
    if bad_shots is not None:
        gd_shots = merged_shots[~merged_shots['shot'].isin(bad_shots)]
    else:
        gd_shots = merged_shots
        
    gd_shots = gd_shots.reset_index(drop=True)
        
    # print(gd_shots)
    if debug:
        print(f"Total shots read: {len(merged_shots)}")
    merged_shots = read_shots(gd_shots, gps_start=gps_start,
                              source_time_separation=source_time_separation,
                              debug=debug)
    return merged_shots


def read_shots_from_excel(excel_path: str, 
                           source_time_separation=None, 
                           bad_shots=None,debug=False) -> pd.DataFrame:
    """
    source_time_separation : dict
        The maximum time separation between shots to consider them part of the same group (in seconds).
        example: {"P1": 76.45, "S1_N": 21, ...}
    bad_shots : list or None
        A list of shot groups to exclude from the final DataFrame. If None, all shots are included.
    debug : bool
        If True, prints debug information during processing.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing all shots, sorted by time.
    """
    if source_time_separation is None:
        source_time_separation = {}
    
    merged_shots = pd.read_excel(excel_path)
    merged_shots['time'] = pd.to_datetime(
                                    merged_shots[['Year', 'Month', 'Day',
                                           'Hour', 'Minute']].assign(Second=merged_shots['Second']),
                                                                format='%Y-%m-%d %H:%M:%S.%f'
                                                                )
    merged_shots.sort_values('time', inplace=True)
    merged_shots["shot"] = range(1,len(merged_shots)+1)  # Assign shot numbers starting from 1
    merged_shots.reset_index(drop=True, inplace=True)
    
    if bad_shots is not None:
        gd_shots = merged_shots[~merged_shots['shot'].isin(bad_shots)]
    else:
        gd_shots = merged_shots
        
    gd_shots = gd_shots.reset_index(drop=True)
        
    # print(gd_shots)
    if debug:
        print(f"Total shots read: {len(merged_shots)}")
    merged_shots = read_shots(gd_shots, source_time_separation=source_time_separation,
                              debug=debug)
    return merged_shots

def plot_shot_data(df, save_path=None, show=True):
    """
    Plot shot timing data with symbols and group transitions.

    Parameters
    ----------
    df : pandas.DataFrame
        Must include 'time', 'time_from_group_lead', 'shot_group'.
    save_path : str or None
        Path to save the figure if provided.
    show : bool
        Whether to display the plot.

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    prev_group = None
    used_labels = set()

    # Counters for P and S_N groups
    p_count = 1
    s_count = 1

    for idx, row in df.iterrows():
        time = row['time']
        y = row['time_from_group_lead']
        group = row['shot_group']

        # If group changed, add a vertical dashed line
        if group != prev_group:
            ax.axvline(time, color='gray', linestyle='--', linewidth=1)

            # Determine if label should be added
            if group.startswith('P'):
                label_text = str(p_count)
                p_count += 1
            elif group.startswith('S') and group.endswith('_N'):
                label_text = str(s_count)
                s_count += 1
            else:
                label_text = None

            # Add number on top
            if label_text:
                ax.text(time, df['time_from_group_lead'].max() + 2,
                        label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')

            prev_group = group

        # Determine color
        color = 'black' if group.startswith('P') else 'red'

        # Marker and symbolic label
        if group.endswith('_N'):
            marker = '^'
            label = 'S_N'
        elif group.endswith('_S'):
            marker = 'v'
            label = 'S_S'
        else:
            marker = 'o'
            label = 'P' if group.startswith('P') else 'S'

        # Avoid duplicate legend labels
        plot_label = label if label not in used_labels else None
        if plot_label:
            used_labels.add(label)

        ax.scatter(time, y, color=color, marker=marker, label=plot_label)

    # Set axis labels and title
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Time from First Shot (s)')
    ax.set_title('Shot Timing')

    # Clean legend
    ax.legend(title='Shot Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()

    return fig, ax

def plot_shot_geometry(shot, receiver=None, save_path=None, show=True):
    """
    Plot shot geometry and timing using symbols to distinguish shot types.
    Vertical dashed lines indicate transitions between shot groups, and 
    numbered labels are added to specific groups. Optionally plots receivers.

    Parameters
    ----------
    shot : pandas.DataFrame
        DataFrame containing shot data. Must include:
        - 'x-coodinate (m)': Shot position along the line.
        - 'time_from_group_lead': Time delay from the group's first shot.
        - 'shot_group': Group identifier (e.g., 'P1', 'S2_N').
    receiver : pandas.DataFrame or None, optional
        DataFrame with receiver coordinates. Must include:
        - 'x-coordinate (m)': Receiver position along the line.
        If None, receivers are not plotted.
    save_path : str or None, optional
        Path to save the plot image. If None, the plot is not saved.
    show : bool, optional
        Whether to display the plot interactively.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib figure.
    ax : matplotlib.axes.Axes
        The axes of the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    prev_group = None           # To track when the group changes
    used_labels = set()         # To avoid repeating labels in the legend
    p_count = 1                 # Counter for P-group labels
    s_count = 1                 # Counter for S_N-group labels

    # Plot receivers if provided
    if receiver is not None:
        receiver["y"] = 0  # Set y=0 for all receivers
        ax.scatter(receiver["x-coordinate (m)"], receiver["y"],
                   color='green', marker='x', linewidth=10, label="Receivers")

    # Loop through each shot and plot
    for idx, row in shot.iterrows():
        x = row["x-coodinate (m)"]
        y = row['time_from_group_lead']
        group = row['shot_group']

        # Add vertical dashed line when a new group starts
        if group != prev_group:
            ax.axvline(x, color='gray', linestyle='--', linewidth=1)

            # Add numerical labels to certain group types
            if group.startswith('P'):
                label_text = str(p_count)
                p_count += 1
            elif group.startswith('S') and group.endswith('_N'):
                label_text = str(s_count)
                s_count += 1
            else:
                label_text = None

            # Optional: Uncomment to display number at top of line
            # if label_text:
            #     ax.text(x, shot['time_from_group_lead'].max() + 2,
            #             label_text, ha='center', va='bottom',
            #             fontsize=9, fontweight='bold')

            prev_group = group

        # Define color based on wave type
        color = 'black' if group.startswith('P') else 'red'

        # Select marker and legend label
        if group.endswith('_N'):
            marker = '^'
            label = 'S_N'
        elif group.endswith('_S'):
            marker = 'v'
            label = 'S_S'
        else:
            marker = 'o'
            label = 'P' if group.startswith('P') else 'S'

        # Add to legend only once
        plot_label = label if label not in used_labels else None
        if plot_label:
            used_labels.add(label)

        # Plot shot point
        ax.scatter(x, y, color=color, marker=marker, label=plot_label)

    # Label axes and set title
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Time from First Shot (s)')
    ax.set_title('Shots')

    # Add legend outside plot
    ax.legend(title='Shot Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save plot if path is given
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    # Show plot if requested
    if show:
        plt.show()

    return fig, ax


def merge_shots_and_geometry(shots, geometry):
    """
    Merge shot data with geometry data based on the 'Shot Location'.

    This function extracts the shot location from the 'shot_group' column in the 
    shots DataFrame, converts it to an integer, and merges it with the geometry 
    DataFrame on the 'Shot Location' column.

    Parameters
    ----------
    shots : pandas.DataFrame
        DataFrame containing shot information. Must include a 'shot_group' column
        with a format like 'P_1', 'S_3', etc.

    geometry : pandas.DataFrame
        DataFrame containing geometry data with a 'Shot Location' column to join on.

    Returns
    -------
    pandas.DataFrame
        The merged DataFrame with geometry information added to each shot.
    """
    # Extract the numeric part from the 'shot_group' column and convert to int
    shots["Shot Location"] = shots["shot_group"].apply(lambda x: x.split("_")[0][1:]).astype(int)

    # Merge the shot data with geometry data on 'Shot Location'
    shots = shots.merge(geometry, on="Shot Location", how="left")

    return shots

def separate_shot_groups(shots_geom):
    """
    Separates the input DataFrame into three shot group categories based on naming patterns.

    This function filters the shots_geom DataFrame into:
    - P shots: Rows where 'shot_group' starts with 'P'
    - S_N shots: Rows where 'shot_group' starts with 'S' and ends with '_N'
    - S_S shots: Rows where 'shot_group' starts with 'S' and ends with '_S'

    Parameters
    ----------
    shots_geom : pandas.DataFrame
        DataFrame that contains a 'shot_group' column with group labels.

    Returns
    -------
    dict
        A dictionary with keys:
        - 'P': DataFrame of P shots
        - 'S_N': DataFrame of S_N shots
        - 'S_S': DataFrame of S_S shots
    """
    # Filter rows where 'shot_group' starts with 'P'
    p_shots = shots_geom[shots_geom["shot_group"].str.startswith("P")]

    # Filter rows where 'shot_group' starts with 'S' and ends with '_N'
    s_N_shots = shots_geom[
        shots_geom["shot_group"].str.startswith("S") &
        shots_geom["shot_group"].str.endswith("_N")
    ]

    # Filter rows where 'shot_group' starts with 'S' and ends with '_S'
    s_S_shots = shots_geom[
        shots_geom["shot_group"].str.startswith("S") &
        shots_geom["shot_group"].str.endswith("_S")
    ]

    # Return a dictionary with the filtered DataFrames
    return {
        "P": p_shots,
        "S_N": s_N_shots,
        "S_S": s_S_shots
    }


def agc(DataO: np.ndarray, time: np.ndarray, agc_type = 'inst',  time_gate = 500e-3):
    """
    agc: applies automatic gain control for a given dataset.

     Usage:
         gained_data = agc(data,time,agc_type, time_gate)

     Parameters
     -----------
     data: np.ndarray
            Input seismic data
     time: np.ndarray
            Time array
     agc_type: string <class 'str'>
            Type of agc to be applied. Options: 1)'inst': instantanous AGC. 2) 'rms': root-mean-square.
            For details, please refere to: https://wiki.seg.org/wiki/Gain_applications
     time_gate: float <class 'float'>
            Time gate used for agc in sec. Defualt value 500e-3.

     Returns
     -------
     gained_data: np.ndarray
        Data after applying AGC

        AGC is python function written by Musab Al Hasani based on the book of Oz Yilmaz (https://wiki.seg.org/wiki/Gain_applications)

    """
    data = np.copy(DataO)

    # # calculate nth-percentile
    # nth_percentile = np.abs(np.percentile(data, 99))

    # clip data to the value of nth-percentile
    # data = np.clip(data, a_min=-nth_percentile, a_max = nth_percentile)


    num_traces = data.shape[1] # number of traces to apply gain on
    gain_data  = np.zeros(data.shape) # initialise the gained data 2D array

    # check what type of agc to use
    if agc_type == 'rms':
        for itrc in range(num_traces):
            gain_data[:, itrc] = rms_agc(data[:, itrc], time, time_gate)

    elif agc_type =='inst':
        for itrc in range(num_traces):
            gain_data[:, itrc] = inst_agc(data[:, itrc], time, time_gate)

    else:
        print('Wrong agc type!')

    return gain_data



def rms_agc(trace: np.ndarray, time: np.ndarray,  time_gate=200e-3)-> np.ndarray:
    """

    rms_agc: apply root-mean-square automatic gain control for a given trace.

     Usage:
         gained_trace = agc(data,time,agc_type, time_gate)

     Parameters
     -----------
     data: np.ndarray
            Input seismic trace
     time: np.ndarray
            Time array
     time_gate: float <class 'float'>
            Time gate used for agc in sec. Defualt value 200e-3 here, though there is  not a typecal value to be used.

     Returns
     -------
     gained_trace: np.ndarray
        trace after applying RMS AGC

        RMS_AGC is python function written by Musab Al Hasani based on the book of Oz Yilmaz (https://wiki.seg.org/wiki/Gain_applications)

    """

    # determine time sampling and num of samples
    dt = time[1]-time[0]
    N = len(trace)

    # determine number of time gates to use
    gates_num = int((time[-1]//time_gate)+1)

    # initialise indecies for the coners of the gate
    time_gate_1st_ind = 0
    time_gate_2nd_ind = int(time_gate/dt)


    # construct lists for begining and ends of tome gates
    start_gate_inds = [(time_gate_1st_ind + i*time_gate_2nd_ind) for i in range(gates_num)]
    end_gate_inds = [start_gate_inds[j] + time_gate_2nd_ind  for j in range(gates_num)]

    # set last gate to the end sample
    end_gate_inds[-1] = N

    # initialise middle gate time and gain function arrays
    t_rms_values   = np.zeros(gates_num+2)
    amp_rms_values = np.zeros(gates_num+2)

    # loop over every gate
    ivalue = 1
    for istart, iend in zip(start_gate_inds, end_gate_inds):
        t_rms_values[ivalue]    = 0.5*(istart + iend)
        amp_rms_values[ivalue] = np.sqrt(np.mean(np.square(trace[istart:iend])))
        ivalue += 1

    # set side values for interpolation
    t_rms_values[-1] = N
    amp_rms_values[0] = amp_rms_values[1]
    amp_rms_values[-1] = amp_rms_values[-2]

    # linear interpolation for the rms amp function for every sample N
    rms_func = np.interp(range(N), t_rms_values, amp_rms_values )

    # calculate the gained trace
    gained_trace = trace*(np.sqrt(np.mean(np.square(trace)))/rms_func)


    return gained_trace


def inst_agc(trace, time, time_gate = 500e-3 ):
    """

    rms_agc: apply instantanous automatic gain control for a given trace.

     Usage:
         gained_trace = agc(data,time,agc_type, time_gate)

     Parameters
     -----------
     data: np.ndarray
            Input seismic trace
     time: np.ndarray
            Time array
     time_gate: float <class 'float'>
            Time gate used for agc in sec. typecal values between 200-500ms.

     Returns
     -------
     gained_trace: np.ndarray
        trace after applying instansous AGC

        INST_AGC is python function written by Musab Al Hasani based on the book of Oz Yilmaz (https://wiki.seg.org/wiki/Gain_applications)

    """
    # determine time sampling and num of samples
    dt = time[1]-time[0]
    N = len(trace)

    # determine the number of sample of a given gate
    end_samples = int(time_gate/dt)

    # calculate gates number not including the last end_samples
    gates_num = N - end_samples

    # initialise gates begining and end indices
    time_gate_1st_ind = 0
    time_gate_2nd_ind = int(time_gate/dt)

    # construct lists for indices of gates corners
    start_gate_inds = [i for i in range(gates_num)]
    end_gate_inds = [start_gate_inds[j] + time_gate_2nd_ind  for j in range(gates_num)]

    #initialise gain function
    amp_inst_values = np.zeros(N)

    # loop over ever sample to calculate gain function
    ivalue = 0
    for istart, iend in zip(start_gate_inds, end_gate_inds):
        amp_inst_values[ivalue] = np.mean(np.abs(trace[istart:iend]))
        ivalue += 1
    amp_inst_values[-end_samples:] = (amp_inst_values[ivalue-1])

    # calculate gained trace
    gained_trace = trace*(np.sqrt(np.mean(np.square(trace)))/amp_inst_values)

    return gained_trace


import numpy as np
import matplotlib.pyplot as plt
import os
from obspy import UTCDateTime

def get_receiver_info(trace_id, receiver_geometry):
    """
    Given a trace id like 'SS.24311.SW.GPZ', extract receiver info.
    """
    node_id = int(trace_id.split(".")[1])
    row = receiver_geometry.loc[node_id]
    return row["gx"], row["gy"], row["gelev"]

def process_and_export_shots(
    st, shots_groups, receiver_geometry, out_folder,
    phase="P",
    delay_dict=None,
    left_seconds=0.001, right_seconds=0.31,
    apply_filter=False, freqmin=10, freqmax=80,
    normalization=True,
    plot=True,
    only_specific_shots = [],
    export_segy=True, 
    export_csv=True,
    verbose=True
):
    
    """
    Processes seismic shots by trimming traces, applying filters/normalization,
    plotting results, and exporting to SEGY and CSV formats.

    Parameters
    ----------
    st : obspy.Stream
        Stream containing seismic data.
    shots_groups : dict
        Dictionary with shot information grouped by phase.
    receiver_geometry : dict
        Dataframe containing receiver geometry information.
        with the following columns:
        - 'Receiver Number','Node ID','gx' (x-coordinate in m), 'x-coordinate (m)'
    out_folder : str
        Directory where outputs will be saved.
    phase : str, optional
        Seismic phase to process (default is 'P').
        other options could be 'S_N', 'S_S'.
    delay_dict : dict, optional
        Dictionary of delays (in seconds) for each shot number.
    left_seconds : float, optional
        Seconds before shot time to start trimming.
    right_seconds : float, optional
        Seconds after shot time to end trimming.
    apply_filter : bool, optional
        Whether to apply bandpass filter.
    freqmin : float, optional
        Minimum frequency for bandpass filter.
    freqmax : float, optional
        Maximum frequency for bandpass filter.
    normalization : bool, optional
        Whether to normalize traces by their max amplitude.
    only_specific_shots : list, optional
        If given, work only these shots.
    plot : bool, optional
        Whether to generate offset-time plots.
    export_segy : bool, optional
        Whether to export trimmed shots to SEGY format.
    export_csv : bool, optional
        Whether to export receiver metadata to CSV.
    verbose : bool, optional
        Whether to print detailed logs.
    """
    
    if delay_dict is None:
        delay_dict = {}
    
    phase_shots = shots_groups[phase]
    phase_shots["shot"] = phase_shots["shot"].astype(str)
    
    if only_specific_shots:
        only_specific_shots = [str(s) for s in only_specific_shots]
        phase_shots = phase_shots[phase_shots["shot"].isin(only_specific_shots)]
    
    
    all_receivers = []
    for i, shot in phase_shots.iterrows():
        shot_time = UTCDateTime(shot["time"])
        shot_group = shot["shot_group"]
        shot_number = int(shot["shot"])
        
        strike = shot_number % 6   # Assuming strike is based on shot number
        if strike == 0:
            strike = 6  # Adjust to match your strike numbering logic
        
        if verbose:
            print(f"\nProcessing shot {i + 1}/{len(shots_groups[phase])}: {shot['shot_group']} - {strike} | Shot time: {shot_time}")


        if str(shot["shot"]) in list(delay_dict.keys()):
            delay = delay_dict[str(shot["shot"])]
        else:
            delay = 0.0  # Default delay if not found in the dictionary
        if verbose:
            print(f"\tApplying delay: {delay} seconds")

        t_start = shot_time + delay - left_seconds 
        t_end = shot_time + delay + right_seconds
        st_shot = st.copy().trim(starttime=t_start, endtime=t_end)

        sx = int(shot["sx"])
        sy = int(shot["sy"])
        selev = int(shot["selev"])

        npts = st_shot[0].stats.npts
        dt = st_shot[0].stats.delta
        time_vec = np.arange(npts) * dt

        offsets = []
        receivers = []
        receivers_names = []
        sources = []
        traces = []

        for tr in st_shot:
            gx, gy, gelev = get_receiver_info(tr.id, receiver_geometry)
            offset = gx - sx
            offsets.append(offset)

            tr.data = tr.data.astype(np.float32)
            # data = tr.data

            if apply_filter:
                tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
                tr.data = tr.data.astype(np.float32)  # Ensure proper dtype
                data = tr.data

            
            if normalization:
                data = tr.data
                max_val = np.max(np.abs(data))
                if max_val > 0:
                    data = data / max_val

            traces.append(data)
            receivers.append(gx)
            receivers_names.append(tr.stats.station)
            sources.append(sx)

            tr.stats.distance = offset
            tr.stats.segy = {
                'source_coordinate_x': sx,
                'receiver_coordinate_x': gx,
                'source_elevation': selev,
                'receiver_elevation': gelev,
                'coordinate_units': 2,
                'ensemble_number': int(shot_group[1:]),
                'trace_number_within_ensemble': tr.stats.trace_number if "trace_number" in tr.stats else 1,
            }
            # if verbose:
            #     print(f"\tSource: {sx}, Receiver: {gx}")

        offsets = np.array(offsets)
        traces = np.array(traces)
        sources = np.array(sources)
        sort_idx = np.argsort(offsets)
        offsets = offsets[sort_idx]
        traces = traces[sort_idx]
        receivers = np.array(receivers)[sort_idx]
        receivers_names = np.array(receivers_names)[sort_idx]
        
        
        receivers_df = pd.DataFrame({
                        'source_distance (m)': sources,
                        'receiver_distance (m)': receivers,
                        'station': receivers_names
                    })
        
        all_receivers.append(receivers_df)



        times = np.arange(npts) * dt - left_seconds

        
        if plot:
            plt.figure(figsize=(10, 6))
            # Good scale factor: small fraction of max amplitude
            scale = 0.5 * np.max(np.abs(traces))
            # scale=1
            for j, trace in enumerate(traces):
                # plt.plot(trace / scale + offsets[j], times, color='black', linewidth=0.5)
                trace_scaled = trace / scale
                # plt.plot(trace_scaled  + offsets[j], times, color='black', linewidth=0.5)
                plt.plot(trace_scaled + offsets[j], times, color='black', 
                         linewidth=0.5)
                
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.3, label='Source')

            plt.gca().invert_yaxis()
            plt.xlabel("Offset (m)")
            plt.ylabel("Time since shot (s)")
            plt.title(f"Shot {shot_number} | Group {shot_group} | Strike {strike} | {shot_time} UTC | Delay {delay} s\n"+\
                     f"Source: {sx} m")
            
            # Major and minor grid setup
            ax = plt.gca()
            # Set major ticks every 2 x units
            ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
            ax.grid(True, which='minor', linestyle=':', linewidth=0.8,color='lightgray')

            # Enable minor ticks
            ax.minorticks_on()

            # Set minor ticks every 2 meters on x-axis
            xmin, xmax = ax.get_xlim()
            minor_xticks = np.arange(np.floor(xmin), np.ceil(xmax), 2)
            ax.set_xticks(minor_xticks, minor=True)
            
            plt.grid(True)
            plt.tight_layout()

        if verbose:
            print(f"\tWriting shot group {shot_group}, strike {strike} with {len(st_shot)} traces")

        segy_out_folder = os.path.join(out_folder, "processing")
        os.makedirs(segy_out_folder, exist_ok=True)

        if export_segy:
            segy_out_path = os.path.join(segy_out_folder, f"{shot_group}_strike_{strike}.segy")
            st_shot.write(segy_out_path, format="SEGY")

        if plot:
            png_out_path = os.path.join(segy_out_folder, f"{shot_group}_strike_{strike}.png")
            plt.savefig(png_out_path, dpi=300)
            plt.close()
            
    if len(all_receivers) == 1:
        all_receivers_df = all_receivers[0]
    else:
        # Concatenate all receivers DataFrames into a single DataFrame
        all_receivers_df = pd.concat(all_receivers, ignore_index=True)
    
    if export_csv:
        receivers_out_path = os.path.join(out_folder, "receivers.csv")
        all_receivers_df.to_csv(receivers_out_path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")


if __name__ == "__main__":
    
    # x= read_shots_from_excel("/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/Nadine/ShotTimes_PackeryFlats.xlsx")
    # print(x)
    # exit()
    
    # Example usage
    solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data/test/raw_shots"
    #raw_shots_paths = glob.glob(solo_folder+"/*.csv")
    
    source_path_1  = os.path.join(solo_folder, "TB_INT00147.csv")
    source_path_2  = os.path.join(solo_folder, "TB_INT00148.csv")
    
    
    
    source_time_separation = { #P Wave
                              "P1": 80,
                              "P2":41,
                              "P3":31,
                              "P4":43,
                              "P5":47,
                              "P6":27,
                              "P7":23,
                              "P8":31,
                              "P9":34,
                              "P10":34,
                              "P11":24,
                              "P12":25,
                              "P13":27,
                              "P14":63,
                              "P15":52,
                              "P16":31,
                              #S Wave
                              "S1_N":22,
                              "S1_S":22,
                              "S2_N":20,
                              "S2_S":25,
                              "S3_N":32,
                              "S3_S":39,
                              "S4_N":36,
                              "S4_S":31,
                              "S5_N":32,
                              "S5_S":22,
                              "S6_N":27,
                              "S6_S":21,
                              "S7_N": 22,
                              "S7_S": 22,
                              "S8_N": 22,
                              "S8_S": 30,
                              "S9_N": 22,
                              "S9_S": 25,
                              "S10_N": 31,
                              "S10_S": 36,
                              "S11_N": 24,
                              "S11_S": 23,
                              "S12_N": 23,
                              "S12_S": 28,
                              "S13_N": 37,
                              "S13_S": 25,
                              "S14_N": 30,
                              "S14_S": 21,
                              "S15_N": 23,
                              "S15_S": 23,
                              "S16_N": 26,
                              "S16_S": 21,
                            }
    bad_shots=[181,182]
    
    source_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/Nadine/ShotTimes_PackeryFlats.xlsx"
    output_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/shots_labeled.csv"
    shots = read_shots_from_excel(source_path,
                                  source_time_separation=source_time_separation,
                                  bad_shots=bad_shots,
                                  debug=True)
    shots.to_csv(output_path, index=False, 
                 date_format="%Y-%m-%d %H:%M:%S.%f")
    # shots = read_shots_from_folder(solo_folder, 
    #                     #    source_time_separation=source_time_separation, 
    #                        bad_shots=[181,182],
    #                        debug=True)
    
    out_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/shots_plot.png"
    plot_shot_data(shots, save_path=out_path, show=True)
    # print(shots)
    # last_group_name = shots.iloc[-1].shot_group
    # last_group = shots[shots["shot_group"] ==  last_group_name]
    # print(last_group.iloc[0:10])
    # print(len(shots))
    exit()
    # print(p_source_path)
    # exit()
    

    # shots = read_shots(p_source_path,source_time_separation=source_time_separation,
    #                    debug=True)
    # max_group = shots["shot_group"].max()
    # last_group = shots[shots["shot_group"] == max_group]
    # print(last_group.iloc[0:10])
    # print(shots.iloc[10:30])
    # shots = read_shots(s_source_path,source_time_separation=35.4)
    # print(shots.head(30))
    # source_time_separation = {
                                
                           
    # print(shots.head(30))
    #                         }
    
    # shots = read_shots(p_source_path,source_time_separation=70,debug=True)
    # # print(shots.head(30))