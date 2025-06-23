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
        timestamp = gps_start + datetime.timedelta(seconds=total_seconds)

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
            
            print(f"Group 'last_group' with {len(bad_shots)} shots")
            
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


if __name__ == "__main__":
    # Example usage
    solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data/test/raw_shots"
    #raw_shots_paths = glob.glob(solo_folder+"/*.csv")
    
    source_path_1  = os.path.join(solo_folder, "TB_INT00147.csv")
    source_path_2  = os.path.join(solo_folder, "TB_INT00148.csv")
    
    
    
    source_time_separation = { #P Wave
                              "P1": 76.45,
                              "P2":40,
                              "P3":30,
                              "P4":42,
                              "P5":46,
                              "P6":26,
                              "P7":22,
                              "P8":30,
                              "P9":33,
                              "P10":33,
                              "P11":23,
                              "P12":24,
                              "P13":26,
                              "P14":62,
                              "P15":51,
                              "P16":30,
                              #S Wave
                              "S1_N":21,
                              "S1_S":21,
                              "S2_N":19,
                              "S2_S":24,
                              "S3_N":31,
                              "S3_S":38,
                              "S4_N":35,
                              "S4_S":30,
                              "S5_N":31,
                              "S5_S":21,
                              "S6_N":26,
                              "S6_S":20,
                              "S7_N": 21, 
                              "S7_S": 21,
                              "S8_N": 21,
                              "S8_S": 29,
                              "S9_N": 21,
                              "S9_S": 24,
                              "S10_N": 30,
                              "S10_S": 35,
                              "S11_N": 23,
                              "S11_S": 22,
                              "S12_N": 22,
                              "S12_S": 27,
                              "S13_N": 36,
                              "S13_S": 24,
                              "S14_N": 29,
                              "S14_S": 20,
                              "S15_N": 22,
                              "S15_S": 22,
                              "S16_N": 25,
                              "S16_S": 20,
                            }
    
    shots = read_shots_from_folder(solo_folder, 
                        #    source_time_separation=source_time_separation, 
                           bad_shots=[181,182],
                           debug=True)
    
    out_path = "/groups/igonin/ecastillo/FieldCampII_2025/out/shots_plot.png"
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