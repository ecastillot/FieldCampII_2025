import pandas as pd
import datetime

import glob
import os
import obsplus
from obspy import read, Stream, UTCDateTime
from obspy import read_inventory

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

def read_shots(filepath, gps_start=None):
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

        lat_str = coord_line.split("Latitude:")[1].strip()
        lat = float(lat_str.split(" ")[0])
        lon_str = coord_line.split("Longitude:")[1].split("Latitude:")[0].strip()
        lon = float(lon_str.split(" ")[0])

        # Append the shot information
        shots.append({
            "shot": i + 1,
            "year": timestamp.year,
            "month": timestamp.month,
            "day": timestamp.day,
            "hour": timestamp.hour,
            "minute": timestamp.minute,
            "second": round(timestamp.second + timestamp.microsecond / 1e6, 2),
            "latitude": round(lat, 2),
            "longitude": round(abs(lon), 2)  # Convert W to positive if required
        })

    return pd.DataFrame(shots)

