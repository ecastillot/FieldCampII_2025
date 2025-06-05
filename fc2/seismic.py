import pandas as pd
import datetime

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
        df['elevation_m'] = df[elev_col_in_km]  # Already in meters
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

    return pd.DataFrame(shots)

