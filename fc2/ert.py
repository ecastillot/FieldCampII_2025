# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2025-05-20 19:10:33
#  * @modify date 2025-05-20 19:10:33
#  * @desc [description]
#  */
# Based on https://github.com/dakodonnell/ERTanalytics/tree/master

import pandas as pd
import numpy as np
from datetime import datetime, time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def parse_stg(filepath, join_date_time=True):
    """
    Read a .stg file exported from SuperSting and return a pandas DataFrame.
    
    Parameters:
        filepath (str): Path to the .stg file.
        join_date_time (bool): If True, combine date and time into a single datetime object in 'date_time' column.
    
    Returns:
        pd.DataFrame: Parsed .stg data.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Skip metadata lines (first three)
    data_lines = lines[3:]

    records = []
    for line in data_lines:
        # Remove newline and split by comma
        parts = [p.strip() for p in line.strip().split(',')]
        if not parts or len(parts) < 22:
            continue  # skip incomplete lines

        # Parse date and time
        raw_date = parts[2]
        raw_time = parts[3]

        date_obj = datetime.strptime(raw_date, "%Y%m%d").date()
        time_obj = datetime.strptime(raw_time, "%H:%M:%S").time()
        date_time_obj = datetime.combine(date_obj, time_obj)

        record = {
            "record_num": int(parts[0]),
            "user": parts[1],
            "date": date_obj if not join_date_time else None,
            "time": time_obj if not join_date_time else None,
            "date_time": date_time_obj if join_date_time else None,
            "V/I": float(parts[4]),
            "error_percent_tenths": int(parts[5]),
            "current_mA": int(parts[6]),
            "resistivity": float(parts[7]),
            "command_id": parts[8],
            "Ax": float(parts[9]),
            "Ay": float(parts[10]),
            "Az": float(parts[11]),
            "Bx": float(parts[12]),
            "By": float(parts[13]),
            "Bz": float(parts[14]),
            "Mx": float(parts[15]),
            "My": float(parts[16]),
            "Mz": float(parts[17]),
            "Nx": float(parts[18]),
            "Ny": float(parts[19]),
            "Nz": float(parts[20]),
        }

        # IP data starts at index 22
        if len(parts) > 21 and parts[21] == "IP:":
            record.update({
                "ip_time_slot_ms": int(parts[22]),
                "ip_time_constant": int(parts[23]),
                "ip_slot1": float(parts[24]),
                "ip_slot2": float(parts[25]),
                "ip_slot3": float(parts[26]),
                "ip_slot4": float(parts[27]),
                "ip_slot5": float(parts[28]),
                "ip_slot6": float(parts[29]),
                "ip_total": float(parts[30]),
            })
        
        records.append(record)

    df = pd.DataFrame(records)

    # Drop 'date' and 'time' if join_date_time is True
    if join_date_time:
        df.drop(columns=["date", "time"], inplace=True)

    return df

def parse_crs(file_path, join_date_time=True):
    """
    Reads a .crs contact resistance file.

    Parameters
    ----------
    file_path : str or Path
        Path to the .crs file.
    join_date_time : bool, default=True
        If True, combines date and time into a single datetime column 'date_time'.
        Otherwise, returns separate 'date' (datetime.date) and 'time' (datetime.time) columns.

    Returns
    -------
    pd.DataFrame
        Parsed contact resistance data.
    """
    # Read the raw file lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Replace 'Time of reading' with 'date,time' in the header line (assumed to be line 5 = index 4)
    header_line_index = 4  # If your header line is on line 6 (0-based index 5)
    # Actually, your example says 5 header lines to skip, so the 6th line is the header (index 5)
    # Let's check the exact header line index:
    # If the first 5 lines are header lines, then line 6 (index 5) is the actual CSV header.
    header_line = lines[header_line_index]
    new_header_line = header_line.replace("Time of reading", "date, time")
    lines[header_line_index] = new_header_line.replace(" ", "").strip() + "\n"

    # Write to a temporary buffer or directly read from modified lines
    from io import StringIO
    modified_csv = StringIO(''.join(lines))
    

    # Skip the 5 header lines and read the CSV part
    df = pd.read_csv(modified_csv, skiprows=header_line_index)

    # Parse 'date' and 'time' columns
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d').dt.date
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time

    if join_date_time:
        df['date_time'] = df.apply(lambda row: datetime.combine(row['date'], row['time']), axis=1)
        df.drop(['date', 'time'], axis=1, inplace=True)
        cols = list(df.columns)
        cols = [cols[-1]] + cols[:-1]
        df = df[cols]

    return df

def parse_cmd(file_path,debug=False):
    """
    Parses a structured geophysical input file into header metadata and tabular data for geometry and commands.
    
    Sections:
        :header   -> key = value pairs
        :geometry -> 4-column comma-separated numerical data
        :commands -> 12-column comma-separated numerical data

    Parameters
    ----------
    file_path : str
        Path to the input file.
    debug : bool
        If True, prints debug information.
        Default is False.

    Returns
    -------
    dict
        A dictionary with:
            'header'   : dict of key-value pairs (str -> str),
            'geometry' : pd.DataFrame with 4 columns,
            'commands' : pd.DataFrame with 12 columns
    """
    data = {
        'header': {},
        'geometry': [],
        'commands': []
    }

    section = None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Switch section
            if line.startswith(':'):
                section = line[1:].lower()
                continue

            if section == 'header' and '=' in line:
                try:
                    key, value = [item.strip() for item in line.split('=', 1)]
                    data['header'][key] = value
                except ValueError:
                    print(f"Warning: Malformed header line skipped: {line}")
                continue

            try:
                values = [float(x) for x in line.split(',')]

                if section == 'geometry':
                    if len(values) != 4:
                        raise ValueError
                    data['geometry'].append(values)

                elif section == 'commands':
                    if len(values) != 12:
                        raise ValueError
                    data['commands'].append(values)

            except ValueError:
                if debug:
                    print(f"Debug: Failed to parse line in section '{section}': {line}")
                

    # Convert lists to DataFrames with default column names
    geometry_df = pd.DataFrame(data['geometry'], columns=['x1', 'x2', 'x3', 'x4'])
    commands_df = pd.DataFrame(data['commands'], columns=[
        'A', 'B', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'channels'
    ])

    return {
        'header': data['header'],
        'geometry': geometry_df,
        'commands': commands_df
    }

def calculate_geofactor(array_type, electrodes):
    """
    Calculates the geometric factor (K) for 1D resistivity survey electrode configurations.

    Parameters
    ----------
    array_type : str
        Type of electrode array. Supported: 'wenner', 'dipoledipole', 'schlumberger'.
    electrodes : np.ndarray
        3D numpy array of shape (N_configs, N_positions, 4) containing electrode
        configurations (C1, C2, P1, P2) as returned by findStations.

    Returns
    -------
    K : np.ndarray
        2D numpy array of shape (N_configs, N_positions) containing geometric factors
        for each valid configuration. Invalid (NaN) configurations are assigned NaN.
    """
    # Initialize output array with same shape as electrodes' first two dimensions
    K = np.full((electrodes.shape[0], electrodes.shape[1]), np.nan)

    for i in range(electrodes.shape[0]):  # Loop over levels (e.g., spacing levels)
        for j in range(electrodes.shape[1]):  # Loop over positions
            # Extract electrode positions (C1, C2, P1, P2)
            c1, c2, p1, p2 = electrodes[i, j, :]

            # Skip if configuration contains NaNs
            if np.any(np.isnan([c1, c2, p1, p2])):
                continue

            if array_type == "wenner":
                # Wenner array: K = 2 * pi * a, where a is the electrode spacing
                # For Wenner, C1, C2, P1, P2 are equally spaced, so a = P1 - C2
                a = p1 - c2
                K[i, j] = 2 * np.pi * a

            elif array_type == "dipoledipole":
                # Dipole-Dipole: K = pi * n * (n + 1) * (n + 2) * a
                # where a is dipole length (C1 - C2), n is number of dipole lengths between dipoles
                a = c1 - c2  # Dipole length (assuming C1 > C2)
                n = (p1 - c1) / a  # Number of dipole lengths between C1 and P1
                K[i, j] = np.pi * n * (n + 1) * (n + 2) * a

            elif array_type == "schlumberger":
                # Schlumberger: K = pi * (L^2 - l^2) / (2 * l)
                # where L = (C2 - C1)/2 (half distance between current electrodes)
                # and l = (P2 - P1)/2 (half distance between potential electrodes)
                L = (c2 - c1) / 2
                l = (p2 - p1) / 2
                K[i, j] = np.pi * (L**2 - l**2) / (2 * l)

            else:
                raise ValueError(
                    "Invalid array_type. Choose from 'wenner', 'dipoledipole', 'schlumberger'."
                )

    return K

def findStations(array_type, a, num_electrodes, x0):
    """
    Generates electrode configurations for 1D resistivity surveys based on the array type.

    Parameters
    ----------
    array_type : str
        Type of electrode array. Supported: 'wenner', 'dipoledipole', 'schlumberger'.
    a : float
        Unit electrode spacing.
    num_electrodes : int
        Total number of electrodes.
    x0 : float
        Starting x-coordinate for the first electrode.

    Returns
    -------
    electrodes : np.ndarray
        3D numpy array of shape (N_configs, N_positions, 4), where each 4-tuple is a
        station configuration (C1, C2, P1, P2) padded with NaNs if needed.
    """

    # Initialize the final array to store configurations
    electrodes = np.empty((0, num_electrodes - 3, 4))

    # Define a padding array filled with NaNs for alignment
    nan_arr = np.array([[[np.nan, np.nan, np.nan, np.nan]]])

    if array_type == "dipoledipole":
        n_max = int(np.floor(num_electrodes - 3))

        for i in range(1, n_max + 1):  # i = spacing level
            configs = []
            for n in range(1, num_electrodes - 2 - i + 1): # n = electrode index
                c2 = x0 + (n-1) * a
                c1 = c2 + a
                p1 = c1 + i * a
                p2 = p1 + a
                station = [c1, c2, p1, p2]
                # print(f"level {i} node {n}|\t", np.mean(station), station)
                configs.append(station)

            # Convert to numpy array
            config_array = np.array(configs).reshape(-1, 4)

            # Pad with NaNs to reach the correct second dimension
            padding_needed = (num_electrodes - 3) - config_array.shape[0]
            if padding_needed > 0:
                padding = np.repeat(nan_arr, padding_needed, axis=1)
                config_array = np.vstack((config_array, padding.reshape(-1, 4)))

            # Add the configurations for this n to the final electrodes array
            electrodes = np.vstack((electrodes, config_array[np.newaxis, :, :]))

    elif array_type == "wenner":
        n_max = int(np.floor((num_electrodes - 1)/3))
        for i in range(1, n_max + 1):
            configs = []  # Temporary list to store valid configurations for this n
            
            # Loop through each possible position of the dipole pair
            for n in range(1, num_electrodes - 3 * i + 1):
                c1 = x0 + a * (n - 1)
                c2 = c1 + a * i
                p1 = c2 + a * i
                p2 = p1 + a * i
                station = [c1, c2, p1, p2]
                # print(f"level {i} node {n}|\t", np.mean(station), station)
                configs.append(station)
            # Convert to numpy array
            config_array = np.array(configs).reshape(-1, 4)

            # Pad with NaNs to reach the correct second dimension
            padding_needed = (num_electrodes - 3) - config_array.shape[0]
            if padding_needed > 0:
                padding = np.repeat(nan_arr, padding_needed, axis=1)
                config_array = np.vstack((config_array, padding.reshape(-1, 4)))

            # Add the configurations for this n to the final electrodes array
            electrodes = np.vstack((electrodes, config_array[np.newaxis, :, :]))

    elif array_type == "schlumberger":
        # Maximum half-current spacing
        n_max = int(np.floor((num_electrodes - 1) / 2))  # conservative estimate

        for i in range(1, n_max + 1):  # i controls current electrode pair spacing
            configs = []

            for n in range(1, num_electrodes - 2 * i):
                # Inner pair (potential electrodes) are 1 unit apart
                c1 = x0 + (n - 1) * a
                c2 = x0 + (n - 1 + 2 * i) * a  # outer electrodes further apart
                mid = (c1 + c2) / 2
                p1 = mid - a / 2
                p2 = mid + a / 2
                station = [c1, c2, p1, p2]
                # print(f"level {i} node {n}|\t", np.mean(station), station)
                configs.append(station)

            # Convert to numpy array
            config_array = np.array(configs).reshape(-1, 4)

            # Pad with NaNs to reach the correct second dimension
            padding_needed = (num_electrodes - 3) - config_array.shape[0]
            if padding_needed > 0:
                padding = np.repeat(nan_arr, padding_needed, axis=1)
                config_array = np.vstack((config_array, padding.reshape(-1, 4)))

            # Add the configurations for this i to the final electrodes array
            electrodes = np.vstack((electrodes, config_array[np.newaxis, :, :]))

    else:
        raise ValueError(
            "Invalid array_type. Choose from 'wenner', 'dipoledipole', 'schlumberger'."
        )

    return electrodes

def get_pseudosection_coords(electrodes, array_type, a, use_pseudodepth=False):
    """
    Compute x (midpoint) and z (pseudo-depth or level) for each configuration.
    
    Parameters
    ----------
    electrodes : np.ndarray
        Array of shape (n_configs, n_positions, 4) with electrode positions [C1, C2, P1, P2].
    array_type : str
        Electrode array type. One of ['wenner', 'dipoledipole', 'schlumberger'].
    a : float
        Electrode spacing.
    use_pseudodepth : bool, optional
        If True, compute pseudo-depths. If False, assign level index. Default is False.
    
    Returns
    -------
    x_coords : np.ndarray
        Midpoint x-coordinates.
    z_coords : np.ndarray
        Pseudo-depths or levels (depending on `use_pseudodepth`).
    """
    x_coords = []
    z_coords = []

    for config_idx in range(electrodes.shape[0]):
        for pos_idx in range(electrodes.shape[1]):
            config = electrodes[config_idx, pos_idx]
            if np.any(np.isnan(config)):
                continue
            c1, c2, p1, p2 = config
            midpoint = np.mean([c1, c2, p1, p2])
            x_coords.append(midpoint)

            if use_pseudodepth:
                if array_type == 'wenner':
                    n = (p2 - c1) / (3 * a)
                    pseudo_depth = a * n / 2
                elif array_type == 'dipoledipole':
                    n = (p1 - c2) / a - 1
                    pseudo_depth = a * n / 2
                elif array_type == 'schlumberger':
                    L = c2 - c1
                    pseudo_depth = L / 4
                else:
                    raise ValueError(f"Unsupported array type: {array_type}")
                z_coords.append(pseudo_depth)
            else:
                # print(array_type)
                # Just use level based on configuration index (could also use pos_idx if preferred)
                z_coords.append(config_idx + 1)

    return np.array(x_coords), np.array(z_coords)

def plot_single_pseudosection(array_type, a=1.0, num_electrodes=8,
                              x0=0, ax=None, show_electrodes=True,
                               use_pseudodepth=False):
    data = findStations(array_type, a, num_electrodes, x0)
    x, z = get_pseudosection_coords(data, array_type, a,use_pseudodepth)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    
    ax.scatter(x, z, marker='s', color='black', label='Midpoints')

    if show_electrodes:
        electrode_positions = np.arange(x0, x0 + a * num_electrodes, a)
        ax.scatter(electrode_positions, np.zeros_like(electrode_positions), color='red', marker='^', label='Electrodes')

    ax.set_title(f"{array_type.capitalize()} Pseudosection")
    
    if use_pseudodepth:
        ax.set_ylabel("Pseudo-depth (m)")
    else:
        ax.set_ylabel("level")
        
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend()

    return ax

def plot_all_pseudosections(a=1.0, num_electrodes=8, x0=0.0, 
                            show_electrodes=True, use_pseudodepth=False,
                            save_path=None):
    """
    Plots pseudosections for Wenner, Dipole-Dipole, and Schlumberger arrays.

    Parameters
    ----------
    a : float
        Electrode spacing.
    num_electrodes : int
        Number of electrodes.
    x0 : float
        Starting x-coordinate.
    show_electrodes : bool
        Whether to display electrode locations at the top.
    save_path : str or None
        If provided, saves the figure to this path.

    Returns
    -------
    axs : list of matplotlib.axes._subplots.AxesSubplot
        List of Axes objects for each subplot.
    """
    array_types = ['wenner', 'dipoledipole', 'schlumberger']
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 9), sharex=True)

    for ax, array_type in zip(axs, array_types):
        try:
            plot_single_pseudosection(array_type, a, num_electrodes, x0, ax=ax, 
                                      show_electrodes=show_electrodes,
                                      use_pseudodepth=use_pseudodepth)
        except:
            print(f"Warning: error for {array_type}")

    axs[-1].set_xlabel("Distance (m)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return axs



