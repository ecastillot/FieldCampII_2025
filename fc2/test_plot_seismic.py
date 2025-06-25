import pandas as pd
import os
import numpy as np
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from seismic import (
           separate_shot_groups,plot_shot_data,
           plot_shot_geometry,read_waveforms,
           get_receiver_data,get_shots_data,
           process_and_export_shots)

receiver_geometry_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/receiver_geometry.csv"
shots_labeled_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/shots_labeled.csv"
source_geometry_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/source_geometry.csv"

out_folder = "/groups/igonin/ecastillo/FieldCampII_2025/out"

shots_labeled = pd.read_csv(shots_labeled_path)
source_geometry = pd.read_csv(source_geometry_path)

receiver_geometry = pd.read_csv(receiver_geometry_path)
receiver_geometry = get_receiver_data(receiver_geometry)


shots_geometry = get_shots_data(shots_labeled, source_geometry)

shots_groups = separate_shot_groups(shots_geom=shots_geometry)


for group_name, shots in shots_groups.items():
    print(f"Group: {group_name}, Number of groups: {len(shots.groupby('shot_group'))}, Shots per group: {shots.groupby('shot_group').size().to_dict()}")
    plot_shot_data(shots,
                   save_path=os.path.join(out_folder, f"shots_{group_name}_data.png"),)
    plot_shot_geometry(shots, receiver=receiver_geometry,
                       save_path=os.path.join(out_folder, f"shots_{group_name}_geometry.png"))

## Downloading data
!wget -O smart_solo_data.zip "https://www.dropbox.com/scl/fo/tkeerdr54zz6r3psn61y1/ABIXTjSj-xDSgWm9sH-jjks?rlkey=jyw3s1cfk8mm3tcu5vuy8u8rk&e=1&dl=1"
!unzip  smart_solo_data.zip -d  smart_solo_data

smart_solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data_bck/smart_solo_data"
starttime = UTCDateTime(shots_groups["P"]["time"].min()) - 5  # buffer in seconds
endtime = UTCDateTime(shots_groups["P"]["time"].max()) + 10   # buffer in seconds

st = read_waveforms(folder_path=smart_solo_folder,
                    component="Z",
                    starttime=starttime,
                    endtime=endtime)

processing = {}

# Define processing parameters
processing["phase"] = "P" #phase to process: "P", "S_N", "S_S". If you use S wave be sure to use "E" component when you read the waveforms

## Define padding or delay centered around the shot time
processing["left_seconds"] = 0.001
processing["right_seconds"] = 0.6
processing["delay_dict"] = {"2":0.08,  # example delay for shot number 2
                            "37": 0.2  # example delay for shot number 37
                            }  

# Define filtering and normalization parameters
processing["apply_filter"] = True
processing["freqmin"] = 20
processing["freqmax"] = 60
processing["normalization"] = True

# Define output options
processing["plot"] = True
processing["only_specific_shots"] = []  # if you only want to work in some specific
                                        # shots, use a list like this: [1,34,53]  # for shots 1 , 34 and 57
processing["export_segy"] = False
processing["verbose"] = True

process_and_export_shots(
    st=st,
    shots_groups=shots_groups,
    receiver_geometry=receiver_geometry,
    out_folder=out_folder,
    
    phase="P",
    
    left_seconds=0.001,
    right_seconds=0.6,
    delay_dict = {"37":0.2},
    apply_filter=True,
    freqmin=20,
    freqmax=60,
    
    normalization=True,
    plot=True,
    only_specific_shots=[1],
    export_segy=False,
    verbose=True
)