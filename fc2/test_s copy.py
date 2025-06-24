import pandas as pd
import os
from obspy import UTCDateTime
from seismic import (merge_shots_and_geometry,
           separate_shot_groups,plot_shot_data,
           plot_shot_geometry,read_waveforms)

receiver_geometry_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/receiver_geometry.csv"
shots_labeled_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/shots_labeled.csv"
source_geometry_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/seismic/June_14/source_geometry.csv"

out_folder = "/groups/igonin/ecastillo/FieldCampII_2025/out"

receiver_geometry = pd.read_csv(receiver_geometry_path)
shots_labeled = pd.read_csv(shots_labeled_path)
source_geometry = pd.read_csv(source_geometry_path)

shots_with_geom = merge_shots_and_geometry(shots=shots_labeled,
                         geometry=source_geometry,)

shots_groups = separate_shot_groups(shots_geom=shots_with_geom)

print(shots_groups.keys())

for group_name, shots in shots_groups.items():
    print(f"Group: {group_name}, Number of groups: {len(shots.groupby('shot_group'))}, Shots per group: {shots.groupby('shot_group').size().to_dict()}")
    plot_shot_data(shots,
                   save_path=os.path.join(out_folder, f"shots_{group_name}_data.png"),)
    plot_shot_geometry(shots, receiver=receiver_geometry,
                       save_path=os.path.join(out_folder, f"shots_{group_name}_geometry.png"))

## put this code in notebook too (for colab)
# !wget -O smart_solo_data.zip "https://www.dropbox.com/scl/fo/tkeerdr54zz6r3psn61y1/ABIXTjSj-xDSgWm9sH-jjks?rlkey=jyw3s1cfk8mm3tcu5vuy8u8rk&e=1&dl=1"
# !unzip  smart_solo_data.zip -d  smart_solo_data

# smart_solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data_bck/smart_solo_data"
# st = read_waveforms(folder_path=smart_solo_folder,
#                     station="4530243*",
#                     component="Z",
#                     starttime=UTCDateTime("2025-06-14T13:30:00"),
#                     endtime=UTCDateTime("2025-06-14T15:30:00"))
# print(st)
