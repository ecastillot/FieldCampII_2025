import pandas as pd
import os
import numpy as np
from obspy import UTCDateTime
from seismic import (merge_shots_and_geometry,
           separate_shot_groups,plot_shot_data,
           plot_shot_geometry,read_waveforms,
           get_receiver_data,get_shots_data)

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

# # print(shots_groups.keys())

# for group_name, shots in shots_groups.items():
#     print(f"Group: {group_name}, Number of groups: {len(shots.groupby('shot_group'))}, Shots per group: {shots.groupby('shot_group').size().to_dict()}")
#     plot_shot_data(shots,
#                    save_path=os.path.join(out_folder, f"shots_{group_name}_data.png"),)
#     plot_shot_geometry(shots, receiver=receiver_geometry,
#                        save_path=os.path.join(out_folder, f"shots_{group_name}_geometry.png"))

## put this code in notebook too (for colab)
# !wget -O smart_solo_data.zip "https://www.dropbox.com/scl/fo/tkeerdr54zz6r3psn61y1/ABIXTjSj-xDSgWm9sH-jjks?rlkey=jyw3s1cfk8mm3tcu5vuy8u8rk&e=1&dl=1"
# !unzip  smart_solo_data.zip -d  smart_solo_data

smart_solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data_bck/smart_solo_data"
starttime = UTCDateTime(shots_groups["P"]["time"].min()) - 5  # buffer in seconds
endtime = UTCDateTime(shots_groups["P"]["time"].max()) + 10   # buffer in seconds

print(starttime, endtime)
st = read_waveforms(folder_path=smart_solo_folder,
                    component="Z",
                    starttime=starttime,
                    endtime=endtime)
# print(st)
# print(shots_groups["P"].head())
# print(receiver_geometry.head())

def get_receiver_info(trace_id, receiver_geometry):
    """
    Given a trace id like 'SS.24311.SW.GPZ', extract receiver info.
    """
    node_id = int(trace_id.split(".")[1])
    row = receiver_geometry.loc[node_id]
    return row["gx"], row["gy"], row["gelev"]

for i, shot in shots_groups["P"].iterrows():
    shot_time = UTCDateTime(shot["time"])
    shot_group = shot["shot_group"]
    shot_number = int(shot["shot"])  # This is the strike number (1–6, etc.)

    # Time window for trace cutting
    t_start = shot_time - 1
    t_end = shot_time + 5

    # Trim trace data
    st_shot = st.copy().trim(starttime=t_start, endtime=t_end)

    # Source geometry
    sx = int(shot["sx"])
    sy = int(shot["sy"])
    selev = int(shot["selev"])

    for j, tr in enumerate(st_shot):
        gx, gy, gelev = get_receiver_info(tr.id, receiver_geometry)
        tr.data = tr.data.astype(np.float32)
        tr.stats.su = {
            "sx": sx,
            "sy": sy,
            "selev": selev,
            "gx": gx,
            "gy": gy,
            "gelev": gelev,
            "scalel": -100,
            "scalco": -100,
            "ep": shot_number,           # Strike number
            "fldr": int(shot_group[1:]), # Group number (e.g., 1 for P1)
            "tracf": j + 1,
        }
        print(f"\tSource: ({sx}, {sy}, {selev}), Receiver: {tr.stats.station} ({gx}, {gy}, {gelev})")
    print(f"Writing shot group {shot_group}, strike {shot_number} with {len(st_shot)} traces")
    
    su_out_folder = os.path.join(out_folder,"su_output")
    su_out_path = os.path.join(su_out_folder, f"{shot_group}_strike_{shot_number}.su")
    os.makedirs(su_out_folder, exist_ok=True)
    st_shot.write(su_out_path, format="SU")


# for tr in st:
#     gx, gy, gelev = get_receiver_info(tr.id, receiver_geometry)
#     print(f"Trace ID: {tr.id}, Receiver X: {gx}, Y: {gy}, Elevation: {gelev}")
#     # tr.stats.su = {
#     #     "sx": sx,
#     #     "sy": sy,
#     #     "selev": selev,
#     #     "gx": gx,
#     #     "gy": gy,
#     #     "gelev": gelev,
#     #     "scalel": -100,
#     #     "scalco": -100,
#     #     "ep": int(shot["shot"]),   # e.g., shot number as ep
#     #     "fldr": int(shot["shot"]), # optional field record
#     #     "tracf": 1                 # update if you know trace number
#     # }