from seismic import (merge_shots_and_geometry,
           separate_shot_groups,plot_shot_data,
           plot_shot_geometry,read_waveforms,
           get_receiver_data,get_shots_data,
           process_and_export_shots)
from obspy import UTCDateTime
from obspy import Stream
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

receivers = "/groups/igonin/ecastillo/FieldCampII_2025/out/receivers.csv"
receivers_df = pd.read_csv(receivers)
receivers_df = receivers_df.iloc[0:30]

station_order = receivers_df["station"].tolist()
station_order = [str(station) for station in station_order]


smart_solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data_bck/smart_solo_data"

# shot 2
# shot_time = UTCDateTime("2025-06-14 14:00:49.350118") #emmanuel
shot_time = UTCDateTime("2025-06-14 14:00:49.285") #nadine

# #shot 3 
# shot_time = UTCDateTime("2025-06-14 14:01:02.534066") #emmanuel
# shot_time = UTCDateTime("2025-06-14 14:01:01.982") #nadine

# #shot 4
# shot_time = UTCDateTime("2025-06-14 14:01:17.335801") #emmanuel
# shot_time = UTCDateTime("2025-06-14 14:01:16.579") #nadine

#shot 5 
shot_time = UTCDateTime("2025-06-14 14:01:25.298") #nadine
shot_time = UTCDateTime("2025-06-14 14:01:25.423376") #emmanuel
 



l_padding = 0.05  # seconds before and after the shot time
r_padding = 0.6  # seconds before and after the shot time
starttime = shot_time-l_padding
endtime = shot_time + r_padding # 10 seconds after the start time
st = read_waveforms(folder_path=smart_solo_folder,
                    station="*",
                    component="Z",
                    starttime=starttime,
                    endtime=endtime)

st_sorted = Stream(
    sorted([tr for tr in st if tr.stats.station in station_order],
           key=lambda tr: station_order.index(tr.stats.station))
)

st_sorted.normalize()
n_traces = len(st_sorted)
npts = st_sorted[0].stats.npts
dt = st_sorted[0].stats.delta
times = np.arange(npts) * dt - l_padding

plt.figure(figsize=(10, 8))

for i, tr in enumerate(st_sorted):
    data = tr.data.astype(np.float32)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val  # normalize individual trace
    offset = i
    plt.plot(times, data + offset, color="black", linewidth=0.5)
    plt.axvline(x=0, color='red', linestyle='-', linewidth=3)
    plt.text(times[-1] + dt * 0.5, offset, tr.stats.station, va="center", fontsize=8)

plt.xlabel("Time (s)")
plt.ylabel("Station")
plt.title("Traces in Station Order")
plt.yticks(range(n_traces), [tr.stats.station for tr in st_sorted])
plt.grid(True)
plt.gca().invert_yaxis()
plt.tight_layout()

# Save the figure
output_path = "/groups/igonin/ecastillo/FieldCampII_2025/fc2/trigger_box.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
