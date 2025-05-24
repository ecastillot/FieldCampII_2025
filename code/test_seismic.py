# from obspy import UTCDateTime
from seismic import read_shots,read_waveforms,read_stations
# from obspy import UTCDateTime

import obsplus
# filepath = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/SourceTesting/TB_INT00142.csv"
# df = parse_shot_file(filepath, gps_start=None)
# print(df)

folder = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/PwaveSeismic"

st = read_waveforms(folder_path=folder, 
                    station="4530243*", 
                    component="Z")
print(st)


folder = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/PwaveSeismic/FDSN Information"
stations = read_stations(folder_path=folder)
print(stations)
# inv = read_inventory(
#     "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/PwaveSeismic/FDSN Information/FDSN_Information_453024131_1.xml"
# )

# data = inv.to_df()
# print(data)






# from code.bck.seismic_client import SmartSoloClient
# client = SmartSoloClient(
#     root=folder,
#     fmt="{station}.0001.{year}.{month:02d}.{day:02d}.{hour:02d}.{minute:02d}.{second:02d}.{milliseconds:03d}.{component}.miniseed"
# )

# st = client.get_waveforms(
#     # network="",  # you can leave as empty string if unused
#     # station="453024131,453024638",
#     # initial_station_name = "",
#     station="24131,24638",
#     # station="*",
#     component="Z",
#     # location="",
#     # channel="EHZ",
#     starttime=UTCDateTime("2025-05-07T15:42:26"),
#     endtime=UTCDateTime("2025-05-07T20:43:50")
# )
# print(st)

# import glob
# from obspy import read
# from obspy import Stream

# paths = glob.glob(
#     "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/PwaveSeismic/*"
# )

# stream = Stream()
# for path in paths:
#     format_file = path.split(".")[-1]
#     if format_file == "miniseed":
#         st = read(path)
#         stream += st
# print(stream)
    
        