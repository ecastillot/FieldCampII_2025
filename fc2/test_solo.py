from seismic import *

# df = read_shots("/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/SourceTesting/TB_INT00142.csv")
# print(df)

# solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/Pwaveseismic_Solo/"
solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/Pwaveseismic_Solo/"
station = get_station_from_solo(folder_path=solo_folder)
station_xy = append_xy_coords(station, lat_col='latitude', lon_col='longitude', 
                              elev_col_in_km='elevation', epsg="epsg:32614")
print(station_xy)