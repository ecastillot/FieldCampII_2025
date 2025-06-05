from seismic import *

# data = get_latlon_from_digisolo_log(path="/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/Pwaveseismic_Solo/453024131/20250507161620/DigiSolo.LOG")
# print(data)

# solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/Pwaveseismic_Solo/"
solo_folder = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_S_050725/Pwaveseismic_Solo/"
station = get_station_from_solo(folder_path=solo_folder)
print(station)