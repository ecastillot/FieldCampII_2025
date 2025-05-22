import ert
import numpy as np
import matplotlib.pyplot as plt

array_type = "wenner"
# array_type = "dipoledipole"
# array_type = "schlumberger"
a = 1 #"A" spacing between quadrapole, in meters
dc = 1.0 #DC current, in amperes
x0 = 0 #Initial location of electrode spread
cmd_filepath = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_ERT_032625/ROC2025.cmd"
stg_filepath = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_ERT_032625/ROC2025.stg"
crs_filepath = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_ERT_032625/ROC2025.crs"
array_save_path = "/groups/igonin/ecastillo/FieldCampII_2025/data/ROC_ERT_032625/ROC2025_arrays.png"

cmd = ert.parse_cmd(cmd_filepath)
stg = ert.parse_stg(stg_filepath)
crs = ert.parse_crs(crs_filepath)

geometry = cmd["geometry"]
commands = cmd["commands"]

num_electrodes = len(geometry)

electrodes = ert.findStations(array_type,a,num_electrodes,x0)
ert.plot_all_pseudosections(a=a,num_electrodes=num_electrodes,x0=x0,
                        show_electrodes=True,
                        use_pseudodepth=True,
                        save_path=array_save_path)


## others
# Midpoint in x
stg['x_mid'] = (stg['Ax'] + stg['Bx'] + stg['Mx'] + stg['Nx']) / 4
# Depth (z) approximation: depth = half of MN spacing (for plotting purposes)
stg['AB'] = np.sqrt((stg['Ax'] - stg['Bx'])**2 + (stg['Ay'] - stg['By'])**2 + (stg['Az'] - stg['Bz'])**2)
stg['depth'] = 0.125 * stg['AB']

import matplotlib.pyplot as plt
import matplotlib.tri as tri
# Apparent resistivity
res = stg['resistivity']  # or 'rho_schlumberger'

# Coordinates
x = stg['x_mid']
z = stg['depth']

# Triangulation
triang = tri.Triangulation(x, z)

# Plot
plt.figure(figsize=(10, 5))
tpc = plt.tricontourf(triang, res, levels=20, cmap='viridis')
plt.gca().invert_yaxis()
plt.colorbar(tpc, label='Apparent Resistivity (Ω·m)')
plt.xlabel("Midpoint X (m)")
plt.ylabel("Depth (m)")
plt.title("ERT Schlumberger Pseudosection")
plt.grid(True)
plt.tight_layout()
plt.savefig("/groups/igonin/ecastillo/FieldCampII_2025/code/pseudosection.png", dpi=300)
plt.show()

