import ert as ert_utils
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ert
from pygimli.physics.ert import drawERTData
import pandas as pd

array_type = "wenner"
a = 1  # "A" spacing between quadrupole, in meters
dc = 1.0  # DC current, in amperes
x0 = 0  # Initial location of electrode spread
cmd_filepath = "/home/emmanuel/FieldCampII_2025/data/ROC_ERT_032625/ROC2025.cmd"
stg_filepath = "/home/emmanuel/FieldCampII_2025/data/ROC_ERT_032625/ROC2025.stg"
crs_filepath = "/home/emmanuel/FieldCampII_2025/data/ROC_ERT_032625/ROC2025.crs"
array_save_path = "/home/emmanuel/FieldCampII_2025/data/ROC_ERT_032625/ROC2025_arrays.png"

# Load data
cmd = ert_utils.parse_cmd(cmd_filepath)
stg = ert_utils.parse_stg(stg_filepath)
crs = ert_utils.parse_crs(crs_filepath)

# Step 0: Filter the DataFrame for valid resistivity and error values
stg = stg[(stg['resistivity'] > 0) & (np.isfinite(stg['resistivity'])) & 
          (stg['error_percent_tenths'] > 0) & (np.isfinite(stg['error_percent_tenths']))]

Q1 = stg['resistivity'].quantile(0.25)
Q3 = stg['resistivity'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter
stg = stg[(stg['resistivity'] >= lower_bound) & (stg['resistivity'] <= upper_bound)]
print(f"Filtered stg DataFrame: {len(stg)} valid rows remaining")

# Step 1: Extract unique electrode positions
electrodes = np.unique(np.concatenate([
    stg[['Ax']].values,  # Assuming 2D, use x-coordinates
    stg[['Bx']].values,
    stg[['Mx']].values,
    stg[['Nx']].values
], axis=0)).flatten()  # Ensure a 1D array

# Sort electrodes by x-coordinate for a 2D line
electrodes = np.sort(electrodes)
sensors = np.column_stack((electrodes, np.zeros_like(electrodes)))  # Add y=0 for 2D

# Debug: Check electrodes
print("Electrodes shape:", electrodes.shape)
print("Electrodes (first few):", electrodes[:5])
print("Min electrode x:", min(electrodes))
print("Max electrode x:", max(electrodes))

# Step 2: Create a PyGIMLi data container
data = pg.DataContainerERT()

# Set sensor (electrode) positions
for i, pos in enumerate(sensors):
    data.createSensor(pos)

# Step 3: Assign A, B, M, N indices and data
electrode_dict = {x: i for i, x in enumerate(electrodes)}
a = [electrode_dict[x] for x in stg['Ax']]
b = [electrode_dict[x] for x in stg['Bx']]
m = [electrode_dict[x] for x in stg['Mx']]
n = [electrode_dict[x] for x in stg['Nx']]
r = [x for x in stg['resistivity']]
e = [x / 1000.0 for x in stg['error_percent_tenths']]

data.resize(len(stg))  # Set number of measurements
data.set('a', a)  # Current electrode A indices
data.set('b', b)  # Current electrode B indices
data.set('m', m)  # Potential electrode M indices
data.set('n', n)  # Potential electrode N indices


data.set('rhoa', r)  # Apparent resistivity
data.set('err', e)  # Error as fraction

# Compute geometric factors based on electrode positions
data['k'] = pg.physics.ert.createGeometricFactors(data, verbose=True)

print("Apparent resistivity values summary:")
print("Min:", np.min(data['rhoa']))
print("Max:", np.max(data['rhoa']))
print("Any positive?", np.any(data['rhoa'] > 0))
print("Any NaNs?", np.any(np.isnan(data['rhoa'])))
print("Any zeros?", np.any(data['rhoa'] == 0))

invalid_indices = [i for i, val in enumerate(data['rhoa']) if val <= 0]
# Create a copy of the original data container
data = data.copy()
# Remove invalid indices
if invalid_indices:
    idx_vec = pg.VectorUInt(invalid_indices)
    data.remove(idx_vec)

print("Apparent resistivity values summary:")
print("Min:", np.min(data['rhoa']))
print("Max:", np.max(data['rhoa']))
print("Any positive?", np.any(data['rhoa'] > 0))
print("Any NaNs?", np.any(np.isnan(data['rhoa'])))
print("Any zeros?", np.any(data['rhoa'] == 0))


fig, ax = plt.subplots()
plt.scatter(data['rhoa'],range(data.size()), color='blue', s=10)  # fixed color
plt.title("Raw Apparent Resistivity Data")
plt.ylabel("Measurement Index")
plt.xlabel("Apparent Resistivity (Ohm·m)")

fig, ax = plt.subplots()
ax.hist(data['rhoa'], bins=50, color='steelblue', edgecolor='black')
ax.set_title("Distribution of Apparent Resistivity")
ax.set_xlabel("Apparent Resistivity (Ohm·m)")
ax.set_ylabel("Count")



plt.show()




# Step 4: Create a geometry for the inversion
x_min = np.min(electrodes) - 10  # Extend 10 units beyond min electrode
x_max = np.max(electrodes) + 10  # Extend 10 units beyond max electrode
y_min = -10  # Depth of the model
y_max = 0    # Surface

# Debug: Print bounds
print(f"World bounds: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

# Define a simple 2D world
world = mt.createWorld(start=[x_min, y_max], end=[x_max, y_min], 
                        worldMarker=True)

# Add electrodes to the geometry for mesh refinement
for p in data.sensors():
    world.createNode(p)
    world.createNode(p - [0, 0.1])  # Refine near electrodes

# Create a mesh for inversion
mesh = mt.createMesh(world, quality=34)

# Take a look at the mesh and the resistivity distribution
pg.show(mesh, showMesh=True)
plt.title("Inversion Mesh")
plt.show()

## Step 5: Initialize ERTManager and perform inversion
# Step 5: Initialize ERTManager and perform inversion
mgr = ert.ERTManager('real_ert_data.dat')
model1 = mgr.invert(lam=1, verbose=False)

# Step 6: Check inversion quality
print(f"Inversion stopped with chi² = {mgr.inv.chi2():.3f}")

# Step 7: Visualize results
ax,cb = pg.show(mgr.paraDomain, model1, label='Resistivity (Ohm·m)', cMap='Spectral_r',
        logScale=True)

ref_xlims = ax.get_xlim()
ref_ylims = ax.get_ylim()

mgr.showResultAndFit()


# Step 8: Different grids

# Previous grid
meshPD = pg.Mesh(mgr.paraDomain) # Save copy of para mesh for plotting later

# custom grid
inversionDomain = pg.createGrid(x=np.linspace(start=ref_xlims [0], stop=ref_xlims[1] , num=33),
                                y=-pg.cat([0], pg.utils.grange(0.1, ref_ylims[0]*-1, n=20))[::-1],
                                marker=2)
grid = pg.meshtools.appendTriangleBoundary(inversionDomain, marker=1,
                                           xbound=25, ybound=10)
pg.show(grid, markers=True)

model2 = mgr.invert(data, mesh=grid, lam=20, verbose=False)
print(f"Inversion stopped with chi² = {mgr.inv.chi2():.3f}")
#np.testing.assert_approx_equal(mgr.inv.chi2(), 1.4, significant=2)

modelPD2 = mgr.paraModel(model2)  # do the mapping
pg.show(mgr.paraDomain, modelPD2, label='Resistivity (Ohm·m)', cMap='Spectral_r',
        logScale=True)
#pg.show(grid, markers=True)

# Step 7: Visualize results
# Show the apparent resistivity data with explicit range

#fig, (ax1, ax2,ax3) = plt.subplots(3,1, sharex=True, sharey=True, figsize=(8,7))

#pg.show(meshPD, inv1, ax=ax1, hold=True, cMap="Spectral_r", logScale=True,
#        orientation="vertical", cMin=25, cMax=150)
#pg.show(mgr.paraDomain, modelPD2,ax=ax2, label='Model', cMap='Spectral_r',
#        logScale=True, cMin=25, cMax=150)
#mgr.showResult(ax=ax3, cMin=25, cMax=150, orientation="vertical")


#labels = ["True model", "Inversion unstructured mesh", "Inversion regular grid"]
#for ax, label in zip([ax1, ax2, ax3], labels):
#    ax.set_xlim(mgr.paraDomain.xmin(), mgr.paraDomain.xmax())
#    ax.set_ylim(mgr.paraDomain.ymin(), mgr.paraDomain.ymax())
#    ax.set_title(label)

