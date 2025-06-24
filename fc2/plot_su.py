from obspy import read

segy_file = "/groups/igonin/ecastillo/FieldCampII_2025/out/segy_output/P1_strike_1.segy"  # change to your actual path
out_file = "/groups/igonin/ecastillo/FieldCampII_2025/out/segy_output/P1_strike_1.png"  # change to your actual path

# Read the SEG-Y file
st = read(segy_file, format="SEGY")

# Plot as a seismic section with multiple traces
fig = st.plot(type='section', recordstart=0, recordlength=2, scale=0.1, show=False)

# Save plot to PNG
fig.savefig(out_file, dpi=300)