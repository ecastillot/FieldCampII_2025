import numpy as np

# data = "/home/edc240000/ERT_FieldCamp/data/ROC_032625/ROC2025.cmd"
# data = "/home/edc240000/ERT_FieldCamp/data/ROC_032625/ROC2025.stg"
data = "/home/edc240000/ERT_FieldCamp/data/ROC_032625/ROC2025.crs"
with open(data, "rb") as file:
    data = file.read()
    print(data)

print(data[:100])  # Print first 100 bytes to inspect the structure