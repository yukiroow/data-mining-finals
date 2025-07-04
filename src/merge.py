import numpy as np
import rasterio

# Use QGIS after to clip to boundary :DDD. Don't forgor to rename the file.
# Boundary file is in `boundary` directory
# Don't forget to change the year after you run the script 2019 -> 2020
band_paths = ['../raw/2019/B02.jp2', '../raw/2019/B03.jp2', '../raw/2019/B04.jp2', '../raw/2019/B08.jp2']
with rasterio.open(band_paths[0]) as src:
    meta = src.meta
meta.update(count=len(band_paths))

# Export clipped imagery from QGIS as `blist_raw_[year].tif`
# `stacked` directory should only contain `blist_raw_2019.tif` and 'blist_raw_2020.tif'
with rasterio.open('../stacked/raw.tif', 'w', **meta) as dst:
    for idx, path in enumerate(band_paths, start=1):
        with rasterio.open(path) as src:
            dst.write(src.read(1), idx)