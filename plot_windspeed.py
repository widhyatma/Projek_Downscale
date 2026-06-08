import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio

def main():
    input_dir = os.path.join("data", "windspeed")
    output_dir = "output_plots"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    
    if not tif_files:
        print(f"No .tif files found in {input_dir}")
        return
        
    # Extent for Indonesia
    extent = [95.0, 141.0, -11.0, 6.0]
    
    for tif_file in tif_files:
        print(f"Processing {tif_file}...")
        
        # Extract year
        filename = os.path.basename(tif_file)
        match = re.search(r'(\d{4})', filename)
        year = match.group(1) if match else "Unknown Year"
        
        # Read raster
        with rasterio.open(tif_file) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
                
            bounds = src.bounds
            raster_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            
        fig = plt.figure(figsize=(10, 5), dpi=600)
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
        img = ax.imshow(data, origin='upper', extent=raster_extent, 
                        transform=ccrs.PlateCarree(), cmap='YlGnBu')
        
        cbar = plt.colorbar(img, ax=ax, orientation='vertical', shrink=0.7, pad=0.02)
        cbar.set_label("Wind Speed (m/s)")
        
        plt.title(f"Average Wind Speed in Indonesia - {year}")
        
        out_filename = f"plot_{os.path.splitext(filename)[0]}.png"
        out_path = os.path.join(output_dir, out_filename)
        
        plt.savefig(out_path, bbox_inches='tight', dpi=600)
        plt.close(fig)
        
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
