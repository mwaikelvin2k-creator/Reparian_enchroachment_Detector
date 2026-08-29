import ee

def initialize_and_download_satelite_tiles():
    print("Connecting to Google Earcth Cloud Servers...")
    ee.Initialize()

    #1. Define Spatial area of interest
    kasarani_aoi = ee.Geometry.Rectangle([36.80. -1.32, 36.95, -1.20])

    #2. Define the Cloud Masking function for Sentinel-2
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        # Bits 10 and 11 represent clouds and cirrus weather blocks
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        # Clear pixels have both flags set to zero
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
            qa.bitwiseAnd(cirrus_bit_mask).eq(0))

        return image.updateMask(mask).divide(10000)

    print("Filtering imagery collections by targeted 12-month climate ranges...")

    #3. EXTRACTION 1: Clean Wet Season Tile (April - May 2026)
    wet_season_composite = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                            .filterDate('2026-04-01', '2026-05-31')
                            .filterBounds(kasarani_aoi)
                            .map(mask_s2_clouds)
                            .median() # Temporal Composite Step
                            .select(['B4', 'B3', 'B2'])) #Extract Red, Green , Blue

    
    dry_season_composite = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                            .filterDate('2025-07-01', '2025-08-31')
                            .filterBounds(kasarani_aoi)
                            .map(mask_s2_clouds)
                            .median() # Temporal Composite Step
                            .select(['B4', 'B3', 'B2'])) #Extract Red, Green , Blue

    print("Cloud processing complete. Ready to export clean geotiff grids to data/raw/")


if __name__ == "__main__":
    print("Satellite Imagery Cloud Preprocessing Engine Active.")
    

