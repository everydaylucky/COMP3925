import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import os

def get_census_tract(latitude, longitude, shapefile_path='LA_City_2020_Census_Tracts_.shp'):
    """
    Given coordinates in Los Angeles, returns the census tract data containing those coordinates.
    
    Parameters:
    -----------
    latitude : float
        Latitude of the location
    longitude : float
        Longitude of the location
    shapefile_path : str
        Path to the shapefile containing census tract data
        
    Returns:
    --------
    pandas.Series or None
        Census tract data if coordinates are within a tract, None otherwise
    """
    # 设置环境变量来恢复/创建缺失的.shx文件
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    
    # Load census tract shapefile
    tracts = gpd.read_file(shapefile_path)
    
    # Create a point from the coordinates (WGS84)
    point = Point(longitude, latitude)
    point_gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326")
    
    # 将点转换为与shapefile相同的坐标系
    point_gdf = point_gdf.to_crs(tracts.crs)
    
    # Perform spatial join to find which tract contains the point
    joined = gpd.sjoin(point_gdf, tracts, how="left", predicate="within")
    
    # Return the tract data if a match is found
    if not joined.empty and pd.notna(joined.index_right[0]):
        return tracts.loc[joined.index_right[0]]
    else:
        return None

if __name__ == "__main__":
    # 打印整个数据帧查看所有可用列
    print("Loading LA census tract shapefile...")
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    tracts = gpd.read_file('LA_City_2020_Census_Tracts_.shp')
    print("All columns in the shapefile:")
    print(tracts.columns.tolist())
    print("First row of data:")
    print(tracts.iloc[0])
    
    # Los Angeles coordinates (downtown)
    la_latitude = 34.0375
    la_longitude = -118.351
    
    # Get census tract for LA coordinates
    la_tract = get_census_tract(la_latitude, la_longitude)
    if la_tract is not None:
        print("\nCensus tract found for Los Angeles coordinates")
        print("All available data for this tract:")
        print(la_tract)
    else:
        print("\nLos Angeles coordinates not found in any census tract")
