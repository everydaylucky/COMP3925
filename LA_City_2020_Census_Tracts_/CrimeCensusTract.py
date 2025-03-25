import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def load_census_tracts(shapefile_path='LA_City_2020_Census_Tracts_.shp'):
    """加载census tract shapefile数据"""
    print("加载census tract数据...")
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    tracts = gpd.read_file(shapefile_path)
    
    # 确保坐标系统正确
    if tracts.crs and tracts.crs != "EPSG:4326":
        print(f"将census tract数据从{tracts.crs}转换为WGS84(EPSG:4326)...")
        tracts = tracts.to_crs("EPSG:4326")
    
    print(f"成功加载{len(tracts)}个census tract区域")
    return tracts

def process_crime_data(crime_csv_path, tracts_gdf, output_csv="crime_data_with_census_tracts.csv", chunk_size=50000):
    """处理犯罪数据并关联census tract信息"""
    # 获取CSV文件总行数(用于进度条)
    print("计算文件总行数...")
    row_count = sum(1 for _ in open(crime_csv_path, 'r')) - 1  # 减去标题行
    print(f"文件共有{row_count}行数据")
    
    # 设置列名映射以便处理
    column_map = {
        'LAT': 'latitude',
        'LON': 'longitude',
        'DR_NO': 'crime_id',
        'DATE OCC': 'date',
        'AREA NAME': 'area_name',
        'Crm Cd Desc': 'crime_type'
    }
    
    # 设定要使用的列
    usecols = list(column_map.keys())
    
    # 分块处理
    print("开始分块处理犯罪数据...")
    chunks_processed = 0
    total_crimes = 0
    found_tract = 0
    no_tract = 0
    
    # 创建结果文件并写入标题
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write("crime_id,date,area_name,crime_type,latitude,longitude,census_tract_id,census_tract_label\n")
    
    # 创建总进度条
    with tqdm(total=row_count, desc="处理进度") as pbar:
        # 分块读取CSV
        for chunk in pd.read_csv(crime_csv_path, chunksize=chunk_size, usecols=usecols, low_memory=False):
            chunk_size = len(chunk)
            
            # 重命名列
            chunk = chunk.rename(columns=column_map)
            
            # 移除坐标为0或缺失的记录
            chunk = chunk.dropna(subset=['latitude', 'longitude'])
            chunk = chunk[(chunk['latitude'] != 0) & (chunk['longitude'] != 0)]
            
            # 创建GeoDataFrame
            geometry = [Point(lon, lat) for lon, lat in zip(chunk['longitude'], chunk['latitude'])]
            crime_gdf = gpd.GeoDataFrame(chunk, geometry=geometry, crs="EPSG:4326")
            
            # 空间连接 - 查找每个点所在的census tract
            joined = gpd.sjoin(crime_gdf, tracts_gdf, how="left", predicate="within")
            
            # 合并结果回原始数据
            chunk['census_tract_id'] = joined['CT20']
            chunk['census_tract_label'] = joined['LABEL']
            
            # 计数
            chunk_found = chunk['census_tract_id'].notna().sum()
            found_tract += chunk_found
            no_tract += (len(chunk) - chunk_found)
            total_crimes += len(chunk)
            
            # 附加到CSV
            chunk.to_csv(output_csv, mode='a', header=False, index=False, 
                        columns=['crime_id', 'date', 'area_name', 'crime_type', 
                                'latitude', 'longitude', 'census_tract_id', 'census_tract_label'])
            
            # 更新进度条
            chunks_processed += 1
            pbar.update(chunk_size)
            
            # 定期状态更新
            if chunks_processed % 10 == 0:
                print(f"\n已处理 {total_crimes} 条记录, 找到census tract: {found_tract}, 未找到: {no_tract}")
    
    print(f"\n完成! 处理了 {total_crimes} 条犯罪记录")
    print(f"找到census tract: {found_tract} ({found_tract/total_crimes*100:.1f}%)")
    print(f"未找到census tract: {no_tract} ({no_tract/total_crimes*100:.1f}%)")
    
    return output_csv

def generate_statistics(crime_tract_csv, output_stats_csv="crime_by_census_tract.csv"):
    """生成census tract犯罪统计"""
    print("开始生成census tract统计数据...")
    
    # 读取带有census tract信息的犯罪数据
    df = pd.read_csv(crime_tract_csv)
    
    # 删除没有census tract的记录
    df = df.dropna(subset=['census_tract_id'])
    
    # 按census tract统计犯罪总数
    stats = df.groupby(['census_tract_id', 'census_tract_label']).size().reset_index(name='total_crimes')
    
    # 按crime_type统计
    crime_types = df.groupby(['census_tract_id', 'crime_type']).size().reset_index(name='count')
    
    # 获取前10种最常见的犯罪类型
    top_crimes = crime_types.groupby('crime_type')['count'].sum().nlargest(10).index.tolist()
    
    # 为每种常见犯罪类型创建统计列
    for crime_type in top_crimes:
        crime_type_data = crime_types[crime_types['crime_type'] == crime_type]
        # 重命名列名，移除特殊字符
        type_col_name = f"crime_{crime_type.lower().replace(' ', '_').replace('-', '_').replace('/', '_')[:20]}"
        # 合并到主统计表
        stats = pd.merge(
            stats, 
            crime_type_data[['census_tract_id', 'count']].rename(columns={'count': type_col_name}),
            on='census_tract_id', 
            how='left'
        )
    
    # 填充缺失值为0
    stats = stats.fillna(0)
    
    # 计算每1000人犯罪率(如果有人口数据)
    # 这部分需要人口数据，先注释掉
    # stats['crime_rate_per_1000'] = (stats['total_crimes'] / stats['population']) * 1000
    
    # 保存统计结果
    stats.to_csv(output_stats_csv, index=False)
    print(f"统计数据已保存至 {output_stats_csv}")
    
    return stats

def visualize_crime_data(stats_df, tracts_gdf, output_path="crime_heatmap.png"):
    """生成犯罪热力图"""
    print("生成犯罪热力图...")
    
    # 合并统计数据和地理数据
    tracts_gdf = tracts_gdf.copy()
    merged_gdf = tracts_gdf.merge(stats_df, left_on='CT20', right_on='census_tract_id', how='left')
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # 绘制热力图
    merged_gdf.plot(
        column='total_crimes',
        cmap='OrRd',
        scheme='quantiles',
        k=5,
        legend=True,
        legend_kwds={'title': '犯罪数量', 'loc': 'lower right'},
        ax=ax,
        missing_kwds={'color': 'lightgrey'}
    )
    
    # 设置标题
    ax.set_title('洛杉矶各Census Tract犯罪热力图', fontsize=15)
    
    # 去除坐标轴
    ax.set_axis_off()
    
    # 保存图片
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热力图已保存至 {output_path}")
    
    return fig

def main():
    """主函数"""
    start_time = time.time()
    
    # 步骤1: 加载census tract数据
    tracts_gdf = load_census_tracts()
    
    # 步骤2: 处理犯罪数据并分配census tract
    crime_tract_csv = process_crime_data(
        crime_csv_path='Crime_Data_from_2020_to_Present.csv',
        tracts_gdf=tracts_gdf
    )
    
    # 步骤3: 生成统计数据
    stats_df = generate_statistics(crime_tract_csv)
    
    # 步骤4: 生成可视化
    visualize_crime_data(stats_df, tracts_gdf)
    
    elapsed_time = time.time() - start_time
    print(f"处理完成! 耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")

if __name__ == "__main__":
    main()