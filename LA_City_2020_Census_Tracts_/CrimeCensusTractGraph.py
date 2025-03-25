import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def load_data():
    """Load data files"""
    print("Loading data...")
    
    # Set environment variable to handle potentially missing .shx file
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    
    # Load census tract shapefile
    tracts_gdf = gpd.read_file('LA_City_2020_Census_Tracts_.shp')
    print(f"Loaded {len(tracts_gdf)} census tract areas")
    
    # Load crime statistics data
    stats_df = pd.read_csv('crime_by_census_tract.csv')
    print(f"Loaded crime statistics for {len(stats_df)} census tracts")
    
    # Print column names to diagnose matching issues
    print("Census tract shapefile columns:", tracts_gdf.columns.tolist())
    print("Crime statistics columns:", stats_df.columns.tolist())
    
    # Print sample data from both datasets
    print("\nSample tract data (first row):")
    print(tracts_gdf.iloc[0])
    print("\nSample crime stats data (first row):")
    print(stats_df.iloc[0])
    
    # 尝试不同的ID匹配方式
    # 1. 尝试使用LABEL列代替CT20列进行匹配
    if 'LABEL' in tracts_gdf.columns and 'census_tract_label' in stats_df.columns:
        print("Trying to match using LABEL column instead")
        tracts_gdf['LABEL'] = tracts_gdf['LABEL'].astype(str)
        stats_df['census_tract_label'] = stats_df['census_tract_label'].astype(str)
        
        # 打印样本数据帮助调试
        print("First 5 LABEL values:", tracts_gdf['LABEL'].head().tolist())
        print("First 5 census_tract_label values:", stats_df['census_tract_label'].head().tolist())
    
    # 2. 确保CT20和census_tract_id数据类型一致
    if 'CT20' in tracts_gdf.columns:
        tracts_gdf['CT20'] = tracts_gdf['CT20'].astype(str)
    if 'census_tract_id' in stats_df.columns:
        stats_df['census_tract_id'] = stats_df['census_tract_id'].astype(str)
    
    # 3. 尝试格式化census_tract_id (移除可能的小数点)
    if 'census_tract_id' in stats_df.columns:
        # 转换浮点数为整数字符串 (例如 "123.0" -> "123")
        try:
            stats_df['census_tract_id'] = stats_df['census_tract_id'].apply(
                lambda x: str(int(float(x))) if pd.notnull(x) and '.' in str(x) else str(x)
            )
        except:
            print("Warning: Failed to format census_tract_id column")
    
    return tracts_gdf, stats_df

def create_choropleth(tracts_gdf, stats_df, output_path="crime_heatmap.png"):
    """Create crime distribution heatmap"""
    print("Generating crime heatmap...")
    
    # 数据类型转换 - 确保匹配
    tracts_gdf = tracts_gdf.copy()
    stats_df = stats_df.copy()
    
    # 转换为字符串进行匹配
    tracts_gdf['CT20'] = tracts_gdf['CT20'].astype(str)
    stats_df['census_tract_id'] = stats_df['census_tract_id'].astype(str)
    
    # 打印匹配前的数据样本用于调试
    print("Shapefile CT20数据样本:", tracts_gdf['CT20'].head().tolist())
    print("统计数据census_tract_id样本:", stats_df['census_tract_id'].head().tolist())
    
    # 合并统计数据和地理数据
    merged_gdf = tracts_gdf.merge(stats_df, left_on='CT20', right_on='census_tract_id', how='left')
    
    # 检查匹配结果
    print(f"成功匹配记录数: {merged_gdf['total_crimes'].notna().sum()} / {len(tracts_gdf)}")
    
    # 如果仍然没有匹配，需要实际检查数据内容
    if merged_gdf['total_crimes'].notna().sum() == 0:
        print("\nWARNING: Still no matches found. Investigating data:")
        print("CT20 values (first 10):", tracts_gdf['CT20'].head(10).tolist())
        print("census_tract_id values (first 10):", stats_df['census_tract_id'].head(10).tolist())
        
        if 'LABEL' in tracts_gdf.columns:
            print("LABEL values (first 10):", tracts_gdf['LABEL'].head(10).tolist())
        if 'census_tract_label' in stats_df.columns:
            print("census_tract_label values (first 10):", stats_df['census_tract_label'].head(10).tolist())
    
    # 如果尝试匹配都失败，生成基本地图
    if merged_gdf['total_crimes'].notna().sum() == 0:
        print("WARNING: No crime data matched with census tracts. Creating a basic map.")
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Draw basic map
        tracts_gdf.plot(
            edgecolor='black',
            linewidth=0.3,
            ax=ax,
            color='lightgrey'
        )
        
        ax.set_title('Los Angeles Census Tracts (No Crime Data Matched)', fontsize=16)
        ax.set_axis_off()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Basic map saved to {output_path}")
        return fig
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Use custom color map
    cmap = plt.cm.OrRd
    
    # Data range - log transform for better distribution display
    vmin = merged_gdf['total_crimes'].min() or 1
    vmax = merged_gdf['total_crimes'].max() or 100
    print(f"Data range: min={vmin}, max={vmax}")
    
    # Use linear norm if min or max is invalid for log scale
    if vmin <= 0 or vmax <= 0:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    
    # Draw heatmap
    merged_gdf.plot(
        column='total_crimes',
        cmap=cmap,
        norm=norm,
        edgecolor='black',
        linewidth=0.3,
        ax=ax,
        missing_kwds={'color': 'lightgrey'}
    )
    
    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Crime Count', fontsize=12)
    
    # Set title
    ax.set_title('Crime Heatmap by Census Tract in Los Angeles', fontsize=16)
    
    # Remove axes
    ax.set_axis_off()
    
    # Save image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")
    
    return fig

def create_crime_type_charts(stats_df, output_path="crime_types_chart.png"):
    """Create crime type distribution charts"""
    # Find crime type columns
    crime_cols = [col for col in stats_df.columns if col.startswith('crime_')]
    
    if not crime_cols:
        print("No crime type data columns found, cannot generate crime type charts")
        return None
    
    # Create crime type totals
    crime_types_sum = {}
    for col in crime_cols:
        if col != 'crime_id':  # Skip non-type columns
            # Extract crime type from column name
            crime_type = col.replace('crime_', '').replace('_', ' ').title()
            crime_types_sum[crime_type] = stats_df[col].sum()
    
    # Create pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Sort and get top 10 crime types
    sorted_types = dict(sorted(crime_types_sum.items(), key=lambda x: x[1], reverse=True))
    top_types = {k: sorted_types[k] for k in list(sorted_types)[:10]}
    
    # Colors
    colors = plt.cm.tab20.colors
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        top_types.values(), 
        labels=None,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors[:len(top_types)]
    )
    
    # Set pie chart text
    for autotext in autotexts:
        autotext.set_fontsize(8)
    
    # Add legend
    ax1.legend(
        wedges, 
        top_types.keys(),
        title="Crime Types",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    ax1.set_title("Top 10 Most Common Crime Types in Los Angeles", fontsize=14)
    
    # Bar chart
    y_pos = np.arange(len(top_types))
    ax2.barh(y_pos, list(top_types.values()), color=colors[:len(top_types)])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(list(top_types.keys()))
    ax2.invert_yaxis()  # Largest value at top
    ax2.set_title("Top 10 Most Common Crime Types Count in Los Angeles", fontsize=14)
    
    # Add value labels
    for i, v in enumerate(list(top_types.values())):
        ax2.text(v + 0.1, i, f"{v:,}", va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Crime type charts saved to {output_path}")
    
    return fig

def create_crime_hotspots_map(tracts_gdf, stats_df, output_path="crime_hotspots.png"):
    """Create crime hotspots map"""
    print("Generating crime hotspots map...")
    
    # Merge data
    merged_gdf = tracts_gdf.merge(
        stats_df, 
        left_on='CT20', 
        right_on='census_tract_id', 
        how='left'
    )
    
    # Calculate quantiles
    q = merged_gdf['total_crimes'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    
    # Create hotspot classification
    merged_gdf['hotspot_level'] = 0  # No data
    merged_gdf.loc[merged_gdf['total_crimes'].notna(), 'hotspot_level'] = 1  # Has data but below Q1
    merged_gdf.loc[merged_gdf['total_crimes'] > q[0.25], 'hotspot_level'] = 2  # Q1-Q2
    merged_gdf.loc[merged_gdf['total_crimes'] > q[0.5], 'hotspot_level'] = 3   # Q2-Q3
    merged_gdf.loc[merged_gdf['total_crimes'] > q[0.75], 'hotspot_level'] = 4  # Q3-90%
    merged_gdf.loc[merged_gdf['total_crimes'] > q[0.9], 'hotspot_level'] = 5   # 90%-95%
    merged_gdf.loc[merged_gdf['total_crimes'] > q[0.95], 'hotspot_level'] = 6  # 95%-99%
    merged_gdf.loc[merged_gdf['total_crimes'] > q[0.99], 'hotspot_level'] = 7  # Top 1%
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    # Create hotspot color scale
    cmap = plt.cm.get_cmap('YlOrRd', 7)
    
    # Draw hotspots map
    merged_gdf.plot(
        column='hotspot_level',
        cmap=cmap,
        edgecolor='black',
        linewidth=0.3,
        ax=ax,
        categorical=True,
        missing_kwds={'color': 'lightgrey'}
    )
    
    # Custom legend
    legend_labels = {
        0: 'No Data',
        1: 'Very Low (< 25%)',
        2: 'Low (25% - 50%)',
        3: 'Medium (50% - 75%)',
        4: 'High (75% - 90%)',
        5: 'Very High (90% - 95%)',
        6: 'Extreme (95% - 99%)',
        7: 'Highest (> 99%)'
    }
    
    # Create legend elements
    legend_elements = [
        Patch(facecolor='lightgrey', edgecolor='black', label='No Data')
    ]
    
    for i in range(1, 8):
        legend_elements.append(
            Patch(facecolor=cmap(i-1), edgecolor='black', 
                  label=f"{legend_labels[i]} ({int(q[list(q.index)[min(i-1, len(q)-1)]]) if i > 1 else 0}+ crimes)")
        )
    
    # Add legend
    ax.legend(
        handles=legend_elements,
        title="Crime Hotspot Level",
        loc="lower right",
        fontsize=10
    )
    
    # Set title
    ax.set_title('Los Angeles Crime Hotspots Map', fontsize=16)
    
    # Remove axes
    ax.set_axis_off()
    
    # Save image
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Hotspots map saved to {output_path}")
    
    return fig

def main():
    """Main function"""
    # Load data
    tracts_gdf, stats_df = load_data()
    
    # Create heatmap
    create_choropleth(tracts_gdf, stats_df)
    
    # Create crime type charts
    create_crime_type_charts(stats_df)
    
    # Create hotspots map
    create_crime_hotspots_map(tracts_gdf, stats_df)
    
    print("All visualization charts generated successfully!")

if __name__ == "__main__":
    main()