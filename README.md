# 数据集
## `Crime`: [Link](https://catalog.data.gov/dataset/crime-data-from-2020-to-present)
### 特征解释（来自GPT），please also refer to official info
1. **DR_NO（报告编号）**：案件的唯一标识号。  
2. **Date Rptd（报告日期）**：案件被报告给警方的日期。  
3. **DATE OCC（发生日期）**：案件实际发生的日期。  
4. **TIME OCC（发生时间）**：案件发生的时间，通常为24小时制（如2130代表21:30）。  
5. **AREA（区域代码）**：发生案件的警察辖区代码。  
6. **AREA NAME（区域名称）**：案件发生的警察辖区名称。  
7. **Rpt Dist No（报告区编号）**：更小的报告区域编号。  
8. **Part 1-2（案件类别）**：  
   - `1`：重大犯罪（如谋杀、抢劫、强奸、严重袭击、偷车等）。  
   - `2`：轻罪（如轻微袭击、伪造、赌博等）。  
9. **Crm Cd（犯罪代码）**：特定犯罪类型的代码。  
10. **Crm Cd Desc（犯罪描述）**：犯罪行为的具体描述，如`VEHICLE - STOLEN`（车辆被盗）。  
11. **Mocodes（作案手法代码）**：表示犯罪的具体作案手段，多个代码用空格隔开。  
12. **Vict Age（受害人年龄）**：受害者的年龄，`0` 可能表示未知或未填写。  
13. **Vict Sex（受害人性别）**：  
    - `M`：男性  
    - `F`：女性  
    - `X`：未知或未指定  
14. **Vict Descent（受害人族裔）**：受害者的种族或族裔代码，如 `O` 代表其他（Other）。  
15. **Premis Cd（案发地点代码）**：代表案件发生地点的代码。  
16. **Premis Desc（案发地点描述）**：描述犯罪发生的具体地点，如`STREET`（街道）或`BUS STOP/LAYOVER`（公交站）。  
17. **Weapon Used Cd（使用武器代码）**：如果案件涉及武器，则此字段会包含对应武器的代码。  
18. **Weapon Desc（武器描述）**：使用武器的描述，如`FIREARM`（枪支）。如果未使用武器，则为空。  
19. **Status（案件状态代码）**：案件的当前状态，例如：  
    - `AA`（Adult Arrest）：成人被捕  
    - `IC`（Invest Cont）：调查中  
20. **Status Desc（案件状态描述）**：对 Status 字段的详细解释，如`Adult Arrest`（成人被捕）。`Invest Cont` 说明该案件尚未完全结案。
21. **Crm Cd 1-4（犯罪代码 1-4）**：一个案件可能涉及多个犯罪类型，这些字段存储最多四个相关的犯罪代码。  
22. **LOCATION（案发地点）**：具体的案发地址。  
23. **Cross Street（交叉街道）**：如果适用，显示案件发生地点附近的交叉街道。  
24. **LAT（纬度）**：案件发生地点的纬度坐标。  
25. **LON（经度）**：案件发生地点的经度坐标。、

## `Education Level`: [Link](https://data.lacounty.gov/datasets/lacounty::bachelors-degree-or-higher-census-tract/about)
### 特征解释（来自官方文档）
1. **tract（人口普查区）**：字符串类型，表示人口普查区编号，用于唯一标识特定地理区域。
2. **name（名称）**：字符串类型，表示区域的名称。
3. **bachelors（学士学位或更高学历百分比）**：双精度小数类型，表示该区域25岁及以上人口中拥有学士学位或更高学历的百分比。
   - 数据来源：2023年美国社区调查5年估计（ACS 5-year estimates）
   - 具体字段：DP02_0068PE
   - 数据链接：https://data.census.gov/table/ACSDP5Y2023.DP02
4. **sup_dist（监督区）**：字符串类型，表示洛杉矶县监督区编号（District 1-5）。
5. **csa（城市或社区）**：字符串类型，表示城市或社区名称。
6. **spa（服务规划区）**：字符串类型，表示服务规划区编号和名称，包括：
   - SPA 1 - Antelope Valley
   - SPA 2 - San Fernando
   - SPA 3 - San Gabriel
   - SPA 4 - Metro
   - SPA 5 - West
   - SPA 6 - South
   - SPA 7 - East
   - SPA 8 - South Bay
7. **Shape__Area（地理区域面积）**：双精度小数类型，表示该区域的地理面积。
8. **Shape__Length（地理区域边界长度）**：双精度小数类型，表示该区域边界的长度。

### 数据说明
- 数据来自2023年美国社区调查5年估计（ACS 5-year estimates）
- 数据由洛杉矶县信息系统部（ISD）维护
- 如需更多信息，请联系：egis@isd.lacounty.gov
- 数据按人口普查区级别提供，可用于分析教育水平的地理分布
- 可与人口普查区其他数据（如人口、收入等）进行关联分析

## `Population`: [Link](https://geohub.lacity.org/datasets/lacounty::2023-population-and-poverty-by-split-tract/about)
### 特征解释（来自官方文档）
1. **CT20（人口普查区编号）**：2020年人口普查区编号，用于标识特定地理区域。
2. **FIP22（FIPS代码）**：2023年城市FIPS代码，用于唯一识别地理区域。
3. **CITY（城市）**：建制城市的城市名称，非建制地区标记为"非建制"（截至2023年7月1日）。
4. **CSA（县级统计区）**：非建制区社区名称和洛杉矶市街区名称。
5. **CT20FIP23CSA（复合地理标识符）**：结合了2020年人口普查区、2023年建制城市、非建制地区和洛杉矶街区的唯一标识。
6. **SPA22（服务规划区编号）**：2022年服务规划区编号。
7. **SPA_NAME（服务规划区名称）**：服务规划区的具体名称。
8. **HD22（健康区编号）**：2022年健康区编号。
9. **HD_NAME（健康区名称）**：健康区的具体名称。
10. **POP23_AGE_0_4 ~ POP23_AGE_85_100（各年龄段人口）**：2023年各年龄段的人口数量，如：
    - `POP23_AGE_0_4`：0-4岁人口
    - `POP23_AGE_5_9`：5-9岁人口
    - `POP23_AGE_10_14`：10-14岁人口
    - `POP23_AGE_15_17`：15-17岁人口
    - `POP23_AGE_18_19`：18-19岁人口
    - `POP23_AGE_20_44`：20-44岁人口
    - `POP23_AGE_25_29`：25-29岁人口
    - `POP23_AGE_30_34`：30-34岁人口
    - `POP23_AGE_35_44`：35-44岁人口
    - `POP23_AGE_45_54`：45-54岁人口
    - `POP23_AGE_55_64`：55-64岁人口
    - `POP23_AGE_65_74`：65-74岁人口
    - `POP23_AGE_75_84`：75-84岁人口
    - `POP23_AGE_85_100`：85岁及以上人口
11. **种族人口统计**：
    - `POP23_WHITE`：2023年非西班牙裔白人人口
    - `POP23_BLACK`：2023年非西班牙裔非洲裔美国人人口
    - `POP23_AIAN`：2023年非西班牙裔美洲印第安人或阿拉斯加原住民人口
    - `POP23_ASIAN`：2023年非西班牙裔亚裔人口
    - `POP23_HNPI`：2023年非西班牙裔夏威夷原住民或太平洋岛民人口
    - `POP23_HISPANIC`：2023年西班牙裔人口
12. **性别人口统计**：
    - `POP23_MALE`：2023年男性人口
    - `POP23_FEMALE`：2023年女性人口
13. **贫困人口统计**：
    - `POV23_WHITE`：2023年非西班牙裔白人贫困人口（收入低于联邦贫困线100%）
    - `POV23_BLACK`：2023年非西班牙裔非洲裔美国人贫困人口
    - `POV23_AIAN`：2023年非西班牙裔美洲印第安人或阿拉斯加原住民贫困人口
    - `POV23_ASIAN`：2023年非西班牙裔亚裔贫困人口
    - `POV23_HNPI`：2023年非西班牙裔夏威夷原住民或太平洋岛民贫困人口
    - `POV23_HISPANIC`：2023年西班牙裔贫困人口
    - `POV23_TOTAL`：2023年总贫困人口
14. **总体统计**：
    - `POP23_TOTAL`：2023年总人口
    - `AREA_SQMil`：面积（平方英里）
    - `POP23_DENSITY`：2023年人口密度（每平方英里）
    - `POV23_PERCENT`：2023年贫困率（百分比）

### 数据说明
- 所有人口和贫困数据估计截至2023年7月1日
- 2010年人口普查区和2020年人口普查区并不相同
- 城市和社区边界截至2023年7月1日
- 数据是通过将人口和贫困信息归因于分割区域的地理信息而创建的
- 分割区域是2020年人口普查区域边界与洛杉矶县法定城市边界和非建制区域（CSA）分割的产物

#任务
1.分析Crime数据集，做Crime相关的一些热力图还有犯罪率、犯罪种类分析的可视化
2.分析Education数据集，做Education相关的一些可视化
3.分析Population数据集，做Population相关的一些可视化（可以有两个人做）
4.Predictive analysis with Power BI and Azure ML，预测犯罪两
4.1.特征工程：时间、地点、案件类型、武器使用等信息。
4.2.犯罪严重程度预测（分类任务）：基于历史数据预测某案件是否是重大犯罪（Part 1 vs Part 2）。
4.3.特征工程：
- 时间特征（Date Occ、Time Occ）
- 地理特征（Area Name, Reporting District）
- 案件特征（Crm Cd, Weapon Used Cd, Premis Cd）
- 建模：训练一个 分类模型（Random Forest / XGBoost） 预测案件是重大犯罪（Part 1）还是轻罪（Part 2）。
5.做网站