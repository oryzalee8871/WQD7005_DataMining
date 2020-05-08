# WQD7005_DataMining 
## Climate, News and Stock Price based Predictability on Crude Palm Oil (CPO) Commodity Price

This is a Data Mining course work project in fulfillment to Master's of Data Science in University of Malaya. 

## Table of contents
* [Introduction](#Introduction)
* [Objectives](#Objectives)
* [Milestone 1 - Data Aquisition & Web Crawling](#Milestone-1---Data-Aquisition-&-Web-Crawling)
* [Milestone 2 - Store Data into Hive DataWarehouse](#Milestone-2---DataWarehouse & DataLake Implementation)
* [Milestone 3 - Empty Placeholder]
* [Milestone 4 - Empty Placeholder]
* [Milestone 5 - Empty Placeholder]
* [YouTube Links](#YouTube-Links)
* [References](#References)


## Introduction

&emsp;&emsp;According to Santhia (2017)[[1]](#1), the top palm oil producing nations in 2016  are Indonesia, leading by 34.52 mil MT of palm oil, followed by Malaysia at 17.32 mil MT where the two countries made up 87% of the total palm oil production market. Palm oil trees, like most plants are susceptible to climate changes and local atmospheric variables. As investigated by Oettli et al (2018)[[2]](#2), woittiez et al (2017)[[3]](#3) and Kamil & Omar (2016)[[4]](#4), attributes such as rainfall, temperature, relative humidity, and net solar radiation have shown to affect the yield of oil palm fresh fruit bunches. 
<br><br>
&emsp;&emsp;The largest derivatives market for CPO is Bursa Malaysian Derivatives. The prices are affected mainly by demand and supply of CPO, and other external factors such as the prices of cooking oil and oil seeds all around the world. Climate variability have a direct impact on the supply of CPO, thus indirectly affect the price of CPO futures. Since Indonesia and Malaysia already have 87% of the production market cap, any climate changes in these two neighbouring countries would surely have some sort of impact on the CPO yield, and indirectly so on the CPO futures. 

![OilPalmPlantationMap](https://raw.githubusercontent.com/oryzalee8871/WQD7005_DataMining/master/A_Raw_Data/GlobalForestWatchDataset/Map/PalmOilPlantationMap.PNG)
Source: [https://www.globalforestwatch.org/](https://www.globalforestwatch.org/)

## Objectives
The primary goal of this project is to test the following hypothesis
<li> Weather data significantly affect / correlate to palm oil production yield
<li> Weather data significantly affect / correlate to CPO commodity prices
<li> Palm oil industry related news sentiment significantly affect / correlate to CPO commodity prices


## Milestone 1 - Data Aquisition & Web Crawling
For this project, the important datas are CPO prices, weather data and related news. Selenium were used for web scraping weather data, CPO prices and company stock prices. Scrappy was used to web scrap related palm oil industry news from a reputable news site. A few other non-significant data were acquired thru means of API and web download.

#### Main Data Sources:
<li> Daily Trading Days CPO Prices on Bursa Malaysia market and MCX India market. Data Source: https://investing.com
<li> Daily Monthly Data including temperature of mean, max and min variants, relative humidity and total rainfall. The data encompassed over 80 locations in Malaysia and Indonesia with data starting as early as 2010 to 2019. Data Source: https://en.tutiempo.net
<li> Monthly Malaysia CPO production from 2014 to 2020. Data Source: http://mpoc.org.my/monthly-palm-oil-trade-statistics-2014/
<li> Daily Palm oil industry related news title from year 2009 to 2020. Data Source: https://www.theedgemarkets.com

#### Additional data to explore:
<li> Stock prices of top five major palm oil key compananies. Data Source: https://investing.com
<li> Daily weather data from DarkSky API with more recent dates data (from 2018 onwards) with daily sunshine time and uv index. Data Source: https://darksky.net
<li> Land use datasets on Oil_palm_concessions, Palm_Oil_Mills, RSPO_mills , RSPOcertified_oil_palm_supply-bases_in_Indonesia and Sarawak_oil_palm_concessions. Data Source: https://data.globalforestwatch.org
    
## Milestone 2 - DataWarehouse & DataLake Implementation
In Milestone 2 we attempted two different file storage system, one using the well establish Hadoop HDFS file system with HIVE, and second using S3 file storage system using MinIO with AWS EC2.

#### 1. Store Data into Hive DataWarehouse
<li>(1) Download Horton Sandbox in this link https://www.cloudera.com/downloads/hortonworks-sandbox/hdp.html
<li>(2) Open the Horton Sandbox docker in Oracle VirtualBox
<li>(3) Make sure to have a least 8GB Free RAM on your PC
<li>(4) Sign in with default username "root" and password "hadoop", then change your password later on
<li>(5) open Sandbox terminal with local host port 4200
<li>(6) copy files from local machine to Sandbox using command [scp -P 2222 'DataSetPath' root@localhost:/root]
<li>(7) Push Dataset from sandbox to Hadoop File System (HDFS) using command [hdfs dfs -put 'Sandbox DataSetPath' 'HDFS Path']
<li>(8) Create Table in hive using below command:
<li>    CREATE TABLE YourTableName (
<li>    columnName1 STRING, columnName2 INT, etc ) 
<li>    ROW FORMAT DELIMITED 
<li>    FIELDS TERMINATED BY ','
<li>    LOCATION 'HDFS Path which store the dataset';
<li>(9) Check your data in table with SQL Query [SELECT * FROM YourTableName LIMIT 10] to check first 10th rows    


#### 2. Setup Datalake on the cloud using MinIO and AWS EC2
<li>(1) Create an EC2 ubuntu instance on AWS, and allow port forwarding on port 9000.
<li>(2) Setup credential on local machine to ssh into EC2 instance.
<li>(3) Setup standalone MinIO following quickstart guide from their documentation page: https://docs.min.io/
<li>(3) Use MinIO SDK for python to interact with MinIO datalake for bucket creation, data uploads and so on.
<li>(4) Use AWS S3 SDK for python to query data directly from MinIO datalake using SQL statement and output into dataframe by using Panda

    
## YouTube Links
Documentation of the processes and tools are done thru presentation on video recording, then uploaded to youtube
<li> Milestone1A & 1B - Web Scrapping https://youtu.be/m7Lqda_E3Fg
<li> Milestone2A - Store Data into Hive Data warehouse  https://youtu.be/UUWTioegn8M
<li> Milestone2B - Setup Datalake on the cloud using MinIO and AWS EC2  https://youtu.be/If27_zNYkx8
    
    
## References 
<a id="1">[1]</a> 
Santhia, V. (2017). Essential Palm Oil Statistics 2017. Palm Oil Analytics.
<br>
<a id="2">[2]</a> 
Oettli, P., Behera, S. K., & Yamagata, T. (2018). Climate based predictability of oil palm tree yield in Malaysia. Scientific reports, 8(1), 1-13.
<br>
<a id="3">[3]</a> 
Woittiez, L. S., van Wijk, M. T., Slingerland, M., van Noordwijk, M., & Giller, K. E. (2017). Yield gaps in oil palm: A quantitative review of contributing factors. European Journal of Agronomy, 83, 57-77.
<br>
<a id="4">[4]</a> 
Kamil, N. N., & Omar, S. F. (2016). Climate variability and its impact on the palm oil industry. Oil Palm Ind Econ J, 16(1), 18-30. 

 
