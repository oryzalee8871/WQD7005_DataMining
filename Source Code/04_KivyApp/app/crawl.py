import os
import tqdm
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from popuphelper import popup_window, PopupWindow
from kivy.uix.screenmanager import ScreenManager

SCRIPT_PATH = os.path.abspath(__file__)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/32.0.1700.107 Safari/537.36' }
options = Options()
options.add_argument("--headless")
options.add_argument('window-size=1920x1080')
options.add_argument("--start-maximized")

def check_url(url):
    '''
    Validate url, return True for successful request attempt.
    '''
    status = requests.get(url, headers=headers).status_code
    status = True if status ==200 else False
    return status

def check_date(datestr):
    ''' Validate url, popup window if invalid'''
    try: 
        datetime.strptime(datestr, "%m/%d/%Y")
        return True
    except:
        return False

def crawl(category, security_name, start_date, end_date):
    '''
    This function crawl company stock prices for investing.com. The company name 
    exist within the investing.com historical price url. 

    params:
    security_name: str
        security_name for which the stock prices is to be crawled
    start_date: str
        starting date for the stock prices in 'MM/DD/YYYY' format
    end_date: str
        end date for the stock prices in 'MM/DD/YYYY' format
    '''
    url = f'https://www.investing.com/{category}/{security_name}-historical-data'
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # This will wait till page is loaded for the specified element id
    WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "widgetFieldDateRange"))
        )

    driver.find_element_by_id('widgetFieldDateRange').click()

    WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "startDate"))
        )
    date_start = driver.find_element_by_id('startDate')
    date_start.clear()
    date_start.send_keys(start_date)

    date_end = driver.find_element_by_id('endDate')
    date_end.clear()
    date_end.send_keys(end_date)
    driver.find_element_by_id('ui-datepicker-div').find_element_by_id('applyBtn').click()

    # This will wait till page is loaded for the specified element id
    driver.implicitly_wait(3)
    WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "curr_table"))
        )

    table = driver.find_element_by_id('curr_table')

    col_names = [th.text for th in table.find_elements_by_tag_name('th')]
    data = np.asarray([[td.text for td in row.find_elements_by_tag_name('td')] for row in tqdm.tqdm(table.find_elements_by_tag_name('tr')[1:])])
    if data[0][0] == 'No results found':
        DF = None
    else:
        DF = pd.DataFrame(data, columns=col_names)
        DF['Date'] = pd.to_datetime(DF['Date'])
        DF['Price'].str.replace(',','').astype('float')
    driver.close()
    return DF



def preprocess(dataframe):
    dataframe['Price'] = dataframe['Price'].str.replace(',','').astype(float)
    dataframe['Open'] = dataframe['Open'].str.replace(',','').astype(float)
    dataframe['High'] = dataframe['High'].str.replace(',','').astype(float)
    dataframe['Low'] = dataframe['Low'].str.replace(',','').astype(float)
    dataframe['Change %'] = dataframe['Change %'].str.strip('%').astype(float)
    dataframe['Vol.'] = dataframe['Vol.'].str.replace(r'[A-Z]','').replace('-','0').astype(float)
    dataframe.rename(columns={'Price':'Close', 'Vol.': 'Volume', 'Change %': 'Change (%)'}, inplace=True)
    dataframe.set_index('Date', inplace=True)
    dataframe = dataframe[['Open', 'High', 'Low', 'Close', 'Volume']].sort_index(ascending=True)
    return dataframe

def get_crawl_data(category, security_name, start_date, end_date):
    '''
    This function handle the logic to retrieved crawled data, 
    or to crawl more data and store them based on the existence of
    requested data

    params:
    security_name: str
        security_name for which the stock prices is to be crawled
    start_date: str
        starting date for the stock prices in 'MM/DD/YYYY' format
    end_date: str
        end date for the stock prices in 'MM/DD/YYYY' format
    '''
    filelist_array = np.asarray(
        [np.array(file.strip('.csv').split('_') + [os.path.join(DATA_DIR,file)])
        for file in os.listdir(DATA_DIR) if file.endswith('csv')])
    index = np.where(np.asarray(filelist_array)==security_name)[0]

    if len(index) > 0:
        index = index[0]
        start_date_dt = datetime.strptime(start_date, "%m/%d/%Y")
        file_start_date_dt = datetime.strptime(filelist_array[index,2], "%m%d%Y")
        end_date_dt = datetime.strptime(end_date, "%m/%d/%Y")
        file_end_date_dt = datetime.strptime(filelist_array[index,3], "%m%d%Y")
        if start_date_dt < file_start_date_dt:
            df = crawl(category, security_name, start_date, end_date)
            if df is not None:
                df = preprocess(df)
                sd = start_date.replace('/', '')
                ed = datetime.strftime(df.index[-1], '%m%d%Y')

                csv_path = os.path.join(DATA_DIR, f'investing_{security_name}_{sd}_{ed}.csv')
                df.to_csv(csv_path, index=True)
                os.remove(filelist_array[index,4])
            else:
                df = pd.read_csv(filelist_array[index,4], index_col=0, parse_dates=True)
                df = df[df.index >= start_date_dt]
        elif end_date_dt > file_end_date_dt:
            crawl_start_date = datetime.strftime(file_end_date_dt+timedelta(1), '%m/%d/%Y')
            df = crawl(category, security_name, crawl_start_date, end_date)
            if df is not None:
                df = preprocess(df)
                sd = filelist_array[index,2]
                ed = datetime.strftime(df.index[-1], '%m%d%Y')
                ori_df = pd.read_csv(filelist_array[index,4], index_col=0, parse_dates=True)
                df = pd.concat([ori_df, df], axis=0)
                csv_path = os.path.join(DATA_DIR, f'investing_{security_name}_{sd}_{ed}.csv')
                df.to_csv(csv_path, index=True)
                os.remove(filelist_array[index,4])
                df = df[df.index >= start_date_dt]
            else:
                df = pd.read_csv(filelist_array[index,4], index_col=0, parse_dates=True)
                df = df[df.index >= start_date_dt]
        else:
            df = pd.read_csv(filelist_array[index,4], index_col=0, parse_dates=True)
            df = df[(df.index >= start_date_dt) & (df.index <=end_date_dt)]
    else:
        df = crawl(category, security_name, start_date, end_date)
        if df is not None:
            df = preprocess(df)
            sd = start_date.replace('/', '')
            ed = datetime.strftime(df.index[-1], '%m%d%Y')

            csv_path = os.path.join(DATA_DIR, f'investing_{security_name}_{sd}_{ed}.csv')
            df.to_csv(csv_path, index=True)
    return df



# if __name__ == '__main__':
    
#     start_date = '01/01/2019'
#     end_date = '01/01/2020'
#     security_name = 'sime-darby-plantation'
#     df = get_crawl_data(security_name, start_date, end_date)
