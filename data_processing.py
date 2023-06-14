import pandas as pd
import numpy as np
def load_full_data_set(file_name):
    print(f'Loading data from {file_name}...{file_name}')
    sf3 = pd.read_csv(file_name)
    sf3 = sf3.dropna()
    df=sf3

    df[['ticker', 'investorname', 'calendardate']] = df[['ticker', 'investorname', 'calendardate']].astype(str)
    df[['value', 'units', 'price']] = df[['value', 'units', 'price']].astype(float)
    df=df[df['securitytype']=='SHR']
    df=df[['ticker', 'investorname', 'calendardate', 'value', 'units', 'price']]
    return df

def get_dimensions(create_dimension_files,all_tickers_file_name,all_investors_file_name,all_dates_file_name,df):
    if create_dimension_files==True:
    # Get the number of unique tickers, investors, and dates
        num_tickers = df['ticker'].nunique()
        num_investors = df['investorname'].nunique()
        num_dates = df['calendardate'].nunique()
    # Create mappings from tickers, investors, and dates to indices
        ticker_to_index = {ticker: i for i, ticker in enumerate(df['ticker'].unique())}
        investor_to_index = {investor: i for i, investor in enumerate(df['investorname'].unique())}
        date_to_index = {date: i for i, date in enumerate(np.sort(df['calendardate'].unique()))}

    # Convert the mappings to DataFrames
        ticker_index_df = pd.DataFrame.from_dict(ticker_to_index, orient='index', columns=['index'])
        investor_index_df = pd.DataFrame.from_dict(investor_to_index, orient='index', columns=['index'])
        date_index_df = pd.DataFrame.from_dict(date_to_index, orient='index', columns=['index'])
    # Save the DataFrames to CSV files
        ticker_index_df.to_csv(all_tickers_file_name)
        investor_index_df.to_csv(all_investors_file_name)
        date_index_df.to_csv(all_dates_file_name)
    else:
    # Load the mappings from CSV files
        ticker_index_df = pd.read_csv(config.all_tickers_file_name, index_col=0)
        investor_index_df = pd.read_csv(config.all_investors_file_name, index_col=0)
        date_index_df = pd.read_csv(config.all_dates_file_name, index_col=0)
    # Convert the DataFrames to dictionaries
        ticker_to_index = ticker_index_df.to_dict()['index']
        investor_to_index = investor_index_df.to_dict()['index']
        date_to_index = date_index_df.to_dict()['index']
    return ticker_to_index, investor_to_index, date_to_index

#(ticker_to_index, investor_to_index, date_to_index)=get_dimensions(config,df)

def get_dense_values(df,ticker_to_index, investor_to_index, date_to_index):
    num_tickers = len(ticker_to_index)
    num_investors = len(investor_to_index)
    num_dates = len(date_to_index)


    # Create a 3D numpy array of the same shape
    array_3d = np.full((num_tickers, num_investors, num_dates), 0.0,dtype=np.float32)

    # Iterate over the DataFrame rows to fill in the values
    count=0
    max_count=15
    for idx, row in df.iterrows():
        array_3d[ticker_to_index[row['ticker']], investor_to_index[row['investorname']], date_to_index[row['calendardate']]] = row['value']
        #print(f'row["value"]={row["value"]}')
        ticker=row["ticker"]
        investorname=row['investorname']
        calenderdate=row["calendardate"]
        array_val=array_3d[ticker_to_index[row['ticker']], investor_to_index[row['investorname']], date_to_index[row['calendardate']]]
        #print(f'array_3d[{ticker}][{investorname}][{calenderdate}]={array_val}')
        count+=1
        #if (count>max_count):
        #    break
    return array_3d


if __name__ == '__main__':
    data_path = 'data/'
    holdings_file_name=data_path+"SHARADAR_holdings.csv"
    data = load_full_data_set(holdings_file_name)
    print(data.head())
