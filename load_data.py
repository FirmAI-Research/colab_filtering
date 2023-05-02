# Path: stock_colab_filter/matrix_factorize_v2.ipynb
import pandas as pd
import random
import argparse

#1 read the sf3 file data
# Read SF3 table from CSV file
#sf3 = pd.read_csv('data/SHARADAR_holdings.csv')
#sf3.head()

def get_sf3_dims(df):
    """
    inputs:
        df: dataframe
    outputs:
        ticker_list: list of tickers
        date_list: list of dates
        investorname_name_list: list of investor names
    """
    ticker_list = df.ticker.unique().tolist()
    date_list = df.calendardate.unique().tolist()
    investorname_name_list = df.investorname.unique().tolist()
    return ticker_list, date_list, investorname_name_list

def print_sf3_dims(dim_list, name):
    """
    inputs:
        dim_list: list of tickers, dates, or investor names
    outputs:
        None
    """
    print(f'len({name})={len(dim_list)}')
    print(f'{name}[0]={dim_list[0]}')
    print(f'{name}[1]={dim_list[1]}')
    print(f'{name}[2]={dim_list[2]}')
    print(f'{name}[-1]={dim_list[-1]}')


def create_small_debug_sf3(sf3, num_tickers, num_dates, num_investor_names):
    """
    inputs:
        sf3: dataframe
    outputs:
        sf3_small: dataframe
        sf3_small is the dataset where the rows involvethe first num_tickers, num_dates, num_investor_names from sf3 dataframe
    """

    ticker_list, date_list, investorname_name_list = get_sf3_dims(sf3)
    ticker_list_small = ticker_list[:num_tickers]
    date_list_small = date_list[:num_dates]
    investorname_name_list_small = investorname_name_list[:num_investor_names]
    sf3_small = sf3[sf3.ticker.isin(ticker_list_small)]
    sf3_small = sf3_small[sf3_small.calendardate.isin(date_list_small)]
    sf3_small = sf3_small[sf3_small.investorname.isin(investorname_name_list_small)]
    return sf3_small

def save_df_to_csv(sf3_small,filename):
    """
    inputs:
        sf3_small: dataframe
    outputs:
        None
    """
    sf3_small.to_csv(filename, index=False)

def split_train_val_test(sf3_small, train_frac, val_frac, test_frac, date_list):
    """
    
    inputs:
        sf3_small: dataframe
        train_frac: float
        val_frac: float
        test_frac: float
    outputs:
        sf3_train: dataframe
        sf3_val: dataframe
        sf3_test: dataframe
        split the sf3_small dataframe into train, val, and test dataframes
        the split is done based on the date_list
        first split the date_list into train val and test parts (with train being the earliest dates, etc
        then split the sf3_small dataframe into train val and test dataframes based on the date_list splits
    """

    # split the date_list into train val and test parts (with train being the earliest dates, etc
    num_dates = len(date_list)
    num_train_dates = int(train_frac*num_dates)
    num_val_dates = int(val_frac*num_dates)
    num_test_dates = int(test_frac*num_dates)
    date_list_train = date_list[:num_train_dates]
    date_list_val = date_list[num_train_dates:num_train_dates+num_val_dates]
    date_list_test = date_list[num_train_dates+num_val_dates:]
    # split the sf3_small dataframe into train val and test dataframes based on the date_list splits
    sf3_train = sf3_small[sf3_small.calendardate.isin(date_list_train)]
    sf3_val = sf3_small[sf3_small.calendardate.isin(date_list_val)]
    sf3_test = sf3_small[sf3_small.calendardate.isin(date_list_test)]
    return sf3_train, sf3_val, sf3_test    



    

    






def main(config):
    pass
    if(config.debug):
        print("main: start of function")
    sf3 = pd.read_csv('data/SHARADAR_holdings.csv')
    if(config.debug):
        print(f'sf3.head()={sf3.head()}')
    sf3['calendardate'] = pd.to_datetime(sf3['calendardate'])
    ticker_list, date_list, investorname_name_list = get_sf3_dims(sf3)
    if(config.debug):
        print_sf3_dims(ticker_list,"ticker_list")
        print_sf3_dims(date_list,"date_list")
        print_sf3_dims(investorname_name_list,"investorname_name_list")
    sf3_small = create_small_debug_sf3(sf3,3,len(date_list), 3)
    if(config.debug):
        print(f'sf3_small.head()={sf3_small.head()}')
        print (f'sf3_small.shape={sf3_small.shape}')
    save_df_to_csv(sf3_small,'data/SHARADAR_holdings_small.csv')

    sf3_train, sf3_val, sf3_test = split_train_val_test(sf3_small, 0.8, 0.1, 0.1, date_list)
    if(config.debug):
        print(f'sf3_train.head()={sf3_train.head()}')
        print (f'sf3_train.shape={sf3_train.shape}')
        print(f'sf3_val.head()={sf3_val.head()}')
        print (f'sf3_val.shape={sf3_val.shape}')
        print(f'sf3_test.head()={sf3_test.head()}')
        print (f'sf3_test.shape={sf3_test.shape}')
    save_df_to_csv(sf3_train,'data/SHARADAR_holdings_train.csv')
    save_df_to_csv(sf3_val,'data/SHARADAR_holdings_val.csv')
    save_df_to_csv(sf3_test,'data/SHARADAR_holdings_test.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--num_shot", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--meta_batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_steps", type=int, default=25000)
    parser.add_argument("--image_caching", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--debug", type=str, default=False)
    parser.add_argument('--cache', action='store_true')

    args = parser.parse_args()

    print (f'outside the main functtion: args: {args}')
    main(args)