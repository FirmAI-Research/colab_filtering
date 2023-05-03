# Path: stock_colab_filter/matrix_factorize_v2.ipynb
import pandas as pd
import random
import argparse
import load_data

#1 read the sf3 file data
# Read SF3 table from CSV file
#sf3 = pd.read_csv('data/SHARADAR_holdings.csv')
#sf3.head()

def create_small_debug_sf3(sf3, num_tickers, num_dates, num_investor_names):
    """
    inputs:
        sf3: dataframe
    outputs:
        sf3_small: dataframe
        sf3_small is the dataset where the rows involvethe first num_tickers, num_dates, num_investor_names from sf3 dataframe
    """

    ticker_list, date_list, investorname_name_list = load_data.get_sf3_dims(sf3)
    ticker_list_small = ticker_list[:num_tickers]
    date_list_small = date_list[:num_dates]
    investorname_name_list_small = investorname_name_list[:num_investor_names]
    sf3_small = sf3[sf3.ticker.isin(ticker_list_small)]
    sf3_small = sf3_small[sf3_small.calendardate.isin(date_list_small)]
    sf3_small = sf3_small[sf3_small.investorname.isin(investorname_name_list_small)]
    return sf3_small



    






def main(config):
    pass
    if(config.debug):
        print("main: start of function")
    sf3 = pd.read_csv('data/SHARADAR_holdings.csv')
    if(config.debug):
        print(f'sf3.head()={sf3.head()}')
    sf3['calendardate'] = pd.to_datetime(sf3['calendardate'])
    ticker_list, date_list, investorname_name_list = load_data.get_sf3_dims(sf3)
    if(config.debug):
        load_data.print_sf3_dims(ticker_list,"ticker_list")
        load_data.print_sf3_dims(date_list,"date_list")
        load_data.print_sf3_dims(investorname_name_list,"investorname_name_list")
    #sf3_small = create_small_debug_sf3(sf3,3,len(date_list), 3)
    #parse arguments to get the number of tickers, dates, and investor names and file paths
    #create the filenames for the small dataset as well as for the train, val, and test datasets
    #the filenames should include the number of tickers, dates, and investor names

    filepath=config.file_path
    num_tickers = config.num_stocks
    num_dates = config.num_quarters
    if (config.use_all_quarters):
        num_dates = len(date_list)
    num_investor_names = config.num_investornames
    filename_small = f'{filepath}_small_t{num_tickers}_i{num_investor_names}_q{num_dates}'
    sf3_small = create_small_debug_sf3(sf3,num_tickers,num_dates, num_investor_names)

    if(config.debug):
        print(f'sf3_small.head()={sf3_small.head()}')
        print (f'sf3_small.shape={sf3_small.shape}')
    load_data.save_df_to_csv(sf3_small,f'{filename_small}_all.csv')

    sf3_train, sf3_val, sf3_test = load_data.split_train_val_test(sf3_small, 0.8, 0.1, 0.1, date_list)
    if(config.debug):
        print(f'sf3_train.head()={sf3_train.head()}')
        print (f'sf3_train.shape={sf3_train.shape}')
        print(f'sf3_val.head()={sf3_val.head()}')
        print (f'sf3_val.shape={sf3_val.shape}')
        print(f'sf3_test.head()={sf3_test.head()}')
        print (f'sf3_test.shape={sf3_test.shape}')
    load_data.save_df_to_csv(sf3_train,f'{filename_small}_train.csv')
    load_data.save_df_to_csv(sf3_val,f'{filename_small}_val.csv')
    load_data.save_df_to_csv(sf3_test,f'{filename_small}_test.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_stocks", type=int, default=100)
    parser.add_argument("--num_investornames", type=int, default=100)
    parser.add_argument("--num_quarters", type=int, default=4)
    parser.add_argument("--use_all_quarters", type=bool, default=True)
    parser.add_argument("--debug", type=str, default=False)
    parser.add_argument("--file_path", type=str, default='data/sf3_small')

    args = parser.parse_args()

    print (f'outside the main functtion: args: {args}')
    main(args)