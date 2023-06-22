import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

def main(relational_df_path):
    # Load the DataFrame from the CSV file
    sf3 = pd.read_csv(relational_df_path)
    sf3 = sf3.dropna()

    # Get the unique tickers, investors, and calendar dates
    ticker_list = sf3['ticker'].unique()
    ticker_list.sort()
    investor_list = sf3['investorname'].unique()
    calendar_list = sf3['calendardate'].unique()

    dim_dates=calendar_list
    num_dates=len(dim_dates)
    dim_tickers=ticker_list
    num_tickers=len(dim_tickers)
    dim_investors=investor_list
    num_investors=len(dim_investors)

    df_train,df_validate=split_train_test(sf3)

    # Print the number of unique tickers, investors, and calendar dates
    print(f'number of tickers: {len(ticker_list)} number of investors: {len(investor_list)} number of calendar dates: {len(calendar_list)}')


def split_train_test(df_train):
    # Split the DataFrame into training and test sets
    df_train2, df_test = train_test_split(df_train, test_size=0.2, random_state=42)
    return df_train2, df_test


if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_file', type=str, help='the path to the relational_df.csv file')
    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args.data_file)
