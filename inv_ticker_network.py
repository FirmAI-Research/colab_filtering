import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class InvOwnershipDataSet:
    def __init__(self, df):
        self.df = df
        #run get_data_from_df to get the dictionaries
        self.ticker_list, self.investor_list, self.calendar_list, self.ticker2idx, self.investor2idx, self.date2idx, self.idx2ticker, self.idx2investor, self.idx2date = InvOwnershipDataSet.get_data_from_df(df)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        investor = row['investorname']
        stock = row['ticker']
        date = row['calendardate']
        target = row['target']
        return investor, stock, date, target
    @staticmethod
    def get_data_from_df(df):
        """
        Get the data from the dataframe
        inputs:
            df: dataframe
        outputs:
            ticker_list: list of tickers
            investor_list: list of investors:
            date_list: list of dates
            ticker2idx: dictionary of tickers to embedding idx
            investor2idx: dictionary of investors to embedding idx
            date2idx: dictionary of dates to embedding idx
            idx2ticker: dictionary of embedding idx to tickers
            idx2investor: dictionary of embedding idx to investors
            idx2date: dictionary of embedding idx to dates

        """
        # Get the unique tickers, investors, and calendar dates
        ticker_list = df['ticker'].unique()
        ticker_list.sort()
        investor_list = df['investorname'].unique()
        calendar_list = df['calendardate'].unique()
        ticker2idx = {ticker: idx for idx, ticker in enumerate(ticker_list)}
        investor2idx = {investor: idx for idx, investor in enumerate(investor_list)}
        date2idx = {date: idx for idx, date in enumerate(calendar_list)}
        idx2ticker = {idx: ticker for idx, ticker in enumerate(ticker_list)}
        idx2investor = {idx: investor for idx, investor in enumerate(investor_list)}
        idx2date = {idx: date for idx, date in enumerate(calendar_list)}
        return ticker_list, investor_list, calendar_list, ticker2idx, investor2idx, date2idx, idx2ticker, idx2investor, idx2date
    def summary(self):
        print(f'Investor Ownership Dataset: rows: {len(self.df)}, num_tickers: {len(self.ticker_list)}, num_investors: {len(self.investor_list)}, num_dates: {len(self.calendar_list)}')
    __repr__ = __str__ = lambda self: '{}({})'.format(self.__class__.__name__, len(self.df))


class SF3Dataset(Dataset):
    def __init__(self, sf3, investor2idx, ticker2idx, date2idx,ticker_date2idx):
        self.sf3 = sf3
        self.investor2idx = investor2idx
        self.ticker2idx = ticker2idx
        self.date2idx = date2idx
        self.ticker_date2idx = ticker_date2idx

    def __len__(self):
        return len(self.sf3)

    def __getitem__(self, idx):
        row = self.sf3.iloc[idx]
        #print(f'__getitem__ row={row}')
        row=row.to_dict()
        #print(f'__getitem__ row={row}')
        investor_idx = self.investor2idx[row['investorname']]
        #print (f'investor_idx={investor_idx}')
        stock_idx = self.ticker2idx[row['ticker']]
        date_idx = self.date2idx[row['calendardate']]
        ticker_date2idx=self.ticker_date2idx[(row['ticker'],row['calendardate'])]
        target = torch.tensor(float(row['value']), dtype=torch.float)
        #print(f'investor_idx={investor_idx}, stock_idx={stock_idx}, date_idx={date_idx}, target={target}')
        return investor_idx, stock_idx, date_idx, ticker_date2idx ,target
    

        # Custom collate_fn

        def sf3_collate_fn(batch):
            #print(f'batch={batch}')
            investor_list = []
            stock_list = []
            date_list = []
            target_list = []
            for investor, stock, date, target in batch:
                investor_list.append(investor)
                stock_list.append(stock)
                date_list.append(date)
                target_list.append(target)
            return torch.tensor(investor_list), torch.tensor(stock_list), torch.tensor(date_list), torch.tensor(target_list)

# Create the dataset and dataloader
class MatrixFactorization(nn.Module):
    def __init__(self, num_investors, num_tickers, num_dates, num_factors=10):
        super().__init__()
        self.investor_factors = nn.Embedding(num_investors, num_factors)
        self.ticker_factors = nn.Embedding(num_tickers, num_factors)
        self.date_factors = nn.Embedding(num_dates, num_factors)
        #define bias just in case but it may not be useable or maybe bias has to be a function of time)
        self.investor_bias = nn.Embedding(num_investors, 1)
        self.ticker_bias = nn.Embedding(num_tickers, 1)
        self.date_bias = nn.Embedding(num_dates, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, investor, ticker, date):
        investor_factor = self.investor_factors(investor)
        ticker_factor = self.ticker_factors(ticker)
        date_factor = self.date_factors(date)
        investor_bias = self.investor_bias(investor).squeeze()
        ticker_bias = self.ticker_bias(ticker).squeeze()
        date_bias = self.date_bias(date).squeeze()
        dot = torch.sum(investor_factor * ticker_factor * date_factor, dim=1)
        return dot + investor_bias + ticker_bias + date_bias + self.global_bias

def main(relational_df_path):
    sf3 = pd.read_csv(relational_df_path)
    sf3 = sf3.dropna()
    #run the code to process the dataframe
    #ticker_list, investor_list, calendar_list, ticker2idx, investor2idx, date2idx, idx2ticker, idx2investor, idx2date = get_data_from_df(sf3)
    
    #create an InvOwnershipDataset object
    #sf3_dataset = SF3Dataset(sf3, investor2idx, ticker2idx, date2idx)
    sf3_dataset = InvOwnershipDataSet(sf3)
    sf3_dataset.summary()


    # Get the unique tickers, investors, and calendar dates
    #ticker_list = sf3['ticker'].unique()
    #ticker_list.sort()
    #investor_list = sf3['investorname'].unique()
    #calendar_list = sf3['calendardate'].unique()

    dim_dates=calendar_list
    num_dates=len(dim_dates)
    dim_tickers=ticker_list
    num_tickers=len(dim_tickers)
    dim_investors=investor_list
    num_investors=len(dim_investors)


    # Print the number of unique tickers, investors, and calendar dates
    print(f'number of tickers: {len(ticker_list)} number of investors: {len(investor_list)} number of calendar dates: {len(calendar_list)}')


    
if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_file', type=str, help='the path to the relational_df.csv file')
    parser.add_argument('--debug', type=bool, default=False, help='whether to run in debug mode')
    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args.data_file)
    
