import argparse
import pandas as pd
import numpy as np
import load_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom dataset class
class SF3Dataset(Dataset):
    def __init__(self, sf3, investor2idx, ticker2idx, date2idx):
        self.sf3 = sf3
        self.investor2idx = investor2idx
        self.ticker2idx = ticker2idx
        self.date2idx = date2idx

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
        target = torch.tensor(float(row['value']), dtype=torch.float)
        #print(f'investor_idx={investor_idx}, stock_idx={stock_idx}, date_idx={date_idx}, target={target}')
        return investor_idx, stock_idx, date_idx, target
    


# Define the model
class MatrixFactorization(nn.Module):
    def __init__(self, num_investors, embedding_dim, num_tickers, num_dates, ticker_embedding_dim, date_embedding_dim):
        super().__init__()
        self.investor_embeddings = nn.Embedding(num_investors, embedding_dim)
        self.ticker_embeddings = nn.Embedding(num_tickers, ticker_embedding_dim)
        self.date_embeddings = nn.Embedding(num_dates, date_embedding_dim)
        
    def forward(self, investor_ids, ticker_ids, date_ids):
        investor_embeds = self.investor_embeddings(investor_ids)
        ticker_embeds = self.ticker_embeddings(ticker_ids)
        date_embeds = self.date_embeddings(date_ids)
        ticker_date_embeds = torch.cat([ticker_embeds, date_embeds], 1)
        return (investor_embeds * ticker_date_embeds).sum(1)

def main(config):
    print(config)
    sf3 = pd.read_csv(config.train_file_name)
    if(config.debug):
        print(f'sf3.head()={sf3.head()}')
    sf3['calendardate'] = pd.to_datetime(sf3['calendardate'])
    (ticker_list, date_list, investorname_list) = load_data.get_sf3_dims(sf3)
    if(config.debug):
        load_data.print_sf3_dims(ticker_list,"ticker_list")
        load_data.print_sf3_dims(date_list,"date_list")
        load_data.print_sf3_dims(investorname_list,"investorname_name_list")

    # Create dictionaries for mapping
    investor2idx = {investor: idx for idx, investor in enumerate(investorname_list)}
    ticker_date2idx = {(ticker, date): idx for idx, (ticker, date) in enumerate([(ticker, date) for ticker in ticker_list for date in date_list])}
    ticker2idx = {ticker: idx for idx, ticker in enumerate(ticker_list)}
    date2idx = {date: idx for idx, date in enumerate(date_list)}
    batch_size = config.batch_size

    sf3_dataset = SF3Dataset(sf3, investor2idx, ticker2idx, date2idx)
    data_loader = DataLoader(sf3_dataset, batch_size=batch_size, shuffle=True)

    if(config.debug):
        print(f'investor2idx={investor2idx}')
        print(f'ticker_date2idx={ticker_date2idx}')
        print(f'ticker2idx={ticker2idx}')
        print(f'date2idx={date2idx}')
    # Hyperparameters
    embedding_dim = 20
    ticker_embedding_dim = 17
    date_embedding_dim = 3
    learning_rate = 0.001
    num_epochs = 100

    model = MatrixFactorization(len(investorname_list), embedding_dim, len(ticker_list), len(date_list), ticker_embedding_dim, date_embedding_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (investor_indices, stock_indices, date_indices, targets) in enumerate(data_loader):
            if config.debug:
                print(f'investor_indices={investor_indices}')
                print(f'stock_indices={stock_indices}')
                print(f'date_indices={date_indices}')
                print(f'targets={targets}')
            # Convert lists to tensors
            #print(f'investor_indices={investor_indices}')
            #investor_indices = torch.tensor(investor_indices, dtype=torch.long)
            #stock_indices = torch.tensor(stock_indices, dtype=torch.long)
            #date_indices = torch.tensor(date_indices, dtype=torch.long)
            print(f'eopch={epoch}, batch_idx={batch_idx}, investor_indeces.shape={investor_indices.shape}')
            optimizer.zero_grad()
            predictions = model(investor_indices, stock_indices, date_indices)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(investorname_list)}")


    

  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--num_classes", type=int, default=5)
    #parser.add_argument("--num_shot", type=int, default=1)
    #parser.add_argument("--num_workers", type=int, default=4)
    #parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--train_file_name", type=str, default='data/SHARADAR_holdings_train.csv')
    parser.add_argument("--meta_batch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--random_seed", type=int, default=123)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--train_steps", type=int, default=25000)
    #parser.add_argument("--image_caching", type=bool, default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--debug", type=str, default=False)
    parser.add_argument('--cache', action='store_true')

    args = parser.parse_args()

    print (f'outside the main functtion: args: {args}')
    main(args)