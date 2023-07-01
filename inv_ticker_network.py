import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from sklearn.model_selection import train_test_split
import math
import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import scipy

class DataSetDescription:
    def __init__(self, df):
        self.df = df
        #run get_data_from_df to get the dictionaries
        self.get_data_from_df(df)
        #self.ticker_list, self.investor_list, self.calendar_list, self.ticker2idx, self.investor2idx, self.date2idx, self.idx2ticker, self.idx2investor, self.idx2date = DataSetDescription.get_data_from_df(df)
        
    """
    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        row = self.df.iloc[idx]
        investor = row['investorname']
        stock = row['ticker']
        date = row['calendardate']
        target = row['target']
        return investor, stock, date, target
    """

    #@staticmethod
    def get_data_from_df(self,df):
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
            ticker_date2idx: dictionary of (ticker, date) to embedding idx
            inv_date2idx: dictionary of (investor, date) to embedding idx
            idx2ticker: dictionary of embedding idx to tickers
            idx2investor: dictionary of embedding idx to investors
            idx2date: dictionary of embedding idx to dates
            idx2ticker_date: dictionary of embedding idx to (ticker, date)
            idx2inv_date: dictionary of embedding idx to (investor, date)

            

        """
        # Get the unique tickers, investors, and calendar dates
        self.ticker_list = df['ticker'].unique()
        self.ticker_list.sort()
        self.investor_list = df['investorname'].unique()
        self.calendar_list = df['calendardate'].unique()
        self.ticker2idx = {ticker: idx for idx, ticker in enumerate(self.ticker_list)}
        self.investor2idx = {investor: idx for idx, investor in enumerate(self.investor_list)}
        self.date2idx = {date: idx for idx, date in enumerate(self.calendar_list)}
        self.idx2ticker = {idx: ticker for idx, ticker in enumerate(self.ticker_list)}
        self.idx2investor = {idx: investor for idx, investor in enumerate(self.investor_list)}
        self.idx2date = {idx: date for idx, date in enumerate(self.calendar_list)}
        self.ticker_date2idx = {(ticker, date): idx for idx, (ticker, date) in enumerate((ticker,date) for ticker in self.ticker_list for date in self.calendar_list)}
        self.idx2ticker_date = {idx: (ticker, date) for idx, (ticker, date) in enumerate((ticker,date) for ticker in self.ticker_list for date in self.calendar_list)}
        self.inv_date2idx = {(investor, date): idx for idx, (investor, date) in enumerate((investor,date) for investor in self.investor_list for date in self.calendar_list)}
        self.idx2inv_date = {idx: (investor, date) for idx, (investor, date) in enumerate((investor,date) for investor in self.investor_list for date in self.calendar_list)}
        return_tuple=(self.ticker_list, self.investor_list, self.calendar_list, self.ticker2idx, self.investor2idx, self.date2idx, self.ticker_date2idx, self.inv_date2idx, self.idx2ticker, self.idx2investor, self.idx2date, self.idx2ticker_date, self.idx2inv_date)
        return return_tuple




    def summary(self):
        print(f'Investor Ownership Dataset: rows: {len(self.df)}, num_tickers: {len(self.ticker_list)}, num_investors: {len(self.investor_list)}, num_dates: {len(self.calendar_list)}')
    __repr__ = __str__ = lambda self: '{}(len:{}/ticker:{}/inve:{}/qtrs:{})'.format(self.__class__.__name__, len(self.df), len(self.ticker_list), len(self.investor_list), len(self.calendar_list))


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


class TorchInvDataset(Dataset):
    def __init__(self, df, dfDescription):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row=row.to_dict()
        investor_idx=self.dfDescription.investor2idx[row['investorname']]
        stock_idx=self.dfDescription.ticker2idx[row['ticker']]
        date_idx=self.dfDescription.date2idx[row['calendardate']]
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

class NoLookupDataSet(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row=row.to_dict()
        investor_idx=torch.tensor(row['investorname_idx'], dtype=torch.long)
        stock_idx=torch.tensor(row['ticker_idx'], dtype=torch.long)
        date_idx=torch.tensor(row['calendardate_idx'], dtype=torch.long)
        ticker_date2idx=torch.tensor(row['ticker_date_idx'], dtype=torch.long)
        target = torch.tensor(float(row['value']), dtype=torch.float)
        #if(math.fabs(target)>.2):
        #    target=torch.tensor(0.0, dtype=torch.float)
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
# this one will use a combined ticker_date embedding
class MatrixFactorization(nn.Module):
    def __init__(self, num_investors, num_tickers, num_dates, num_factors=10):
        super().__init__()
        self.investor_factors = nn.Embedding(num_investors, num_factors)
        #self.ticker_factors = nn.Embedding(num_tickers, num_factors)
        #date_factors not used now
        #self.date_factors = nn.Embedding(num_dates, num_factors)
        self.ticker_date_factors = nn.Embedding(num_tickers*num_dates, num_factors)
        #define bias just in case but it may not be useable or maybe bias has to be a function of time)
        #self.investor_bias = nn.Embedding(num_investors, 1)
        #self.ticker_bias = nn.Embedding(num_tickers, 1)
        #self.date_bias = nn.Embedding(num_dates, 1)
        #self.ticker_date_bias = nn.Embedding(num_tickers*num_dates, 1)
        initrange = 1 / np.log(num_factors)
        init.uniform_(self.investor_factors.weight, -initrange, initrange)
        init.uniform_(self.ticker_date_factors.weight, -initrange, initrange)

    def forward(self, investor, ticker, date, ticker_date):
        investor_factor = self.investor_factors(investor)
        #ticker_factor = self.ticker_factors(ticker)
        #date_factor = self.date_factors(date)
        ticker_date_factor = self.ticker_date_factors(ticker_date)
        #biases not used
        #investor_bias = self.investor_bias(investor).squeeze()
        #ticker_bias = self.ticker_bias(ticker).squeeze()
        #date_bias = self.date_bias(date).squeeze()
        dot = torch.sum(investor_factor * ticker_date_factor, dim=1)
        #result=dot + investor_bias + ticker_bias + date_bias + self.global_bias
        result=dot
        return result


def create_indexed_df(df,df_description,remove_str_columns=False):
    """
    create a new dataframe with indeces into the embedding dimension instead of strings
    inputs:
        df: dataframe with the following columns
            'ticker'
            'investorname'
            'calendardate'
            ...
        df_descriptions:
            contains indeces that convert from strings and tuples into integers
    outputs:
        df_result:
            contains the same columns as df if remove_str_columns==false, otherwise same columns except for the ticker,investorname and calendardate columns
            contains the following additional integer columns:
                'ticker_idx'
                'investorname_idx'
                'calendardate_idx'
                'ticker_date_idx'
                'investorname_date_idx'

    """
    df_result=df.copy()
    new_column_names=['ticker_idx','investorname_idx','calendardate_idx','ticker_date_idx','investorname_date_idx']

    #create the new columns with integer types
    for column_name in new_column_names:
        df_result[column_name]=0
    

    #go through all the rows in the dataframe and fill in the index columns form the df_description class
    #for 
    """
    for ind,row in df_result.iterrows():

        #print(f'row={row}')
        row['ticker_idx']=df_description.ticker2idx[row['ticker']]
        row['investorname_idx']=df_description.investor2idx[row['investorname']]
        row['calendardate_idx']=df_description.date2idx[row['calendardate']]
        row['ticker_date_idx']=df_description.ticker_date2idx[(row['ticker'],row['calendardate'])]
        row['investorname_date_idx']=df_description.inv_date2idx[(row['investorname'],row['calendardate'])]
        df_result.loc[ind]=row
        #print(f'row={row}')
    """
    df_result['ticker_idx']=df_result['ticker'].apply(lambda x: df_description.ticker2idx[x])
    df_result['investorname_idx']=df_result['investorname'].apply(lambda x: df_description.investor2idx[x])
    df_result['calendardate_idx']=df_result['calendardate'].apply(lambda x: df_description.date2idx[x])
    df_result['ticker_date_idx']=df_result.apply(lambda x: df_description.ticker_date2idx[(x['ticker'],x['calendardate'])],axis=1)
    df_result['investorname_date_idx']=df_result.apply(lambda x: df_description.inv_date2idx[(x['investorname'],x['calendardate'])],axis=1)

    if remove_str_columns:
        df_result=df_result.drop(columns=['ticker','investorname','calendardate'])
        
    #print(df_result.head())
    return df_result

def main(config):
    relational_df_path=config.data_file
    num_epochs=config.num_epochs
    batch_size=config.batch_size
    learning_rate=config.learning_rate
    num_factors=config.num_factors

    use_gpu = False
    if config.device == "gpu" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        use_gpu = True
    elif config.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        use_gpu = True
    else:
        device = torch.device("cpu")
    print("Using device: ", device)

    sf3 = pd.read_csv(relational_df_path)
    sf3 = sf3.dropna()
    #run the code to process the dataframe
    #ticker_list, investor_list, calendar_list, ticker2idx, investor2idx, date2idx, idx2ticker, idx2investor, idx2date = get_data_from_df(sf3)
    
    #create an InvOwnershipDataset object
    #sf3_description = SF3Dataset(sf3, investor2idx, ticker2idx, date2idx)
    sf3_description = DataSetDescription(sf3)
    sf3_description.summary()
    df_indexed=create_indexed_df(sf3,sf3_description)
    #split into validation and training sets
    if config.test_run:
        df_indexed=df_indexed.head(1000)
    print(f'df_indexed.shape before filtering = {df_indexed.shape}')
    df_indexed=df_indexed[(df_indexed['value']>-config.remove_threshold) & (df_indexed['value']<config.remove_threshold)]
    print(f'df_indexed.shape after filtering = {df_indexed.shape}')
    #Scale by 100 to make numvers more reasonable?
    df_indexed['value']=df_indexed['value'].astype(float)
    mu=np.mean(df_indexed['value'])
    std1=np.std(df_indexed['value'])
    all_values=df_indexed['value'].copy()
    cdfs=scipy.stats.lognorm.cdf(all_values,s=std1,loc=mu,scale=1)
    print(f'all_values.shape={all_values.shape}')
    print(f'cdfs.shape={cdfs.shape}')
    #put all_values and cdfs next to each other and print
    df_temp=pd.DataFrame({'value':all_values,'cdf':cdfs})
    print(df_temp.head(1000))
    df_indexed['value'] = scipy.stats.lognorm(s=std1, scale=np.exp(0)).cdf(1+df_indexed['value'])
    temp=scipy.stats.lognorm(s=std1, scale=np.exp(0)).cdf(df_indexed['value'])
    print(f'tem[0:10]={temp[0:10]}')
    print("df_index transformed ={0}".format(df_indexed[['ticker','investorname','calendardate']].head()))
    mu=np.mean(df_indexed['value'])
    std=np.std(df_indexed['value'])
    df_indexed['value']=(df_indexed['value']-mu)/std
    print(f'df_indexed.shape after filtering = {df_indexed.shape}')
    #check the mean square value of df_indexed['value']
    print(f'mean square value of df_indexed["value"] = {np.sqrt(np.mean(df_indexed["value"]**2))}')
    print (f'mean value of df_indexed["value"] = {np.mean(df_indexed["value"])}')
    #df_indexed['value'] = (df_indexed['value'] - np.mean(df_indexed['value'])) / (np.std(df_indexed['value']) + 1e-6)
    print(f'rescaling value: initial std={std1} mu={mu},std={std}')

    #df_indexed=df_indexed[np.abs(df_indexed['value'])>0.0001]
    df_train, df_validate=train_test_split(df_indexed,test_size=0.2,random_state=42)
    print(f'df_train.shape={df_train.shape}')
    print(f'{df_train[["ticker","investorname","calendardate","value"]].head()}')
    std_train=np.std(df_train['value'])
    print(f'std_train={std_train}')
    df_train.to_csv('df_train.csv')
    ds = NoLookupDataSet(df_train)
    ds_validate=NoLookupDataSet(df_validate)
    loader=DataLoader(ds,batch_size=config.batch_size,shuffle=True)
    loader_validate=DataLoader(ds_validate,batch_size=config.batch_size,shuffle=False)

    #create the model

    #create the model
    num_investors = len(sf3_description.investor2idx)
    num_tickers = len(sf3_description.ticker2idx)
    num_dates = len(sf3_description.date2idx)
    num_ticker_dates = len(sf3_description.ticker_date2idx)
    num_investor_dates = len(sf3_description.inv_date2idx)
    model = MatrixFactorization(num_investors, num_tickers, num_dates,  num_factors)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    writer = SummaryWriter(config.logdir)
    total_loss=0
    total_ownership=0
    num_batches_validate=len(loader_validate)
    num_batches=len(loader)
    total_ownership=0
    #create a tensor to hold the ownership values, then keep concatenating batch ownership values to it
    ownership_tensor=torch.tensor([])
    model.to(device)
    ownership_tensor=ownership_tensor.to(device)

    for (batch_idx, batch) in enumerate(loader):
        batch = [b.to(device) for b in batch]
        (investor_idx, ticker_idx, date_idx, ticker_date_idx, ownership) = batch
        ownership=ownership.float()
        ownership_pred = model(investor_idx, ticker_idx, date_idx, ticker_date_idx)
        loss = loss_function(ownership_pred, ownership)
        ownership_tensor=torch.cat((ownership_tensor,ownership.detach().flatten()))
        total_ownership+=(ownership**2).sum()/ownership.shape[0]/num_batches
        total_loss += loss.item()/num_batches
    print(f'initial parameters loss: loss={total_loss} ownership={torch.sqrt(torch.mean(ownership_tensor**2))}')
    print(f'mean of ownership={torch.mean(ownership_tensor)}')
    print(f'std of ownership={torch.std(ownership_tensor)}')


    for epoch in range(num_epochs):
        total_loss = 0
        #print(f'epoch={epoch}')
        num_batches=len(loader)
        if (epoch==0):
            print("num_batches={0}".format(num_batches))
        for (batch_idx, batch) in enumerate(loader):
            batch=[b.to(device) for b in batch]
            (investor_idx, ticker_idx, date_idx, ticker_date_idx, ownership) = batch
            model.zero_grad()
            ownership_pred = model(investor_idx, ticker_idx, date_idx, ticker_date_idx)
            loss = loss_function(ownership_pred, ownership)
            if math.isnan(loss.item()) or math.isinf(loss.item()):
                print(f'loss is nan for batch_idx={batch_idx}')
                print(f'target={ownership}')
                print(f'ownership_pred={ownership_pred}')
                print(f'investor_idx={investor_idx}')
                print(f'ticker_idx={ticker_idx}')
                print(f'date_idx={date_idx}')
                print(f'ticker_date_idx={ticker_date_idx}')
                print(f'model.investor_factors={model.investor_factors(investor_idx)}')
                print(f'model.ticker_date_factors={model.ticker_date_factors(ticker_date_idx)}')
                break
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()/num_batches
        print(f'epoch={epoch} total_loss={total_loss}')
        #if batch_idx % 100 == 0:
            #print(f'batch_idx={batch_idx}/{num_batches} loss={loss.item()}')
        #run the validation
        if(epoch % 5 == 0):
            total_loss_validate=0
            num_batches_validate=len(loader_validate)
            for (batch_idx, batch) in enumerate(loader_validate):
                batch=[b.to(device) for b in batch]
                (investor_idx, ticker_idx, date_idx, ticker_date_idx, ownership) = batch
                ownership_pred = model(investor_idx, ticker_idx, date_idx, ticker_date_idx)
                loss = loss_function(ownership_pred, ownership)
                total_loss_validate += loss.cpu().item()/num_batches_validate
            print(f'epoch={epoch} validation loss={total_loss_validate}')
        writer.add_scalar('training/MSE', total_loss, epoch)
        writer.add_scalar('validation/MSE', total_loss_validate, epoch)

        investor_embedding_weights = model.investor_factors.weight.data.cpu().numpy()
        writer.add_embedding(investor_embedding_weights, metadata=sf3_description.investor2idx.keys(), tag='investor_factors',global_step=epoch)
        ticker_embedding_weights = model.ticker_date_factors.weight.data.cpu().numpy()
        writer.add_embedding(ticker_embedding_weights, metadata=sf3_description.ticker_date2idx.keys(), tag='ticker_date_factors',global_step=epoch)
        #writer.flush()
    #save the model
    torch.save(model.state_dict(), 'model.pth')





    # Get the unique tickers, investors, and calendar dates
    #ticker_list = sf3['ticker'].unique()
    #ticker_list.sort()
    #investor_list = sf3['investorname'].unique()
    #calendar_list = sf3['calendardate'].unique()

    dim_dates=sf3_description.calendar_list
    num_dates=len(dim_dates)
    dim_tickers=sf3_description.ticker_list
    num_tickers=len(dim_tickers)
    dim_investors=sf3_description.investor_list
    num_investors=len(dim_investors)


    # Print the number of unique tickers, investors, and calendar dates
    #print(f'sf3_description.summary()={sf3_description.summary()}')
    #sf3_description.summary()
    #print(f'sf_dataset:{sf3_description}')


    
if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_file', type=str, help='the path to the relational_df.csv file')
    parser.add_argument('--debug', type=bool, default=False, help='whether to run in debug mode')
    parser.add_argument('--num_epochs', type=int, default=50, help='the number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=1000, help='the batch size')
    parser.add_argument('--num_factors', type=int, default=1, help='the number of factors')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='the learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='the weight decay')
    parser.add_argument('--remove_str_columns', type=bool, default=True, help='whether to remove the string columns')
    parser.add_argument('--logdir', type=str, default='runs', help='the log directory')
    parser.add_argument('--test_run', type=bool, default=False, help='whether to run a test run')
    parser.add_argument('--remove_threshold', type=float, default=0.1, help='the threshold for removing rows')
    parser.add_argument('--device', type=str, default="cpu", help='the device to run on')
    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args)
    
