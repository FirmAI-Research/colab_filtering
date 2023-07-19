import numpy as np
import pandas as pd
import argparse

def generate_test_dimensions(sz):
    #sz is a tuple of (x,y,z)
    #x is the lenth of ticker dimension
    #y is the length of investorname dimension
    #z is the length of the calendardate dimension
    #retturn a tuple of numpy arrays of the dimensions

    #ticker dimension
    ticker = [f'T{i}' for i in range(sz[0])]
    ticker = np.array(ticker)
    investorname = [f'I{i}' for i in range(sz[1])]
    investorname = np.array(investorname)
    calendardate = [f'D{i}' for i in range(sz[2])]
    calendardate = np.array(calendardate)
    return (ticker, investorname, calendardate)

def generate_test_data(sz):
    #sz is a tuple of (x,y,z)
    #x is the lenth of ticker dimension
    #y is the length of investorname dimension
    #z is the length of the calendardate dimension
    #return a uniform 3d array of size (x,y,z), distributed between 0 and 1
    return np.random.uniform(size=sz)

    #ticker dimension
def remove_empty(data,x,y,z):
    #data is a 3d numpy array
    #x is the lenth of ticker dimension
    #y is the length of investorname dimension
    #z is the length of the calendardate dimension
    # remove all of the tickers that have all zeros for all investors, for any calendar date
    # remove all of the investornames that have all zeros for all tickers, for any calendar date
    # return a tuple of data, x,y,z with the empty tickers and investors removed
    

    ticker_sum=np.sum(data,axis=(1))
    investor_sum=np.sum(data,axis=(0))
    ticker_with_missing=np.any(ticker_sum==0,axis=1)
    investor_with_missing=np.any(investor_sum==0,axis=1)
    ticker_keep=np.logical_not(ticker_with_missing)
    investor_keep=np.logical_not(investor_with_missing)
    data=data[ticker_keep,:,:]
    data=data[:,investor_keep,:]
    x=x[ticker_keep]
    y=y[investor_keep]
    return (data,x,y,z)

def convert_to_relational(data,x,y,z):
    #data is a 3d numpy array   o
    #x is the ticker dimension
    #y is the investorname dimension
    #z is the calendardate dimension
    #create a dataframe with columns ['ticker','investorname','calendardate','value']
    #where value is the value of the data at that point


    x=np.array(list(x))
    y=np.array(list(y))
    z=np.array(list(z))

    ind=np.indices(data.shape)
    df=pd.DataFrame({'ticker':x[ind[0].flatten()],'investorname':y[ind[1].flatten()],'calendardate':z[ind[2].flatten()],'value':data.flatten()})
    #df['ticker']=x[ind[0].flatten()I]
    #df['investorname']=y[ind[1].flatten()]
    #df['calendardate']=z[ind[2].flatten()]
    #df['value']=data.flatten()
    print("converting the datqa")
    print(df.head())

    return df


def normalize_along_investorname(data):
    #data is a 3d numpy array
    #normalize the data along the investorname dimension
    #return the normalized data
    return data/(np.sum(data,axis=(0))+.0000000000001)

def compute_relative_change(data):
    #data is a 3d numpy array
    #compute the relative change in the data along the calendardate dimension
    #return the relative change data
    return data[:,:,1:]/data[:,:,:-1]-1


                                


def main(config):
    sz=(10,10,4)
    ticker, investorname, calendardate = generate_test_dimensions(sz)
    print(f'ticker: {ticker}')
    print(f'investorname: {investorname}')
    print(f'calendardate: {calendardate}')
    generated_data = generate_test_data(sz)
    print(f'generated_data: {generated_data}')
    generated_data[0,:,:]=0
    data,ticker,investorname,calendardate=remove_empty(generated_data,ticker,investorname,calendardate)
    print(f'data.shape: {data.shape}')
    print(f'ticker: {ticker}')
    print(f'investorname: {investorname}')
    print(f'calendardate: {calendardate}')

    df=convert_to_relational(data,ticker,investorname,calendardate)
    print(df.head())
    t_arr=np.ones(sz)
    t_arr=normalize_along_investorname(t_arr)
    df.to_csv('test.csv',index=False)
    #print(f't_arr: {t_arr}')
    #print(f't_arr.sum(axis=0): {t_arr.sum(axis=0)}')



if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_config_file', type=str, default='test_config2.yml', help='the path for the config file')

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args)
