from io import StringIO
import os
import argparse
import numpy as np
import pandas as pd
import data_processing as dp
import configparser
import pickle
import time
import process_data_utils as data_utils





class CacheFiles:
    def __init__(self,config):
        self.config=config
        self.path=config['cache_data']['cache_path']
        self.file_name=config['cache_data']['cache_file']

    def pickle_cache_files(self, data_dict):
        # Create the cache directory if it does not exist
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        # Pickle the data dictionary to the cache file
        with open(os.path.join(self.path, self.file_name), 'wb') as f:
            pickle.dump(data_dict, f)
    
    def unpickle_cache_files(self):
        # Unpickle the data dictionary from the cache file
        with open(os.path.join(self.path, self.file_name), 'rb') as f:
            data_dict = pickle.load(f)
        
        return data_dict

class FileLocations:
    """
    create a class that encapsulates the following structuer
    example of data_config file
[input_data]
  input_path: data
  ownership_file: SHARADAR_holdings.csv # input file


  #will probably have an additional path parameter for these
  ticker_index_file: ticker_index.cs(self):
  investor_index_file: investor_index.csv
  date_index_file: date_index.csv
[output_data] 
  output_path: data/pipeline1
  values_filename: test_output.csv
  values_file_type: relational
  inferred_ticker_dimenension: inferred_ticker_dim.csv
  inferred_date_dimension: inferred_date_dim.csv
  inferred_investorname_dimension: inverred_investorname_dim.csv
  ticker_dimension: ticker_dimension.csv
  date_dimension: date_dimension.csv
  investor_dimension: investor_dimension.csv
    """
    def __init__(self,config):
        self.config=config
        self.input_path=config['input_data']['input_path']
        self.input_ownership_file=os.path.join(self.input_path,config['input_data']['ownership_file'])
        self.input_ticker_dimension_file=os.path.join(self.input_path,config['input_data']['ticker_dimension_file'])
        self.input_investor_dimension_file=os.path.join(self.input_path,config['input_data']['investor_dimension_file'])
        self.input_date_dimension_file=os.path.join(self.input_path,config['input_data']['date_dimension_file'])
        self.output_path=config['output_data']['output_path']
        self.output_values_filename=os.path.join(self.output_path,config['output_data']['values_filename'])
        self.output_values_file_type=config['output_data']['values_file_type']
        self.output_inferred_ticker_dimension=os.path.join(self.output_path,config['output_data']['inferred_ticker_dimension'])
        self.output_inferred_date_dimension=os.path.join(self.output_path,config['output_data']['inferred_date_dimension'])
        self.output_inferred_investor_dimension=os.path.join(self.output_path,config['output_data']['inferred_investorname_dimension'])
        self.output_ticker_dimension=os.path.join(self.output_path,config['output_data']['ticker_dimension'])
        self.output_date_dimension=os.path.join(self.output_path,config['output_data']['date_dimension'])
        self.output_investor_dimension=os.path.join(self.output_path,config['output_data']['investor_dimension'])
        self.output_ticker_dimension=os.path.join(self.output_path,config['output_data']['ticker_dimension'])
        self.value_field="value"

    def get_dimension_names(self):
        dimension_names=["investorname","ticker","calendardate"]
        return dimension_names



    def get_input_dimension_files(self):
        dimension_files=[self.input_investor_dimension_file,self.input_ticker_dimension_file,self.input_date_dimension_file]
        return dimension_files

    def get_output_dimension_files(self):
        dimension_files=[self.output_investor_dimension,self.output_ticker_dimension,self.output_date_dimension]
        return dimension_files

    def get_inferred_dimension_files(self):
        dimension_files=[self.output_inferred_investor_dimension,self.output_inferred_ticker_dimension,self.output_inferred_date_dimension]
        return dimension_files



def get_dimensions(df,file_locations,infer_dimensions=True,create_inferred_dimension_files=False):
    #file_locations is an instance of FileLocations
    #df is a dataframe with the ownership data

    if infer_dimensions==True:
    # Get the number of unique tickers, investors, and dates
    # get all  the dimension names ()
        num_members=[]
        for dimension in file_locations.get_dimension_names():
            num_members.append(len(df[dimension].unique()))
        num_tickers,num_investors,num_dates=num_members
        dimension_indeces=[]
        for (dimension,dimension_file) in zip(file_locations.get_dimension_names(),file_locations.get_inferred_dimension_files()):
            dim_members=df[dimension].unique()
            dim_dict=dict(zip(dim_members,range(len(dim_members))))
            dimension_indeces.append(dim_dict)
            if create_inferred_dimension_files==True:
                dim_df=pd.DataFrame.from_dict(dim_dict,orient='index')
                if not os.path.exists(file_locations.output_path):
                    os.makedirs(file_locations.output_path)
                dim_df.to_csv(dimension_file)
    else:
        dimension_indeces=[]
        for (dimension,dimension_file) in zip(file_locations.get_dimension_names(),file_locations.get_input_dimension_files()):
            dim_df=pd.read_csv(dimension_file)
            dim_dict=dict(zip(dim_df[dimension],dim_df.index))
            dimension_indeces.append(dim_dict)
    return tuple(dimension_indeces)

def select_dimension_subset(df, dimension_name='investorname'):
    """
    Selects a subset of the specified dimension from the input DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame to select the dimension subset from.
    dimension_name (str): The name of the dimension to select the subset for. Defaults to 'investorname'.

    Returns:
    dimension_subset (Index): The resulting subset of the specified dimension.
    """
    # Group the DataFrame by the specified dimension and calendardate, and sum the values
    grouped_df = df.groupby([dimension_name, 'calendardate']).sum()

    # Pivot the grouped DataFrame using the specified dimension as the index and calendardate as the columns
    dimension_quarter_table = grouped_df.pivot_table(index=dimension_name, columns='calendardate', values='value')

    # Select the rows in the DataFrame where all quarters have a value greater than zero for the specified dimension
    all_quarters_positive = (dimension_quarter_table > 0).all(axis=1)
    dimension_subset = dimension_quarter_table.loc[all_quarters_positive].index

    # Return the resulting subset of the specified dimension
    return dimension_subset
        


def main(config):
    start_time = time.time()


    if(config.debug==True):
        print(f'config: {config}')

    data_config = configparser.ConfigParser()
    data_config.read(config.data_config_file)
    file_locations=FileLocations(data_config)
    cache_files=CacheFiles(data_config)
    #full_ownership_filename=data_config['data']['full_ownership_file']
    """
    example of data_config file
[input_data]
  ownership_file: data/SHARADAR_holdings.csv # input file
  #will probably have an additional path parameter for these
  ticker_index_file: ticker_index.csv
  investor_index_file: ginvestor_index.csv
  date_index_file: gdate_index.csv
[output_data] 
  path: data/pipeline1
  values_filename: test_output.csv
  values_file_type: relational
  ticker_dimension: ticker_dimension.csv
  date_dimension: date_dimension.csv
  investor_dimension: investor_dimension.csv
    """
    #print the config file
    if(config.debug==True):
        buffer=StringIO()
        data_config.write(buffer)
        print(f'data_config: {buffer.getvalue()}')
        print(f'file_locations: {file_locations}')
        print(f'cache_files: {cache_files}')
        #print(f'full_ownership: {full_ownership_filename}')


    #input_data_file=data_config['input_data']['ownership_file']
    input_data_file=file_locations.input_ownership_file
    if ("ticker_index_file" in data_config['input_data']):
        ticker_index_file=data_config['input_data']['ticker_index_file']
    else:
        ticker_index_file=None
    if ("investor_index_file" in data_config['input_data']):
        investor_index_file=data_config['input_data']['investor_index_file']
    else:
        investor_index_file=None

    if (config.starting_checkpoint==0):
        df=None
        print ("Step 1Loading full data set")
        print(config.data_file)
        print(f'Loading data from {input_data_file}')
        if(config.load_full_data==True):
            #df=dp.load_full_data_set(config.data_file)
            df=dp.load_full_data_set(input_data_file)
            if(config.debug==True):
                print(df.head())

        # Step 1
        # Code for step 1 here
        step1_time_mark=time.time()
        step1_time = time.time() - start_time
        print(f'Time spent on step 1: {step1_time:.2f} seconds')
        print(f'step 2: inferring the dimensions')
        # Step 2


        # possibilities of loading the date:
        # 1. load the full relational file, create the dimensions
        # 2. load the full relational file, load the dimensions, select the rows where the tickers, investornames and calendardates are along the dimensions
        # 3. read a binary file with the values, load the dimension files
        # output options:
        # 1. save the dimensions to csv files, save the values to a binary file
        # 2. save the dimensions to a csv file, save the values to a relational file (this might be rows along a  smaller subset of dimension values)

        if (config.create_dimension_files==True):
            print ("Creating dimension files")
            (investor_index,ticker_index,date_index)=get_dimensions(df,file_locations,infer_dimensions=True,create_inferred_dimension_files=True)
        else:
            (investor_index,ticker_index,date_index)=get_dimensions(df,file_locations,infer_dimensions=False,create_inferred_dimension_files=False)   

        if (config.debug==True):
            print(f'investor_index: {investor_index}')
            print(f'ticker_index: {ticker_index}')
            print(f'date_index: {date_index}')

        # Step 3
        # Create a dense values matrix
        step2_time = time.time() - step1_time_mark
        step2_time_mark=time.time()
        print(f'Time spent on step 2: {step2_time:.2f} seconds')

        print(f'step 3:  filter the dataset')
        investor_subset=select_dimension_subset(df, dimension_name='investorname')
        ticker_subset=select_dimension_subset(df, dimension_name='ticker')
        
        # along each dimension select a subset of dimension values that have values for all quarters
        filtered_df = df[df['ticker'].isin(ticker_subset) & df['investorname'].isin(investor_subset)]
        subset_investor_to_index={inv_name:inv_ind for inv_ind,inv_name in enumerate(investor_subset)}
        subset_ticker_to_index={ticker_name:ticker_ind for ticker_ind,ticker_name in enumerate(ticker_subset)}
        subset_date_to_index={date:date_ind for date_ind,date in enumerate(sorted(df['calendardate'].unique().tolist()))}
        #create dense matrices for all of the value fields
        arr_vals=dp.get_dense_values(filtered_df,subset_ticker_to_index,subset_investor_to_index,subset_date_to_index,'value')
        arr_units=dp.get_dense_values(filtered_df,subset_ticker_to_index,subset_investor_to_index,subset_date_to_index,'units')
        arr_price=dp.get_dense_values(filtered_df,subset_ticker_to_index,subset_investor_to_index,subset_date_to_index,'price')

        if (config.debug==True):
            print(f'val_arr.shape:,{arr_vals.shape}')

        # Step 4
        # Create a sparse values matrix
        step3_time = time.time() - step2_time_mark
        step3_time_mark=time.time()
        print(f'Time spent on step 3: {step3_time:.2f} seconds')

        cache_contents={
            "investor_index":subset_investor_to_index,
            "ticker_index":subset_ticker_to_index,
            "date_index":subset_date_to_index,
            "arr_vals":arr_vals,
            "arr_units":arr_units,
            "arr_price":arr_price,
        }
        cache_files.pickle_cache_files(cache_contents)
        print(f'Time spent on step 4: {step3_time:.2f} seconds')
    else:
        cache_contents=cache_files.unpickle_cache_files()

    step4_time_mark=time.time()


    if (config.debug==True):
        print(f'investor_index: {cache_contents["investor_index"]}')
        print(f'ticker_index: {cache_contents["ticker_index"]}')
        print(f'date_index: {cache_contents["date_index"]}')
        print(f'val_arr.shape:,{cache_contents["val_arr"].shape}')
    
    # Step 5
    #calculate percent ownership
    print(f'step 5: calculate percent ownership')
    arr_percent_ownership=data_utils.normalize_along_investorname(cache_contents["arr_vals"])
    if (config.debug==True):
        print(f'arr_percent_ownership.shape:,{arr_percent_ownership.shape}')
        print(f'arr_percent_ownership:,{arr_percent_ownership}')
    
    cache_contents["arr_percent_ownership"]=arr_percent_ownership
    cache_files.pickle_cache_files(cache_contents)

    #convert the data to relational format

    # Step 6
    #convert the data to relational format  
    print(f'step 6: convert the data to relational format')
    df_relational=data_utils.convert_to_relational(cache_contents["arr_percent_ownership"],cache_contents["ticker_index"].keys(),cache_contents["investor_index"].keys(),cache_contents["date_index"].keys())
    if (config.debug==True):
        print(f'df_relational:,{df_relational}')
        print(f'df_relational.shape:,{df_relational.shape}')
    df_relational.to_csv(file_locations.output_value_file,index=False)


    
if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_file', type=str, help='the path for the raw input date file')
    parser.add_argument('--debug', type=bool, default=False, help='whether to run in debug mode')
    parser.add_argument('--test_run', type=bool, default=False, help='whether to run a test run')
    parser.add_argument('--out_data_dir', type=str, default='data', help='the path for the output data directory')
    parser.add_argument('--load_full_data', type=bool, default=False, help='whether to load the full data set')
    parser.add_argument('--create_dimension_files', type=bool, default=False, help='whether to create the dimension files')
    parser.add_argument('--data_config_file', type=str, default='test_config2.yml', help='the path for the config file')
    parser.add_argument('--starting_checkpoint', type=int, default=0, help='the starting checkpoint')

    # Parse the command line arguments
    args = parser.parse_args()

    # Call the main function with the command line arguments
    main(args)
    
