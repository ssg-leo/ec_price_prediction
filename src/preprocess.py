import pandas as pd
import re
from datetime import datetime
import os
import json
import hashlib

district_names = district_names = {
    1: 'Raffles Place, Cecil, Marina, People\'s Park',
    2: 'Anson, Tanjong Pagar',
    3: 'Queenstown, Tiong Bahru',
    4: 'Telok Blangah, Harbourfront',
    5: 'Pasir Panjang, Clementi',
    6: 'High Street, Beach Road (part)',
    7: 'Middle Road, Golden Mile',
    8: 'Little India, Farrer Park',
    9: 'Orchard, Cairnhill, River Valley',
    10: 'Holland, Bukit Timah, Tanglin',
    11: 'Novena, Thomson, Watten Estate',
    12: 'Toa Payoh, Serangoon, Balestier',
    13: 'Macpherson, Braddell',
    14: 'Geylang, Paya Lebar, Sims',
    15: 'Katong, Joo Chiat, Amber Road',
    16: 'Bedok, Upper East Coast, Eastwood, Kew Drive',
    17: 'Changi, Loyang, Pasir Ris',
    18: 'Pasir Ris, Tampines',
    19: 'Hougang, Punggol, Sengkang',
    20: 'Ang Mo Kio, Bishan, Braddell Road',
    21: 'Upper Bukit Timah, Clementi Park, Ulu Pandan',
    22: 'Boon Lay, Jurong, Tuas',
    23: 'Bukit Batok, Choa Chu Kang, Bukit Panjang',
    24: 'Kranji, Lim Chu Kang, Tengah',
    25: 'Admiralty, Woodlands',
    26: 'Upper Thomson, Springleaf',
    27: 'Sembawang, Yishun',
    28: 'Seletar'
}

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

def extract_built_year(tenure_str):
    """
    Extracts the built year from a tenure string.

    Args:
        tenure_str (str): The tenure string containing the built year.

    Returns:
        int: The built year as an integer, or None if not found.
    """
    match = re.search(r'\d{4}', tenure_str)
    return int(match.group()) if match else None


def preprocess_transaction(data):
    """
    Preprocesses transaction data for Executive Condominiums.

    Args:
        data (pandas.DataFrame): A pandas DataFrame containing transaction data for all property types.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing preprocessed transaction data for Executive Condominiums.
    """
    executive_condos = data[data['property_type'] == 'Executive Condominium'].copy()
    if not executive_condos.empty:
        executive_condos['contract_date'] = pd.to_datetime(executive_condos['contract_date'], format='%m%y')
        executive_condos['built_year'] = executive_condos['tenure'].apply(extract_built_year)

        executive_condos['built_year'] = pd.to_datetime(executive_condos['built_year'], format='%Y', errors='coerce')
        executive_condos['contract_date'] = pd.to_datetime(executive_condos['contract_date'], format='%m%y',
                                                        errors='coerce')
        executive_condos['age_at_sale'] = (executive_condos['contract_date'] - executive_condos['built_year']).dt.days / 365
        executive_condos['remaining_lease'] = 99 - executive_condos['age_at_sale']
        executive_condos['type_of_sale'] = executive_condos['type_of_sale'].replace({
            1: 'New Sale',
            2: 'Sub Sale',
            3: 'Resale'
        })
        executive_condos['district_name'] = data['district'].map(district_names)
        executive_condos['psm'] = executive_condos['price'] / executive_condos['area']

        return executive_condos
    
    else:
        None


def merge_df(transaction_data, hdb_resale_price_index):
    """
    Merges transaction data and HDB resale price index data.

    Args:
        transaction_data (pandas.DataFrame): Preprocessed transaction data for Executive Condominiums.
        hdb_resale_price_index (pandas.DataFrame): HDB resale price index data.

    Returns:
        pandas.DataFrame: A pandas DataFrame containing the merged transaction and HDB resale price index data.
    """
    # Convert the 'quarter' column in the HDB resale price index data to a period object
    hdb_resale_price_index['quarter'] = pd.PeriodIndex(hdb_resale_price_index['quarter'], freq='Q')

    # Extract the year and quarter from the 'quarter' column
    hdb_resale_price_index['year'] = hdb_resale_price_index['quarter'].dt.year
    hdb_resale_price_index['quarter_number'] = hdb_resale_price_index['quarter'].dt.quarter

    # Extract the transaction year and quarter from the 'contract_date' column in the preprocessed transaction data
    transaction_data['transaction_year'] = transaction_data['contract_date'].dt.year
    transaction_data['transaction_quarter'] = transaction_data['contract_date'].dt.quarter

    # Merge the two datasets together using the 'transaction_year' and 'transaction_quarter' columns in the
    # transaction data and the 'year' and 'quarter_number' columns in the HDB resale price index data as the join keys
    merged_data = pd.merge(transaction_data, hdb_resale_price_index,
                           left_on=['transaction_year', 'transaction_quarter'],
                           right_on=['year', 'quarter_number'], how='left')

    return merged_data


def select_columns(df, columns_to_keep):
    """
    Selects specified columns from a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns_to_keep (list): A list of column names to keep.

    Returns:
        pandas.DataFrame: A DataFrame containing only the specified columns.
    """
    output = df[columns_to_keep]
    return output

def parse_raw():
    """
    Parse raw data from data/raw_data/* into a transactions.csv file
    """
    result_list = []
    for file in os.listdir(os.path.join(root_dir, "data/raw_data")):
        file_path =os.path.join(root_dir, "data/raw_data", file)
        row_batch = []
        if file.endswith(".txt"):
            with open(file_path, "r") as f:
                file_b = f.read()
                file_json = json.loads(file_b)
                result = file_json.get('Result')
                for row in result:
                    project_id = hashlib.md5(
                        (row['street'] + row['project']).encode('utf-8')
                        ).hexdigest()
                    
                    for transaction in row['transaction']:
                        t_dict = dict()
                        t_dict['transaction_id'] = hashlib.md5(
                            (project_id + str(transaction['contractDate']) + str(transaction['area']) + str(transaction['price'])).encode('utf-8')
                            ).hexdigest()
                        t_dict['project_id'] = project_id
                        t_dict['project'] = row['project']
                        try:
                            t_dict['x'] = row['x']
                        except:
                            t_dict['x'] = ''
                        try:
                            t_dict['y'] = row['y']
                        except:
                            t_dict['y'] = ''
                        t_dict['street'] = row['street']
                        t_dict['area'] = transaction['area']
                        t_dict['floor_range'] = transaction['floorRange']
                        t_dict['no_of_units'] = transaction['noOfUnits']
                        t_dict['contract_date'] = str(transaction['contractDate'])
                        t_dict['type_of_sale'] = transaction['typeOfSale']
                        t_dict['price'] = transaction['price']
                        t_dict['property_type'] = transaction['propertyType']
                        t_dict['district'] = transaction['district']
                        t_dict['type_of_area'] = transaction['typeOfArea']
                        t_dict['tenure'] = transaction['tenure']

                        row_batch.append(t_dict)
                result_list += row_batch
                print(f"Processed {file_path}")

    df = pd.DataFrame(result_list)
    try:
        existing_df = pd.read_csv(os.path.join(root_dir, "data/transactions.csv"))
    except:
        existing_df = pd.DataFrame(columns=['transaction_id','project_id','project','x','y','street','area','floor_range','no_of_units','contract_date','type_of_sale','price','property_type','district','type_of_area','tenure'])

    diff = change_data_capture(df, existing_df)
    
    df = pd.concat([existing_df, diff])

    df.to_csv(os.path.join(root_dir, "data/transactions.csv"), index=False)
    print("Data processed to transactions.csv")

def change_data_capture(df, existing_df):

    merged = pd.merge(df, existing_df, on=['transaction_id'], how='left', indicator=True, suffixes=(None, '_y'))
    # filter the merged dataframe to include only the rows present in the left dataframe but not the right dataframe
    diff = merged[merged['_merge'] == 'left_only']
    # filter the cols that ends with _y
    keep_columns = [col for col in merged.columns if not (col.endswith('_y') or col.endswith('_merge'))]
    diff = diff[keep_columns]
    if not diff.empty:
        print(f'Detected {len(diff)} new transactions...')
        print('Saving new transactions to transactions_delta.csv and appending to transactions.csv')
    else:
        print('No change in extracted transactions vs our database')
    diff.to_csv(os.path.join(root_dir, "data/transactions_delta.csv"), index=False)

    return diff

def final_process(df, output_name):
    """
    Merge data with hdb resale price index and select relevant columns
    """
    hdb_resale_price_index = pd.read_csv("data/hdb-resale-price-index.csv")
    processed_data = preprocess_transaction(df)
    if processed_data is not None:
        merged_df = merge_df(processed_data, hdb_resale_price_index)
        output_df = select_columns(merged_df, columns_to_keep=["x", "y", "area", "floor_range", "type_of_sale", "district",
                                                            "district_name", "remaining_lease", "index", "psm"])
        output_df = output_df.rename(columns={'index': 'price_index'})
    else:
        output_df = pd.DataFrame(columns=["x", "y", "area", "floor_range", "type_of_sale", "district",
                                                            "district_name", "remaining_lease", "index", "psm"])
    output_df.to_csv(f"data/{output_name}", index=False)
    

if __name__ == "__main__":
    parse_raw()
    data = pd.read_csv('data/transactions.csv')
    final_process(data, 'clean_data.csv')

    #Process change data as well
    change_data = pd.read_csv('data/transactions_delta.csv')
    final_process(change_data, 'clean_data_delta.csv')
