import pandas as pd
import re
from datetime import datetime

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


if __name__ == "__main__":
    data = pd.read_csv('data/transactions.csv')
    hdb_resale_price_index = pd.read_csv("data/hdb-resale-price-index.csv")
    processed_data = preprocess_transaction(data)
    merged_df = merge_df(processed_data, hdb_resale_price_index)
    output_df = select_columns(merged_df, columns_to_keep=["x", "y", "area", "floor_range", "type_of_sale", "district",
                                                           "district_name", "remaining_lease", "index", "psm"])
    output_df = output_df.rename(columns={'index': 'price_index'})
    output_df.to_csv("data/clean_data.csv", index=False)
