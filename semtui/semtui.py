import os
import tempfile
import zipfile
import requests
import json
import pandas as pd

# Correct the import statement in semtui.py
from .utils import (TokenManager, FileReader, FileSaver, create_zip_file, get_dataset_tables, get_table, process_data, 
                    cleanServiceList, getReconciliatorData, getExtenderData, getReconciliator,
                    createReconciliationPayload, updateMetadataTable, createCellMetadataNameField,
                    updateMetadataCells, updateMetadataColumn, getExtender, createExtensionPayload,
                    parseNameField, createCellMetadataNameField, calculateScoreBoundCell, 
                    valueMatchCell, createAnnotationMetaCell, updateMetadataCells, 
                    calculateNCellsReconciliatedColumn, createContextColumn, getColumnMetadata, 
                    createMetadataFieldColumn, calculateScoreBoundColumn, createAnnotationMetaColumn, 
                    updateMetadataColumn, calculateScoreBoundTable, calculateNCellsReconciliated, 
                    updateMetadataTable, addExtendedColumns, update_table)

API_URL = "http://localhost:3003/api/"
SIGNIN_ENDPOINT = "auth/signin"
DATASETS_ENDPOINT = "dataset/"

def obtain_auth_token(username, password):
    signin_data = {"username": username, "password": password}
    signin_headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    token_manager = TokenManager(signin_data, signin_headers)
    return token_manager.get_token()

def process_csv_data(file_path):
    reader = FileReader(file_path)
    try:
        df = reader.read_csv()  # Reading a CSV file using the FileReader
        # Process the DataFrame as needed for SEMT-specific functionalities
        return df
    except Exception as e:
        print(f"An error occurred while processing the CSV file: {e}")

def add_dataset(token_manager, zip_file_path, dataset_name):
    """
    Adds a new dataset to the server by sending a POST request with the given zip file and dataset name.

    Args:
        token_manager (TokenManager): The TokenManager instance to obtain the authentication token.
        zip_file_path (str): The path to the zip file to be uploaded.
        dataset_name (str): The name of the new dataset.

    Returns:
        tuple: A tuple containing a boolean success flag and the dataset ID if the dataset was added successfully,
        or an error message if the dataset could not be added.
    """
    url = f"{API_URL}{DATASETS_ENDPOINT}"
    token = token_manager.get_token()
    # Set the headers with the token
    headers = {
        'Authorization': f'Bearer {token}'
    }
    # Open the zip file in binary mode
    with open(zip_file_path, 'rb') as file:
        files = {
            'file': (file.name, file, 'application/zip')
        }
        data = {
            'name': dataset_name
        }
        # Send the POST request to add the dataset
        response = requests.post(url, headers=headers, data=data, files=files, timeout=30)
        # Check the response
        if response.status_code == 200:
            print("Dataset added successfully!")
            # Extract the dataset ID from the response
            response_data = response.json()
            dataset_id = response_data['datasets'][0]['id']
            return True, dataset_id
        elif response.status_code == 400:
            # Extract the error message and dataset ID from the response
            response_data = response.json()
            error_message = response_data.get('error', 'Unknown error')
            dataset_id = response_data.get('datasetId')
            return False, error_message
        else:
            print(f"Failed to add dataset: {response.status_code}, {response.text}")
            return False, f"Failed to add dataset: {response.status_code}, {response.text}"

def get_database_list(api_url, datasets_endpoint, token_manager):
    url = f"{api_url}{datasets_endpoint}"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {token_manager.get_token()}"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Will handle HTTP errors
        database_list = response.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None  # Return None or raise an exception instead of an empty DataFrame

    # Initialize data dictionary with keys and empty lists
    data = {key: [] for key in ['id', 'userId', 'name', 'nTables', 'lastModifiedDate']}

    for dataset in database_list.get('collection', []):  # Safe default if 'collection' key is missing
        for key in data:
            data[key].append(dataset.get(key, None))  # None if key doesn't exist in dataset

    return pd.DataFrame(data)

def delete_dataset(api_url, dataset_endpoint, token_manager, dataset_id):
    """
    Deletes a specific dataset from the server using the specified API endpoint.
    
    Args:
        api_url (str): The base URL of the API server.
        dataset_endpoint (str): The endpoint path for dataset operations.
        token_manager (TokenManager): The token manager object that handles authentication.
        dataset_id (str): The unique identifier of the dataset to be deleted.
    
    Returns:
        str: A message indicating the result of the operation.
    """
    # Get the authentication token from the token manager
    token = token_manager.get_token()

    # Set the headers with the token
    headers = {
        'Authorization': f'Bearer {token}'
    }

    # Construct the full URL for the DELETE request
    url = f"{api_url}{dataset_endpoint}/{dataset_id}"

    # Send the DELETE request to remove the dataset
    response = requests.delete(url, headers=headers)

    # Check the response and return appropriate messages
    if response.status_code == 200:
        return f"Dataset with ID {dataset_id} deleted successfully!"
    elif response.status_code == 401:
        return "Unauthorized: Invalid or missing token."
    elif response.status_code == 404:
        return f"Dataset with ID {dataset_id} not found."
    else:
        return f"Failed to delete dataset: {response.status_code}, {response.text}"

def get_table_by_name(dataset_id, table_name, token_manager):
    """
    Retrieves a table by its name from a specific dataset.
    
    Args:
        dataset_id (str): The ID of the dataset.
        table_name (str): The name of the table to retrieve.
        token_manager (TokenManager): An instance of the TokenManager class.
    
    Returns:
        dict: The table data in JSON format.
    """
    tables = get_dataset_tables(dataset_id, token_manager)
    
    for table in tables:
        if table["name"] == table_name:
            table_id = table["id"]
            return get_table(dataset_id, table_id, token_manager)
    
    print(f"Table '{table_name}' not found in the dataset.")
    return None

def getReconciliatorsList():
    """
    Provides a list of available reconciliators with their main information

    :return: a dataframe containing reconciliators and their information
    """
    response = getReconciliatorData()
    return cleanServiceList(response)

def getExtendersList():
    """
    Provides a list of available extenders with their main information

    :return: a dataframe containing extenders and their information
    """
    response = getExtenderData()
    return cleanServiceList(response)

def reconcile(table, columnName, idReconciliator):
    """
    Reconciles a column with the chosen reconciliator

    :table: the table with the column to reconcile 
    :columnName: the name of the column to reconcile 
    :idReconciliator: ID of the reconciliator to use 
    :return: table with reconciled column
    """
    reconciliatorResponse = getReconciliatorData()
    reconciliator = getReconciliator(idReconciliator, reconciliatorResponse)
    if reconciliator is None:
        print(f"Reconciliator with ID {idReconciliator} not found.")
        return None
    # creating the request
    url = API_URL + 'reconciliators' + str(reconciliator['relativeUrl'])
    payload = createReconciliationPayload(table, columnName, idReconciliator)
    response = requests.post(url, json=payload)
    response = json.loads(response.text)
    # inserting data into the table
    metadata = createCellMetadataNameField(response, idReconciliator, reconciliatorResponse)
    table = updateMetadataCells(table, metadata)
    table = updateMetadataColumn(table, columnName, idReconciliator, metadata, reconciliatorResponse)
    table = updateMetadataTable(table)
    return {'raw': table}

def extract_georeference_data(reconciled_data):
    columns_data = reconciled_data['raw']['columns']
    rows_data = reconciled_data['raw']['rows']

    # Build the list of columns to keep
    columns_to_keep = ['City', 'City URI', 'Latitude', 'Longitude']

    # Prepare the list of rows with reconciled data
    rows_list = []
    for row_id, row_content in rows_data.items():
        row_values = []
        city_cell = row_content['cells'].get('City', {})
        city_label = city_cell.get('label', '')
        city_uri = ''
        latitude = ''
        longitude = ''

        for meta in city_cell.get('metadata', []):
            if 'name' in meta and 'uri' in meta['name']:
                city_uri = meta['name']['uri']
                if 'georss' in meta['id']:
                    coordinates = meta['id'].split(':')[1].split(',')
                    latitude = coordinates[0]
                    longitude = coordinates[1]
                break

        row_values.extend([city_label, city_uri, latitude, longitude])
        rows_list.append(row_values)

    # Create a DataFrame
    df = pd.DataFrame(rows_list, columns=columns_to_keep)

    return df

def extract_city_reconciliation_metrics(data):
    city_data = data['raw']['columns']['City']['metadata']
    
    if not city_data:
        return "No reconciliation data available for 'City'."
    
    num_reconciled = len(city_data)
    num_matches = sum(1 for entry in city_data if entry['match'] == True)
    num_non_matches = num_reconciled - num_matches
    scores = [entry['score'] for entry in city_data if 'score' in entry]
    max_score = max(scores)
    min_score = min(scores)
    average_score = sum(scores) / num_reconciled
    std_dev_score = pd.Series(scores).std()  # Standard deviation
    
    return {
        'Total Reconciled': num_reconciled,
        'Matches': num_matches,
        'Non-Matches': num_non_matches,
        'Max Score': max_score,
        'Min Score': min_score,
        'Average Score': average_score,
        'Standard Deviation of Scores': std_dev_score
    }

def extendColumn(table, reconciliatedColumnName, idExtender, properties, newColumnsName, dateColumnName):
    """
    Allows extending specified properties present in the Knowledge Graph as a new column.

    :param table: the table containing the data
    :param reconciliatedColumnName: the column containing the ID in the KG
    :param idExtender: the extender to use for extension
    :param properties: the properties to extend in the table
    :param newColumnsName: the name of the new columns to add
    :param dateColumnName: the name of the date column to extract date information for each row
    :return: the extended table
    """
    # Prepare the dates information dynamically
    dates = {}
    for row_key, row_data in table['rows'].items():
        # Safely extract date_value from the label key of the date column
        date_value = row_data['cells'].get(dateColumnName, {}).get('label')
        
        if date_value:
            dates[row_key] = [date_value]
        else:
            print(f"Missing or invalid date for row {row_key}, skipping this row.")
            continue  # Optionally skip this row or handle accordingly

    reconciliatorResponse = getReconciliatorData()
    extenderData = getExtender(idExtender, getExtenderData())
    url = API_URL + "extenders/" + extenderData['relativeUrl']
    payload = createExtensionPayload(table, reconciliatedColumnName, properties, idExtender, dates)
    headers = {"Accept": "application/json"}
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"HTTP Status Code: {response.status_code}")
        print(f"HTTP Response Text: '{response.text}'")
        response.raise_for_status()
        data = response.json()
        table = addExtendedColumns(table, data, newColumnsName, reconciliatorResponse)
        return table
    except requests.RequestException as e:
        print(f"Network-related error occurred: {e}")
    except json.JSONDecodeError as e:
        print("Error parsing JSON response:", e)
        print("Raw response for debugging:", response.text)
    except KeyError as e:
        print(f"Missing expected key in data:", e)

def create_extension_payload_for_backend(table_json):
    """
    Function that creates the payload required to perform the table update operation.

    This function takes a JSON representation of a table and creates a payload required to perform the table update
    operation. The payload includes the table instance data, column data, and row data. The function processes the
    columns and rows in the table and adds them to the payload. The function returns the payload.

    Args:
        table_json (dict): JSON representation of the table.

    Returns:
        dict: Request payload.
    """
    payload = {
        "tableInstance": {
            "id": table_json["table"]["id"],
            "idDataset": table_json["table"]["idDataset"],
            "name": table_json["table"]["name"],
            "nCols": table_json["table"]["nCols"],
            "nRows": table_json["table"]["nRows"],
            "nCells": table_json["table"]["nCells"],
            "nCellsReconciliated": table_json["table"]["nCellsReconciliated"],
            "lastModifiedDate": table_json["table"]["lastModifiedDate"],
            "minMetaScore": table_json["table"]["minMetaScore"],
            "maxMetaScore": table_json["table"]["maxMetaScore"]
        },
        "columns": {
            "byId": {},
            "allIds": []
        },
        "rows": {
            "byId": {},
            "allIds": []
        }
    }

    # Process columns
    for column_id, column_data in table_json["columns"].items():
        payload["columns"]["byId"][column_id] = {
            "id": column_data["id"],
            "label": column_data["label"],
            "status": column_data["status"],
            "context": column_data["context"],
            "metadata": column_data["metadata"],
            "annotationMeta": column_data.get("annotationMeta", {})
        }
        payload["columns"]["allIds"].append(column_id)

    # Process rows
    for row_id, row_data in table_json["rows"].items():
        payload["rows"]["byId"][row_id] = {
            "id": row_data["id"],
            "cells": {}
        }
        for cell_id, cell_data in row_data["cells"].items():
            payload["rows"]["byId"][row_id]["cells"][cell_id] = {
                "id": cell_data["id"],
                "label": cell_data["label"],
                "metadata": cell_data["metadata"],
                "annotationMeta": cell_data.get("annotationMeta", {})
            }
        payload["rows"]["allIds"].append(row_id)

    return payload

def create_reconciliation_payload_for_backend(table_json):
    """
    Creates the payload required to perform the table update operation

    :param table_json: JSON representation of the table
    :return: request payload
    """
    payload = {
        "tableInstance": {
            "id": table_json["table"]["id"],
            "idDataset": table_json["table"]["idDataset"],
            "name": table_json["table"]["name"],
            "nCols": table_json["table"]["nCols"],
            "nRows": table_json["table"]["nRows"],
            "nCells": table_json["table"]["nCells"],
            "nCellsReconciliated": table_json["table"]["nCellsReconciliated"],
            "lastModifiedDate": table_json["table"]["lastModifiedDate"]
        },
        "columns": {
            "byId": table_json["columns"],
            "allIds": list(table_json["columns"].keys())
        },
        "rows": {
            "byId": table_json["rows"],
            "allIds": list(table_json["rows"].keys())
        }
    }
    return payload

def load_json_to_dataframe(data):
    """
    Converts a JSON object containing dataset information into a pandas DataFrame.
    
    This function assumes the JSON object has 'columns' and 'rows' keys. Each row is expected to
    contain 'cells', which are transformed into DataFrame columns.

    Args:
        data (dict): A JSON-like dictionary containing the data. This dictionary should
                     have at least two keys: 'columns' and 'rows', where 'rows' should
                     be a dictionary of dictionaries containing cell data.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to entries in the 'rows' of the input JSON.
                      Each cell in 'rows' becomes a column in the DataFrame.

    Raises:
        KeyError: If the expected keys ('columns' or 'rows') are missing in the input data.
        Exception: For other issues that might occur during DataFrame creation.
    """
    try:
        # Extract columns and rows from the data
        columns = data['columns']  # This is extracted but not used, assuming future use cases
        rows = data['rows']

        # Initialize a list to store each row's data as a dictionary
        data_list = []
        for row_id, row_info in rows.items():
            row_data = {}
            # Extract cell data into dictionary form, using the label as the value
            for cell_key, cell_value in row_info['cells'].items():
                row_data[cell_key] = cell_value['label']
            data_list.append(row_data)

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data_list)
        return df

    except KeyError as e:
        print(f"Key error: Missing {str(e)} in the data.")
        raise
    except Exception as e:
        print(f"An error occurred while converting JSON to DataFrame: {str(e)}")
        raise





