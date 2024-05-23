import os
import csv
import tempfile
import zipfile
import requests
import json
import pandas as pd


from .utils import (
    TokenManager,
    FileReader,
    FileSaver,
    create_zip_file,
    create_temp_csv,
    get_dataset_tables,
    get_table,
    get_dataset_tables,
    getReconciliatorData,
    cleanServiceList,
    getExtenderData,
    getReconciliator,
    createReconciliationPayload,
    parseNameField,
    createCellMetadataNameField,
    calculateScoreBoundCell,
    valueMatchCell,
    createAnnotationMetaCell,
    updateMetadataCells,
    calculateNCellsReconciliatedColumn,
    createContextColumn,
    getColumnMetadata,
    createMetadataFieldColumn,
    calculateScoreBoundColumn,
    createAnnotationMetaColumn,
    updateMetadataColumn,
    calculateScoreBoundTable,
    calculateNCellsReconciliated,
    updateMetadataTable,
    getExtender,
    createExtensionPayload,
    getReconciliatorFromPrefix,
    getColumnIdReconciliator,
    checkEntity,
    parseNameEntities,
    addExtendedCell,
    addExtendedColumns,
    parseNameMetadata,
    addExtendedColumns, 
    update_table
)

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

def process_data(df, date_col=None, lowercase_col=None, dropna=False, column_rename_dict=None, dtype_dict=None, new_column_order=None):
    # If a date column is specified, convert it to ISO format
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
        df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')

    # If a column for lowercase conversion is specified, convert it
    if lowercase_col:
        df[lowercase_col] = df[lowercase_col].str.lower()

    # If dropna is True, drop null values
    if dropna:
        df.dropna(inplace=True)

    # Rename columns if column_rename_dict is provided
    if column_rename_dict:
        df = df.rename(columns=column_rename_dict)

    # Convert data types if dtype_dict is provided
    if dtype_dict:
        for col, dtype in dtype_dict.items():
            df[col] = df[col].astype(dtype)

    # Reorder columns if new_column_order is provided
    if new_column_order:
        df = df[new_column_order]

    # Add more transformations as needed

    return df

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
        dict: The table data in JSON format, including the table_id.
    """
    tables = get_dataset_tables(dataset_id, token_manager)
    
    for table in tables:
        if table["name"] == table_name:
            table_id = table["id"]
            table_data = get_table(dataset_id, table_id, token_manager)
            if table_data:
                table_data["id"] = table_id  # Ensure the ID is included in the returned data
                return table_data
    
    print(f"Table '{table_name}' not found in the dataset.")
    return None

def delete_table(dataset_id, table_name, token_manager):
    """
    Deletes a table by its name from a specific dataset.
    
    Args:
        dataset_id (str): The ID of the dataset.
        table_name (str): The name of the table to delete.
        token_manager (TokenManager): An instance of the TokenManager class.
    """
    table = get_table_by_name(dataset_id, table_name, token_manager)
    if not table:
        return
    
    table_id = table.get("id")
    if not table_id:
        print(f"Failed to retrieve the ID for table '{table_name}'.")
        return
    
    url = f"{API_URL}{DATASETS_ENDPOINT}{dataset_id}/table/{table_id}"
    headers = {
        'Authorization': f'Bearer {token_manager.get_token()}',
        'Accept': 'application/json'
    }
    
    response = requests.delete(url, headers=headers)
    
    if response.status_code == 200:
        print(f"Table '{table_name}' deleted successfully!")
    elif response.status_code == 401:
        print("Unauthorized: Invalid or missing token.")
    elif response.status_code == 404:
        print(f"Table '{table_name}' not found in the dataset.")
    else:
        print(f"Failed to delete table: {response.status_code}, {response.text}")

def add_table_to_dataset(dataset_id, table_data, table_name, token_manager):
    """
    Adds a table to a specific dataset.
    
    Args:
        dataset_id (str): The ID of the dataset.
        table_data (DataFrame): The table data to be added.
        table_name (str): The name of the table to be added.
        token_manager (TokenManager): An instance of the TokenManager class.
    """
    url = f"{API_URL}{DATASETS_ENDPOINT}{dataset_id}/table"
    
    token = token_manager.get_token()
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }
    
    # Create a temporary CSV file from the DataFrame
    temp_file_path = create_temp_csv(table_data)
    
    try:
        with open(temp_file_path, 'rb') as file:
            files = {
                'file': (file.name, file, 'text/csv')
            }
            
            data = {
                'name': table_name
            }
            
            response = requests.post(url, headers=headers, data=data, files=files, timeout=30)
        
        if response.status_code in [200, 201]:
            print("Table added successfully!")
            response_data = response.json()
            if 'tables' in response_data:
                tables = response_data['tables']
                for table in tables:
                    table_id = table['id']
                    table_name = table['name']
                    print(f"New table added: ID: {table_id}, Name: {table_name}")
            else:
                print("Response JSON does not contain 'tables' key.")
        else:
            print(f"Failed to add table: {response.status_code}, {response.text}")
    
    except requests.RequestException as e:
        print(f"Request error occurred: {e}")
    
    except IOError as e:
        print(f"File I/O error occurred: {e}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def update_table(dataset_id, table_name, update_payload, token_manager):
    """
    Updates a table in a specific dataset.
    
    Args:
        dataset_id (str): The ID of the dataset.
        table_name (str): The name of the table to update.
        update_payload (dict): The payload containing the updated table data.
        token_manager (TokenManager): An instance of the TokenManager class.
    
    Returns:
        None
    """
    tables = get_dataset_tables(dataset_id, token_manager)
    
    for table in tables:
        if table["name"] == table_name:
            table_id = table["id"]
            url = f"{API_URL}{DATASETS_ENDPOINT}{dataset_id}/table/{table_id}"
            headers = {
                "Authorization": f"Bearer {token_manager.get_token()}",
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.put(url, headers=headers, json=update_payload)
                
                if response.status_code == 200:
                    print("Table updated successfully!")
                    response_data = response.json()
                    print("Response data:", response_data)
                elif response.status_code == 401:
                    print("Unauthorized: Invalid or missing token.")
                elif response.status_code == 404:
                    print(f"Dataset or table with ID {dataset_id}/{table_id} not found.")
                else:
                    print(f"Failed to update table: {response.status_code}, {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"Error occurred while updating table: {e}")
            
            return
    
    print(f"Table '{table_name}' not found in the dataset.")

def list_tables_in_dataset(dataset_id, token_manager):
    """
    Lists all tables in a specific dataset.
    
    Args:
        dataset_id (str): The ID of the dataset.
        token_manager (TokenManager): An instance of the TokenManager class.
    """
    tables = get_dataset_tables(dataset_id, token_manager)
    
    if not tables:
        print(f"No tables found in dataset with ID: {dataset_id}")
        return
    
    print(f"Tables in dataset {dataset_id}:")
    for table in tables:
        table_id = table.get('id')
        table_name = table.get('name')
        if table_id and table_name:
            print(f"ID: {table_id}, Name: {table_name}")
        else:
            print("A table with missing ID or name was found.")

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

def getReconciliatorParameters(idReconciliator, print_params=False):
    """
    Retrieves the parameters needed for a specific reconciliator service.

    :param idReconciliator: the ID of the reconciliator service
    :param print_params: (optional) whether to print the retrieved parameters or not
    :return: a dictionary containing the parameter details
    """
    mandatory_params = [
        {'name': 'table', 'type': 'json', 'mandatory': True, 'description': 'The table data in JSON format'},
        {'name': 'columnName', 'type': 'string', 'mandatory': True, 'description': 'The name of the column to reconcile'},
        {'name': 'idReconciliator', 'type': 'string', 'mandatory': True, 'description': 'The ID of the reconciliator to use'}
    ]
    
    reconciliatorData = getReconciliatorData()
    for reconciliator in reconciliatorData:
        if reconciliator['id'] == idReconciliator:
            parameters = reconciliator.get('formParams', [])
            optional_params = [
                {
                    'name': param['id'],
                    'type': param['inputType'],
                    'mandatory': 'required' in param.get('rules', []),
                    'description': param.get('description', ''),
                    'label': param.get('label', ''),
                    'infoText': param.get('infoText', '')
                } for param in parameters
            ]

            param_dict = {
                'mandatory': mandatory_params,
                'optional': optional_params
            }

            if print_params:
                print(f"Parameters for reconciliator '{idReconciliator}':")
                print("Mandatory parameters:")
                for param in param_dict['mandatory']:
                    print(f"- {param['name']} ({param['type']}): Mandatory")
                    print(f"  Description: {param['description']}")
                
                print("\nOptional parameters:")
                for param in param_dict['optional']:
                    mandatory = "Mandatory" if param['mandatory'] else "Optional"
                    print(f"- {param['name']} ({param['type']}): {mandatory}")
                    print(f"  Description: {param['description']}")
                    print(f"  Label: {param['label']}")
                    print(f"  Info Text: {param['infoText']}")

            return param_dict

    return None

def getExtenderParameters(idExtender, print_params=False):
    """
    Retrieves the parameters needed for a specific extender service.

    :param idExtender: the ID of the extender service
    :param print_params: (optional) whether to print the retrieved parameters or not
    :return: a dictionary containing the parameter details
    """
    extenderData = getExtenderData()
    for extender in extenderData:
        if extender['id'] == idExtender:
            parameters = extender.get('formParams', [])
            mandatory_params = [
                {
                    'name': param['id'],
                    'type': param['inputType'],
                    'mandatory': 'required' in param.get('rules', []),
                    'description': param.get('description', ''),
                    'label': param.get('label', ''),
                    'infoText': param.get('infoText', ''),
                    'options': param.get('options', [])
                } for param in parameters if 'required' in param.get('rules', [])
            ]
            optional_params = [
                {
                    'name': param['id'],
                    'type': param['inputType'],
                    'mandatory': 'required' in param.get('rules', []),
                    'description': param.get('description', ''),
                    'label': param.get('label', ''),
                    'infoText': param.get('infoText', ''),
                    'options': param.get('options', [])
                } for param in parameters if 'required' not in param.get('rules', [])
            ]

            param_dict = {
                'mandatory': mandatory_params,
                'optional': optional_params
            }

            if print_params:
                print(f"Parameters for extender '{idExtender}':")
                print("Mandatory parameters:")
                for param in param_dict['mandatory']:
                    print(f"- {param['name']} ({param['type']}): Mandatory")
                    print(f"  Description: {param['description']}")
                    print(f"  Label: {param['label']}")
                    print(f"  Info Text: {param['infoText']}")
                    print(f"  Options: {param['options']}")
                    print("")

                print("Optional parameters:")
                for param in param_dict['optional']:
                    print(f"- {param['name']} ({param['type']}): Optional")
                    print(f"  Description: {param['description']}")
                    print(f"  Label: {param['label']}")
                    print(f"  Info Text: {param['infoText']}")
                    print(f"  Options: {param['options']}")
                    print("")

            return param_dict

    return None

def generateExtendColumnGuide(idExtender):
    """
    Generates a user guide for calling the extendColumn function with the specified extender.

    :param idExtender: the ID of the extender service
    :return: a string containing the user guide
    """
    parameters_info = getExtenderParameters(idExtender, print_params=False)
    
    if not parameters_info:
        return f"Failed to retrieve parameters for extender '{idExtender}'."
    
    guide = []
    
    guide.append("To call the extendColumn function, use the following template:")
    guide.append("")
    guide.append("```python")
    guide.append("update_payload = Reconciled_data['raw']  # Your reconciled data")
    guide.append('reconciliatedColumnName = "City"  # The name of the reconciled column')
    guide.append(f'idExtender = "{idExtender}"  # The ID of the extender to use')
    guide.append('newColumnsName = ["apparent_temperature_max", "apparent_temperature_min", "precipitation_sum"]  # New columns names')
    guide.append('dateColumnName = "Fecha_id"  # Column name for the date')
    guide.append('weatherParams = ["apparent_temperature_max", "apparent_temperature_min", "precipitation_sum"]  # Weather parameters')
    guide.append("")
    guide.append("# Mandatory parameters:")
    guide.append("New_data = extendColumn(update_payload, reconciliatedColumnName, idExtender,")

    for param in parameters_info['mandatory']:
        guide.append(f"    {param['name']}={param['name']},  # {param['description']}")
    
    guide.append(")")

    if parameters_info['optional']:
        guide.append("")
        guide.append("# Optional parameters (add as needed):")
        for param in parameters_info['optional']:
            guide.append(f"{param['name']} = ...  # {param['description']}")

    guide.append("```")
    
    return "\n".join(guide)

def generateReconcileGuide(idReconciliator):
    """
    Generates a user guide for calling the reconcile function with the specified reconciliator.

    :param idReconciliator: the ID of the reconciliator service
    :return: a string containing the user guide
    """
    parameters_info = getReconciliatorParameters(idReconciliator, print_params=False)
    
    if not parameters_info:
        return f"Failed to retrieve parameters for reconciliator '{idReconciliator}'."
    
    guide = []
    
    guide.append("To call the reconcile function, use the following template:")
    guide.append("")
    guide.append("```python")
    guide.append("table = table_json  # Your table data in JSON format")
    guide.append('columnName = "City"  # The name of the column to reconcile')
    guide.append(f'idReconciliator = "{idReconciliator}"  # The ID of the reconciliator to use')
    guide.append("")
    
    # Add mandatory parameters
    for param in parameters_info['mandatory']:
        guide.append(f'# {param["description"]}')
        guide.append(f'{param["name"]} = ...  # Define your {param["type"]} here')
        guide.append("")
    
    # Add optional parameters
    if parameters_info['optional']:
        guide.append("# Optional parameters:")
        for param in parameters_info['optional']:
            guide.append(f'# {param["description"]}')
            guide.append(f'{param["name"]} = ...  # Define your {param["type"]} here (optional)')
            guide.append("")
    
    # Add the function call
    guide.append("Reconciled_data = reconcile(table, columnName, idReconciliator")
    if parameters_info['optional']:
        for param in parameters_info['optional']:
            guide.append(f'    {param["name"]}={param["name"]},')
    guide.append(")")
    guide.append("```")
    
    return "\n".join(guide)

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

def extendColumn(table, reconciliatedColumnName, idExtender, properties, newColumnsName, dateColumnName, weatherParams, decimalFormat=None):
    """
    Extends the specified properties present in the Knowledge Graph as new columns.

    :param table: the table containing the data
    :param reconciliatedColumnName: the column containing the ID in the KG
    :param idExtender: the extender to use for extension
    :param properties: the properties to extend in the table
    :param newColumnsName: the names of the new columns to add
    :param dateColumnName: the name of the date column to extract date information for each row
    :param weatherParams: a list of weather parameters to include in the request
    :return: the extended table
    """
    reconciliatorResponse = getReconciliatorData()
    extenderData = getExtender(idExtender, getExtenderData())
    
    if extenderData is None:
        raise ValueError(f"Extender with ID '{idExtender}' not found.")
    
    url = API_URL + "extenders/" + extenderData['relativeUrl']
    
    # Prepare the dates information dynamically
    dates = {}
    for row_key, row_data in table['rows'].items():
        date_value = row_data['cells'].get(dateColumnName, {}).get('label')
        if date_value:
            dates[row_key] = [date_value]
        else:
            print(f"Missing or invalid date for row {row_key}, skipping this row.")
            continue  # Optionally skip this row or handle accordingly
    decimalFormat = ["comma"]  # Use comma as the decimal separator
    payload = createExtensionPayload(table, reconciliatedColumnName, properties, idExtender, dates, weatherParams, decimalFormat)
    #payload = createExtensionPayload(table, reconciliatedColumnName, properties, idExtender, dates, weatherParams)
    headers = {"Accept": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        table = addExtendedColumns(table, data, newColumnsName, reconciliatorResponse)
        return table
    except requests.RequestException as e:
        print(f"An error occurred while making the request: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_reconciliation_payload_for_backend(table_json):
    """
    Creates the payload required to perform the table update operation

    :param table_json: JSON representation of the table
    :return: request payload
    """
    payload = {
        "tableInstance": {
            "id": table_json["raw"]["table"]["id"],
            "idDataset": table_json["raw"]["table"]["idDataset"],
            "name": table_json["raw"]["table"]["name"],
            "nCols": table_json["raw"]["table"]["nCols"],
            "nRows": table_json["raw"]["table"]["nRows"],
            "nCells": table_json["raw"]["table"]["nCells"],
            "nCellsReconciliated": table_json["raw"]["table"]["nCellsReconciliated"],
            "lastModifiedDate": table_json["raw"]["table"]["lastModifiedDate"]
        },
        "columns": {
            "byId": table_json["raw"]["columns"],
            "allIds": list(table_json["raw"]["columns"].keys())
        },
        "rows": {
            "byId": table_json["raw"]["rows"],
            "allIds": list(table_json["raw"]["rows"].keys())
        }
    }
    return payload

def reconciled_table_update(token_manager, dataset_id, table_id, Reconciled_data):
    """
    Updates a table in the dataset with reconciled data.

    :param dataset_id: The ID of the dataset containing the table.
    :param table_id: The ID of the table to update.
    :return: A message indicating the status of the update operation.
    """
    # API configuration
    API_URL = "http://localhost:3003/api/"
    DATASETS_ENDPOINT = "dataset/"
    
    headers = {
        'Authorization': f'Bearer {token_manager.get_token()}',
        'Content-Type': 'application/json'
    }

    url = f"{API_URL}{DATASETS_ENDPOINT}{dataset_id}/table/{table_id}"

    # Create the update payload
    update_payload = create_reconciliation_payload_for_backend(Reconciled_data)

    try:
        # Send the PUT request to update the table
        response = requests.put(url, headers=headers, json=update_payload)

        if response.status_code == 200:
            message = "Table updated successfully!"
            response_data = response.json()
            message += f"\nResponse data: {response_data}"
        elif response.status_code == 401:
            message = "Unauthorized: Invalid or missing token."
        elif response.status_code == 404:
            message = f"Dataset or table with ID {dataset_id}/{table_id} not found."
        else:
            message = f"Failed to update table: {response.status_code}, {response.text}"

        return message

    except requests.exceptions.RequestException as e:
        return f"Error occurred while updating table: {e}"

def evaluate_reconciliation(data, reconciliatedColumnName):
    """
    Evaluates the reconciliation and extracts metrics from the metadata.

    :param data: The table containing the data
    :param reconciliatedColumnName: The column containing the reconciled data
    :return: A DataFrame with extracted metrics
    """
    metrics = []

    # Iterate through each row
    for row_id, row_data in data['rows'].items():
        row_dict = {'row_id': row_id}

        # Iterate through each cell in the row
        for cell_id, cell_data in row_data['cells'].items():
            metadata_list = cell_data.get('metadata', [])
            
            if metadata_list:
                for metadata in metadata_list:
                    if metadata.get('match'):
                        row_dict[f'{cell_id}_score'] = metadata.get('score', None)
                        row_dict[f'{cell_id}_type'] = [t['name'] for t in metadata.get('type', [])]
                        row_dict[f'{cell_id}_features'] = metadata.get('feature', [])
        
        metrics.append(row_dict)

    # Create a DataFrame from the metrics
    metrics_df = pd.DataFrame(metrics)

    return metrics_df

def extend_Reconciliation_Results(json, reconciliatedColumnName, properties, newColumnsName):
    """
    Extends the reconciled column by creating new columns for each property in the metadata.

    :param data: the table containing the data
    :param reconciliatedColumnName: the column containing the reconciled data
    :param properties: the properties to extend in the table
    :param newColumnsName: the names of the new columns to add
    :return: the extended data as a DataFrame
    """
    if len(properties) != len(newColumnsName):
        raise ValueError("The number of properties and new column names should be the same.")
    
    # Initialize a list to store the extracted data
    extracted_data = []

    # Iterate through each row
    for row_id, row_data in data['raw']['rows'].items():
        row_dict = {}
        row_dict['row_id'] = row_id
        
        # Iterate through each cell in the row
        for cell_id, cell_data in row_data['cells'].items():
            cell_label = cell_data.get('label')
            metadata_list = cell_data.get('metadata', [])
            
            # If metadata is not empty, process it
            if metadata_list:
                for metadata in metadata_list:
                    if metadata.get('match'):
                        # Extract metadata details
                        for prop, newCol in zip(properties, newColumnsName):
                            row_dict[newCol] = metadata.get(prop, '')
            else:
                # If no metadata, just add the cell label
                row_dict[cell_id] = cell_label
        
        extracted_data.append(row_dict)

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(extracted_data)
    
    return df

def extract_nested_values_reconciliation(df, column, new_columns):
    """
    Extracts nested values from a column in the DataFrame and adds them as new columns.

    :param df: The original DataFrame
    :param column: The column containing the nested values
    :param new_columns: The names of the new columns to add
    :return: The updated DataFrame with the new columns
    """
    df[new_columns[0]] = df[column].apply(lambda x: x.get('value') if isinstance(x, dict) else None)
    df[new_columns[1]] = df[column].apply(lambda x: x.get('uri') if isinstance(x, dict) else None)
    return df

def create_extension_payload_for_backend(table_json):
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
            "status": column_data.get("status", ""),
            "context": column_data.get("context", {}),
            "metadata": column_data.get("metadata", []),
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
                "label": cell_data.get("label", ""),
                "metadata": cell_data.get("metadata", []),
                "annotationMeta": cell_data.get("annotationMeta", {})
            }
        payload["rows"]["allIds"].append(row_id)

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
