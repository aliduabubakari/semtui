import requests
from time import time, sleep
import pandas as pd
import chardet
import io
import logging
import os
import tempfile
import zipfile
import json


# Set up logging
logging.basicConfig(level=logging.INFO)

API_URL = "http://localhost:3003/api/"
SIGNIN_ENDPOINT = "auth/signin"
DATASETS_ENDPOINT = "dataset/"

class TokenManager:
    def __init__(self, signin_data, signin_headers):
        self.signin_url = f"{API_URL}{SIGNIN_ENDPOINT}"
        self.signin_data = signin_data
        self.signin_headers = signin_headers
        self.token = None
        self.token_expiry = 0

    def get_token(self):
        if self.token is None or time() >= self.token_expiry:
            self.refresh_token()
        return self.token

    def refresh_token(self):
        try:
            response = requests.post(self.signin_url, headers=self.signin_headers, json=self.signin_data)
            response.raise_for_status()
            user_info = response.json()
            self.token = user_info.get("token")
            self.token_expiry = time() + 3600  # Assuming the token expires in 1 hour
        except requests.RequestException as e:
            print(f"Sign-in request failed: {e}")
            self.token = None
            self.token_expiry = 0

class FileReader:
    """ A class for reading files with detected encoding and delimiter. """
    def __init__(self, file_path):
        """ Initializes the FileReader object with the given file path.

        Args:
            file_path (str): The path to the file to be read.
        """
        self.file_path = file_path
    def read_csv(self):
        """
        Reads a CSV file with detected encoding and a tab delimiter.

        Returns:
            pd.DataFrame: The DataFrame containing the contents of the CSV file.

        Raises:
            FileNotFoundError: If the file is not found.
            PermissionError: If the file cannot be accessed due to permission issues.
            Exception: If any other error occurs during file reading.
        """
        try:
            # Detect the encoding of the file
            with open(self.file_path, 'rb') as file:
                encoding = chardet.detect(file.read())['encoding']

            # Read the CSV file with pandas using the detected encoding and a tab delimiter
            df = pd.read_csv(self.file_path, sep='\t', encoding=encoding)
            print(f"File '{self.file_path}' read successfully with encoding '{encoding}'")
            return df

        except FileNotFoundError:
            print(f"File '{self.file_path}' not found.")
            raise
        except PermissionError:
            print(f"Permission denied to read file '{self.file_path}'.")
            raise
        except Exception as e:
            print(f"Error reading file '{self.file_path}': {str(e)}")
            raise

class FileSaver:
    """ A class for saving data to files with specified encoding and format. """
    def __init__(self, file_path):
        """
        Initializes the FileSaver object with the given file path.

        Args:
            file_path (str): The path where the file will be saved.
        """
        self.file_path = file_path

    def save_csv(self, data_frame, delimiter=',', encoding='utf-8'):
        """
        Saves a DataFrame to a CSV file with specified encoding and delimiter.

        Args:
            data_frame (pd.DataFrame): The DataFrame to save.
            delimiter (str): Delimiter to use in the CSV file, default is ','.
            encoding (str): Encoding for the CSV file, default is 'utf-8'.

        Raises:
            Exception: If any error occurs during file writing.
        """
        try:
            data_frame.to_csv(self.file_path, sep=delimiter, encoding=encoding, index=False)
            print(f"Data saved successfully to '{self.file_path}' in CSV format.")
        except Exception as e:
            print(f"Failed to save data to '{self.file_path}': {str(e)}")
            raise

    def save_excel(self, data_frame, sheet_name='Sheet1'):
        """
        Saves a DataFrame to an Excel file.

        Args:
            data_frame (pd.DataFrame): The DataFrame to save.
            sheet_name (str): The name of the worksheet to use.

        Raises:
            Exception: If any error occurs during file writing.
        """
        try:
            data_frame.to_excel(self.file_path, sheet_name=sheet_name, index=False)
            print(f"Data saved successfully to '{self.file_path}' in Excel format.")
        except Exception as e:
            print(f"Failed to save data to '{self.file_path}': {str(e)}")
            raise

    def save_json(self, data_frame, orient='records'):
        """
        Saves a DataFrame to a JSON file.

        Args:
            data_frame (pd.DataFrame): The DataFrame to save.
            orient (str): The format of the JSON string.

        Raises:
            Exception: If any error occurs during file writing.
        """
        try:
            data_frame.to_json(self.file_path, orient=orient)
            print(f"Data saved successfully to '{self.file_path}' in JSON format.")
        except Exception as e:
            print(f"Failed to save data to '{self.file_path}': {str(e)}")
            raise

def create_zip_file(df, zip_filename):
    """
    Creates a zip file containing a CSV file from the given DataFrame.
    The zip file is created in the system's temporary directory.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved as a CSV file.
        zip_filename (str): The name of the zip file to be created.

    Returns:
        str: The path to the created zip file.
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Define the path for the CSV file
        csv_file = os.path.join(temp_dir, 'data.csv')
        # Save the DataFrame to a CSV file
        df.to_csv(csv_file, index=False)
        # Create a zip file containing the CSV
        zip_path = os.path.join(temp_dir, zip_filename)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(csv_file, os.path.basename(csv_file))
        # Return the path to the created zip file
        return zip_path
    except Exception as e:
        # Clean up the temporary directory if an exception occurs
        os.remove(temp_dir)
        raise e

def create_temp_csv(table_data):
    """
    Creates a temporary CSV file from a DataFrame.
    
    Args:
        table_data (DataFrame): The table data to be written to the CSV file.
        
    Returns:
        str: The path of the temporary CSV file.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as temp_file:
        table_data.to_csv(temp_file, index=False)
        temp_file_path = temp_file.name
    
    return temp_file_path

def get_dataset_tables(dataset_id, token_manager):
    """
    Retrieves the list of tables for a given dataset.
    
    Args:
        dataset_id (str): The ID of the dataset.
        token_manager (TokenManager): An instance of the TokenManager class.
    
    Returns:
        list: A list of tables in the dataset.
    """
    try:
        url = f"{API_URL}{DATASETS_ENDPOINT}{dataset_id}/table"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {token_manager.get_token()}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception if the request was not successful
        return response.json()["collection"]
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"Error getting dataset tables: {e}")
        return []

def get_table(dataset_id, table_id, token_manager):
    """
    Retrieves a table by its ID from a specific dataset.
    
    Args:
        dataset_id (str): The ID of the dataset.
        table_id (str): The ID of the table to retrieve.
        token_manager (TokenManager): An instance of the TokenManager class.
    
    Returns:
        dict: The table data in JSON format.
    """
    url = f"{API_URL}{DATASETS_ENDPOINT}{dataset_id}/table/{table_id}"
    headers = {
        "Authorization": f"Bearer {token_manager.get_token()}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception if the request was not successful
        table_data = response.json()
        
        return table_data
    
    except requests.RequestException as e:
        print(f"Error occurred while retrieving the table data: {e}")
        return None

def get_dataset_tables(dataset_id, token_manager):
    """
    Retrieves the list of tables for a given dataset.
    
    Args:
        dataset_id (str): The ID of the dataset.
        token_manager (TokenManager): An instance of the TokenManager class.
    
    Returns:
        list: A list of tables in the dataset.
    """
    try:
        url = f"{API_URL}{DATASETS_ENDPOINT}{dataset_id}/table"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {token_manager.get_token()}"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception if the request was not successful
        return response.json().get("collection", [])
    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"Error getting dataset tables: {e}")
        return []

def getReconciliatorData():
    """
    Retrieves reconciliator data from the backend
    :return: data of reconciliator services in JSON format
    """
    try:
        response = requests.get(API_URL + 'reconciliators/list')
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while retrieving reconciliator data: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error occurred while parsing reconciliator data: {e}")
        print(f"Response content: {response.text}")
        return None

def cleanServiceList(serviceList):
    """
    Cleans and formats the service list

    :serviceList: data regarding available services
    :return: dataframe containing reconciliators information
    """
    serviceList = serviceList
    reconciliators = pd.DataFrame(columns=["id", "relativeUrl", "name"])
    for reconciliator in serviceList:
        reconciliators.loc[len(reconciliators)] = [
            reconciliator["id"], reconciliator["relativeUrl"], reconciliator["name"]]
    return reconciliators

def getExtenderData():
    """
    Retrieves extender data from the backend

    :return: data of extension services in JSON format
    """
    response = requests.get(API_URL + 'extenders/list')
    return json.loads(response.text)

def getReconciliator(idReconciliator, response):
    """
    Function that, given the reconciliator's ID, returns a dictionary 
    with all the service information

    :idReconciliator: the ID of the reconciliator in question
    :return: a dictionary with the reconciliator's information
    """
    for reconciliator in response:
        if reconciliator['id'] == idReconciliator:
            return {
                'uri': reconciliator['uri'],
                'prefix': reconciliator['prefix'],
                'name': reconciliator['name'],
                'relativeUrl': reconciliator['relativeUrl']
            }
    return None

def createReconciliationPayload(table, columnName, idReconciliator):
    """
    Creates the payload for the reconciliation request

    :table: table in raw format
    :columnName: the name of the column to reconcile
    :idReconciliator: the id of the reconciliation service to use
    :return: the request payload
    """
    rows = []
    rows.append({"id": 'column$index', "label": columnName})
    for row in table['rows'].keys():
        rows.append({"id": row+"$"+columnName,
                    "label": table['rows'][row]['cells'][columnName]['label']})
    return {"serviceId": idReconciliator, "items": rows}

def parseNameField(name, uriReconciliator, idEntity):
    """
    The actual function that changes the name format to the one required for visualization

    :name: entity name
    :uriReconciliator: the URI of the affiliated knowledge graph
    :idEntity: entity ID
    :return: the name in the correct format
    """
    return {
        'value': name,
        'uri': uriReconciliator + idEntity
    }

def createCellMetadataNameField(metadata, idReconciliator, reconciliatorResponse):
    """
    Refactor of the name field within cell-level metadata
    necessary for visualization within SEMTUI

    :metadata: column-level metadata
    :idReconciliator: ID of the reconciliator performed in the operation
    :reconciliatorResponse: response containing reconciliator information
    :return: metadata containing the name field in the new format
    """
    for row in range(len(metadata)):
        try:
            for item in range(len(metadata[row]["metadata"])):
                value = metadata[row]["metadata"][item]['name']
                uri = metadata[row]["metadata"][item]['id']
                metadata[row]["metadata"][item]['name'] = parseNameField(value, getReconciliator(
                    idReconciliator, reconciliatorResponse)['uri'], uri.split(':')[1])
        except:
            return []
    return metadata

def calculateScoreBoundCell(metadata):
    """
    Calculates the min and max value of the score of the results obtained for
    a single cell

    :metadata: metadata of a single cell
    :return: a dictionary containing the two values
    """
    try:
        scoreList = [item['score'] for item in metadata]
        return {'lowestScore': min(scoreList), 'highestScore': max(scoreList)}
    except:
        return {'lowestScore': 0, 'highestScore': 0}

def valueMatchCell(metadata):
    """
    Returns whether a cell has obtained a match or not

    :metadata: cell-level metadata
    :return: True or False based on the match occurrence
    """
    for item in metadata:
        if item['match'] == True:
            return True
    return False

def createAnnotationMetaCell(metadata):
    """
    Creates the annotationMeta field at the cell level, 
    which will then be inserted into the table

    :metadata: cell-level metadata
    :return: the dictionary with data regarding annotationMeta
    """
    scoreBound = calculateScoreBoundCell(metadata)
    return {'annotated': True,
            'match': {'value': valueMatchCell(metadata)},
            'lowestScore': scoreBound['lowestScore'],
            'highestScore': scoreBound['highestScore']}

def updateMetadataCells(table, metadata):
    """
    Allows inserting new cell-level metadata

    :table: table in raw format
    :metadata: cell-level metadata
    :return: the table in raw format with metadata
    """
    for item in metadata:
        item["id"] = item["id"].split("$")
        try:
            table["rows"][item["id"][0]]["cells"][item["id"]
                                                  [1]]["metadata"] = item["metadata"]
            table["rows"][item["id"][0]]["cells"][item["id"][1]
                                                  ]["annotationMeta"] = createAnnotationMetaCell(item["metadata"])
        except:
            print("")
    return table

def calculateNCellsReconciliatedColumn(table, columnName):
    """
    Calculates the number of reconciled cells within 
    a column

    :table: table in raw format
    :columnName: name of the column in question
    :return: the number of reconciled cells
    """
    cellsReconciliated = 0
    rowsIndex = table["rows"].keys()
    for row in rowsIndex:
        try:
            if table['rows'][row]['cells'][columnName]['annotationMeta']["annotated"] == True:
                cellsReconciliated += 1
        except:
            cellsReconciliated = cellsReconciliated
    return cellsReconciliated

def createContextColumn(table, columnName, idReconciliator, reconciliatorResponse):
    """
    Creates the context field at the column level by retrieving the necessary data

    :table: table in raw format
    :columnName: the name of the column for which the context is being created
    :idReconciliator: the ID of the reconciliator used for the column
    :reconciliatorResponse: response containing reconciliator information
    :return: the context field of the column
    """
    nCells = len(table["rows"].keys())
    reconciliator = getReconciliator(idReconciliator, reconciliatorResponse)
    return {reconciliator['prefix']: {
            'uri': reconciliator['uri'],
            'total': nCells,
            'reconciliated': calculateNCellsReconciliatedColumn(table, columnName)
            }}

def getColumnMetadata(metadata):
    """
    Allows retrieving column-level data, particularly
    the entity corresponding to the column, the column types,
    and the match value of the entities in the column

    :metadata: column metadata obtained from the reconciliator
    :return: dictionary containing the different data
    """
    entity = []
    types = []
    for i in range(len(metadata)):
        try:
            if metadata[i]['id'] == ['column', 'index']:
                entity = metadata[i]['metadata']
        except:
            print("No column entity is provided")
        try:
            if metadata[i]['id'] != ['column', 'index']:
                for j in range(len(metadata[i]['metadata'])):
                    if metadata[i]['metadata'][j]['match'] == True:
                        types.append(metadata[i]['metadata'][j]['type'][0])
        except:
            print("No column type is provided")
    matchMetadataValue = True
    for item in entity:
        if item['match'] == False:
            matchMetadataValue = False
    return {'entity': entity, 'type': types, 'matchMetadataValue': matchMetadataValue}

def createMetadataFieldColumn(metadata):
    """
    Allows creating the metadata field for a column, which will
    then be inserted into the general column-level metadata

    :metadata: column-level metadata
    :return: the metadata field at the column level
    """
    return [
        {'id': '',
         'match': getColumnMetadata(metadata)['matchMetadataValue'],
         'score': 0,
         'name':{'value': '', 'uri': ''},
         'entity': getColumnMetadata(metadata)['entity'],
         'property':[],
         'type': getColumnMetadata(metadata)['type']}
    ]

def calculateScoreBoundColumn(table, columnName, reconciliatorResponse):
    allScores = []
    matchValue = True
    rows = table["rows"].keys()
    for row in rows:
        try:
            annotationMeta = table["rows"][row]['cells'][columnName]['annotationMeta']
            if annotationMeta['annotated'] == True:
                allScores.append(annotationMeta['lowestScore'])
                allScores.append(annotationMeta['highestScore'])
            if annotationMeta['match']['value'] == False:
                matchValue = False
        except KeyError:
            print(f"Missing key in cell annotation metadata: 'annotationMeta'")
            print(f"Row: {row}, Column: {columnName}")
            print(f"Cell data: {table['rows'][row]['cells'][columnName]}")
    
    if allScores:
        return {'lowestScore': min(allScores), 'highestScore': max(allScores), 'matchValue': matchValue}
    else:
        print("No valid annotation metadata found for the column.")
        return {'lowestScore': None, 'highestScore': None, 'matchValue': None}

def createAnnotationMetaColumn(annotated, table, columnName, reconciliatorResponse):
    scoreBound = calculateScoreBoundColumn(
        table, columnName, reconciliatorResponse)
    return {'annotated': annotated,
            'match': {'value': scoreBound['matchValue']},
            'lowestScore': scoreBound['lowestScore'],
            'highestScore': scoreBound['highestScore']
            }

def updateMetadataColumn(table, columnName, idReconciliator, metadata, reconciliatorResponse):
    """
    Allows inserting column-level metadata

    :table: table in raw format
    :columnName: name of the column to operate on
    :idReconciliator: ID of the reconciliator used
    :metadata: column-level metadata
    :reconciliatorResponse: response containing reconciliator information
    :return: the table with the new metadata inserted
    """
    # inquire about the different states
    table['columns'][columnName]['status'] = 'pending'
    table['columns'][columnName]['kind'] = "entity"
    table['columns'][columnName]['context'] = createContextColumn(
        table, columnName, idReconciliator, reconciliatorResponse)
    table['columns'][columnName]['metadata'] = createMetadataFieldColumn(
        metadata)
    table['columns'][columnName]['annotationMeta'] = createAnnotationMetaColumn(
        True, table, columnName, reconciliatorResponse)
    return table

def calculateScoreBoundTable(table):
    """
    Calculates the minimum and maximum score obtained in
    the results of the entire table

    :table: the table in raw format
    :return: a dictionary containing the two values
    """
    allScores = []
    reconciliateColumns = [column for column in table['columns'].keys(
    ) if table['columns'][column]['status'] != 'empty']
    for column in reconciliateColumns:
        try:
            if table['columns'][column]['annotationMeta']['annotated'] == True:
                allScores.append(table['columns'][column]
                                 ['annotationMeta']['lowestScore'])
                allScores.append(table['columns'][column]
                                 ['annotationMeta']['highestScore'])
        except:
            print("Missed column annotation metadata")
    try:
        return {'lowestScore': min(allScores), 'highestScore': max(allScores)}
    except:
        return {'lowestScore': 0, 'highestScore': 0}

def calculateNCellsReconciliated(table):
    """
    Calculates the number of reconciled cells within the
    table

    :table: the table in raw format
    :return: the number of reconciled cells
    """
    cellsReconciliated = 0
    columnsName = table['columns'].keys()
    for column in columnsName:
        try:
            contextReconciliator = table['columns'][column]['context'].keys()
            for reconcliator in contextReconciliator:
                cellsReconciliated += int(table['columns'][column]
                                          ['context'][reconcliator]['reconciliated'])
        except:
            cellsReconciliated += 0
    return cellsReconciliated

def updateMetadataTable(table):
    """
    Inserts metadata at the table level

    :table: table in raw format
    :return: the table with the new metadata inserted
    """
    scoreBound = calculateScoreBoundTable(table)
    table['table']['minMetaScore'] = scoreBound['lowestScore']
    table['table']['maxMetaScore'] = scoreBound['highestScore']
    table['table']['nCellsReconciliated'] = calculateNCellsReconciliated(table)
    return table

def getExtender(idExtender, response):
    """
    Given the extender's ID, returns the main information in JSON format

    :idExtender: the ID of the extender in question
    :response: JSON containing information about the extenders
    :return: JSON containing the main information of the extender
    """
    for extender in response:
        if extender['id'] == idExtender:
            return {
                'name': extender['name'],
                'relativeUrl': extender['relativeUrl']
            }
    return None

def createExtensionPayload(data, reconciliatedColumnName, properties, idExtender, dates, weatherParams, decimalFormat=None):
    """
    Creates the payload for the extension request

    :param data: table in raw format
    :param reconciliatedColumnName: the name of the column containing reconciled id
    :param properties: the properties to use in a list format
    :param idExtender: the ID of the extender service
    :param dates: a dictionary containing the date information for each row
    :param weatherParams: a list of weather parameters to include in the request
    :param decimalFormat: the decimal format to use for the values (default: None)
    :return: the request payload
    """
    items = {}
    if 'rows' not in data:
        raise KeyError("The 'data' dictionary does not contain the 'rows'")
    rows = data['rows'].keys()
    for row in rows:
        if 'cells' not in data['rows'][row]:
            raise KeyError(f"The 'data['rows'][{row}]' dictionary does not contain the 'cells' key.")
        
        cell = data['rows'][row]['cells'].get(reconciliatedColumnName)
        if cell and cell.get('annotationMeta', {}).get('match', {}).get('value') == True:
            for metadata in cell.get('metadata', []):
                if metadata.get('match') == True:
                    items[row] = metadata.get('id')
                    break    
    payload = {
        "serviceId": idExtender,
        "items": {
            str(reconciliatedColumnName): items
        },
        "property": properties,
        "dates": dates,
        "weatherParams": weatherParams,
        "decimalFormat": decimalFormat or []
    }
    
    return payload

def getReconciliatorFromPrefix(prefixReconciliator, response):
    """
    Function that, given the reconciliator's prefix, returns a dictionary 
    with all the service information

    :prefixReconciliator: the prefix of the reconciliator in question
    :return: a dictionary with the reconciliator's information
    """
    for reconciliator in response:
        if reconciliator['prefix'] == prefixReconciliator:
            return {
                'uri': reconciliator['uri'],
                'id': reconciliator['id'],
                'name': reconciliator['name'],
                'relativeUrl': reconciliator['relativeUrl']
            }
    return None

def getColumnIdReconciliator(table, columnName, reconciliatorResponse):
    """
    Specifying the column of interest returns the reconciliator's ID,
    if the column is reconciled

    :table: table in raw format
    :columnName: name of the column in question
    :return: the ID of the reconciliator used
    """
    prefix = list(table['columns'][columnName]['context'].keys())
    return getReconciliatorFromPrefix(prefix[0], reconciliatorResponse)['id']

def checkEntity(newColumnData):
    entity = False
    # Assuming newColumnData refers to a row payload, which contains a 'cells' list
    for cell in newColumnData['cells']:
        # Check if there is metadata and it's not empty
        if 'metadata' in cell and cell['metadata']:
            entity = True
            break
    return entity

def parseNameEntities(entities, uriReconciliator):
    """
    Function iterated in parseNameMetadata, works at the entity level

    :param entities: List of entities present in the cell/column
    :param uriReconciliator: the URI of the affiliated knowledge graph
    :return: List of entities with updated names
    """
    for entity in entities:
        if 'id' in entity and ':' in entity['id']:
            entity_type = entity['id'].split(':')[1]  # Safely extract after colon
            entity['name'] = parseNameField(
                entity.get('name', ''),  # Safely access 'name'
                uriReconciliator,
                entity_type
            )
    return entities

def addExtendedCell(table, newColumnData, newColumnName, idReconciliator, reconciliatorResponse):
    if 'cells' not in newColumnData:
        raise ValueError("newColumnData must contain 'cells'")
    
    rowKeys = newColumnData['cells']
    entity = checkEntity(newColumnData)
    columnType = newColumnData.get('kind', 'entity' if entity else 'literal')

    for rowKey in rowKeys:
        cellData = newColumnData['cells'][rowKey]
        newCell = table['rows'][rowKey]['cells'].setdefault(newColumnName, {})
        newCell['id'] = f"{rowKey}${newColumnName}"
        newCell['label'] = cellData.get('label', '')

        uriReconciliator = getReconciliator(idReconciliator, reconciliatorResponse)['uri']
        newCell['metadata'] = parseNameEntities(cellData.get('metadata', []), uriReconciliator)

        if columnType == 'entity':
            newCell['annotationMeta'] = createAnnotationMetaCell(newCell['metadata'])
        else:
            newCell['annotationMeta'] = {}

    return table

def addExtendedColumns(table, extensionData, newColumnsName, reconciliatorResponse):
    """
    Allows iterating the operations to insert a single column for
    all the properties to be inserted.

    :param table: table in raw format
    :param extensionData: data obtained from the extender
    :param newColumnsName: names of the new columns to insert into the table
    :param reconciliatorResponse: response containing reconciliator information
    :return: the table with the new fields inserted
    """
    if 'columns' not in extensionData or 'meta' not in extensionData:
        raise ValueError("extensionData must contain 'columns' and 'meta'")

    # Iterating through new columns to be added
    for i, columnKey in enumerate(extensionData['columns'].keys()):
        if i >= len(newColumnsName):
            raise IndexError("There are more columns to add than names provided.")
        
        # Fetching reconciliator ID for the current column
        idReconciliator = getColumnIdReconciliator(
            table, extensionData['meta'][columnKey], reconciliatorResponse)
        
        # Adding the extended cell/column to the table
        table = addExtendedCell(
            table, extensionData['columns'][columnKey], newColumnsName[i], idReconciliator, reconciliatorResponse)
        
    return table

def parseNameMetadata(metadata, uriReconciliator):
    if not isinstance(metadata, list):
        raise ValueError("Expected metadata to be a list")
    
    for item in metadata:
        if 'entity' in item:
            try:
                item['entity'] = parseNameEntities(item['entity'], uriReconciliator)
            except KeyError as e:
                raise KeyError(f"Missing expected key in entity data: {str(e)}")
            except Exception as e:
                raise Exception(f"An error occurred while parsing entities: {str(e)}")
        else:
            raise KeyError("Expected 'entity' key in each metadata item")

    return metadata

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


