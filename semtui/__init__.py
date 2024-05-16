# SEMT-py/semtui/__init__.py

# Import key classes and functions from the modules to make them
# available at the package level
from .semtui import (
    obtain_auth_token, process_csv_data, add_dataset, add_table_to_dataset, delete_dataset, delete_table, get_database_list, delete_dataset, 
    get_table_by_name, getReconciliatorsList, getExtendersList, reconcile, load_json_to_dataframe, list_tables_in_dataset, 
    extract_georeference_data, extract_city_reconciliation_metrics, extendColumn, 
    create_extension_payload_for_backend, create_reconciliation_payload_for_backend
)

from .utils import (
    TokenManager, FileReader, FileSaver, create_zip_file, get_dataset_tables, get_table, 
    cleanServiceList, getReconciliatorData, getExtenderData, getReconciliator, get_dataset_tables, 
    createReconciliationPayload, updateMetadataTable, createCellMetadataNameField, 
    updateMetadataCells, updateMetadataColumn, getExtender, createExtensionPayload, 
    parseNameField, calculateScoreBoundCell, valueMatchCell, createAnnotationMetaCell, 
    calculateNCellsReconciliatedColumn, createContextColumn, getColumnMetadata, 
    createMetadataFieldColumn, calculateScoreBoundColumn, createAnnotationMetaColumn, 
    updateMetadataColumn, calculateScoreBoundTable, calculateNCellsReconciliated, 
    updateMetadataTable, addExtendedColumns, update_table
)

# Define what should be accessible when importing the package
__all__ = [
    'obtain_auth_token', 'process_csv_data', 'add_dataset', 'get_database_list', 
    'get_table_by_name', 'getReconciliatorsList', 'getExtendersList', 'reconcile',
    'extract_georeference_data', 'extract_city_reconciliation_metrics', 'extendColumn', 
    'create_extension_payload_for_backend', 'create_reconciliation_payload_for_backend', 
    'TokenManager', 'FileReader', 'create_zip_file', 'get_dataset_tables', 'get_table', 
    'cleanServiceList', 'getReconciliatorData', 'getExtenderData', 'getReconciliator', 
    'createReconciliationPayload', 'updateMetadataTable', 'createCellMetadataNameField', 
    'updateMetadataCells', 'updateMetadataColumn', 'getExtender', 'createExtensionPayload', 
    'parseNameField', 'calculateScoreBoundCell', 'valueMatchCell', 'createAnnotationMetaCell', 
    'calculateNCellsReconciliatedColumn', 'createContextColumn', 'getColumnMetadata', 
    'createMetadataFieldColumn', 'calculateScoreBoundColumn', 'createAnnotationMetaColumn', 
    'updateMetadataColumn', 'calculateScoreBoundTable', 'calculateNCellsReconciliated', 
    'updateMetadataTable', 'addExtendedColumns', 'update_table'
]

# Optional: Initialize anything that needs to be set up when the package is loaded
# e.g., logging setup, configuration checks, etc.
