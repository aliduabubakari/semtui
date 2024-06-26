# Import key classes and functions from the modules to make them
# available at the package level
from .semtui import (
    obtain_auth_token,
    process_csv_data,
    process_data,
    add_dataset,
    get_database_list,
    delete_dataset,
    get_table_by_name,
    delete_table,
    add_table_to_dataset,
    update_table,
    list_tables_in_dataset,
    getReconciliatorsList,
    getExtendersList,
    reconcile,
    getReconciliatorParameters,
    getExtenderParameters,
    generateExtendColumnGuide,
    generateReconcileGuide,
    extract_georeference_data,
    extract_city_reconciliation_metrics,
    extendColumn,
    create_reconciliation_payload_for_backend,
    evaluate_reconciliation,
    extract_nested_values_reconciliation,
    extend_Reconciliation_Results, 
    push_reconciliation_data_to_backend, 
    load_json_to_dataframe
)

from .utils import (
    TokenManager,
    FileReader,
    FileSaver,
    create_zip_file,
    create_temp_csv,
    get_dataset_tables,
    get_table,
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
    update_table
)

# Define what should be accessible when importing the package
__all__ = [
    'obtain_auth_token',
    'process_csv_data',
    'process_data',
    'add_dataset',
    'get_database_list',
    'delete_dataset',
    'get_table_by_name',
    'delete_table',
    'add_table_to_dataset',
    'update_table',
    'list_tables_in_dataset',
    'getReconciliatorsList',
    'getExtendersList',
    'reconcile',
    'getReconciliatorParameters',
    'getExtenderParameters',
    'generateExtendColumnGuide',
    'generateReconcileGuide',
    'extract_georeference_data',
    'extract_city_reconciliation_metrics',
    'extendColumn',
    'create_reconciliation_payload_for_backend',
    'evaluate_reconciliation',
    'extract_nested_values_reconciliation',
    'create_extension_payload_for_backend',
    'load_json_to_dataframe',
    'TokenManager',
    'FileReader',
    'FileSaver',
    'create_zip_file',
    'create_temp_csv',
    'get_dataset_tables',
    'get_table',
    'getReconciliatorData',
    'cleanServiceList',
    'getExtenderData',
    'getReconciliator',
    'createReconciliationPayload',
    'parseNameField',
    'createCellMetadataNameField',
    'calculateScoreBoundCell',
    'valueMatchCell',
    'createAnnotationMetaCell',
    'updateMetadataCells',
    'calculateNCellsReconciliatedColumn',
    'createContextColumn',
    'getColumnMetadata',
    'createMetadataFieldColumn',
    'calculateScoreBoundColumn',
    'createAnnotationMetaColumn',
    'updateMetadataColumn',
    'calculateScoreBoundTable',
    'calculateNCellsReconciliated',
    'updateMetadataTable',
    'getExtender',
    'createExtensionPayload',
    'getReconciliatorFromPrefix',
    'getColumnIdReconciliator',
    'checkEntity',
    'parseNameEntities',
    'addExtendedCell',
    'addExtendedColumns',
    'parseNameMetadata',
    'update_table',
    'extend_Reconciliation_Results',
    'push_reconciliation_data_to_backend'
]

# Optional: Initialize anything that needs to be set up when the package is loaded
# e.g., logging setup, configuration checks, etc.
