"""
Structured Data Loading - Part 1b

Shows how to:
- Load CSV data with LangChain CSVLoader
- Handle JSON documents with JSONLoader
"""
from langchain_community.document_loaders import CSVLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tempfile
import os
import csv
import json

def create_sample_csv():
    """Create sample CSV data for demonstration."""
    csv_data = [
        {"product_name": "TechCorp AI Studio", "category": "AI Platform", "price": "99", "description": "No-code ML platform with drag-and-drop interface"},
        {"product_name": "TechCorp Compute", "category": "Infrastructure", "price": "0.05", "description": "Virtual machines with auto-scaling capabilities"},
        {"product_name": "TechCorp Database", "category": "Database", "price": "25", "description": "Managed PostgreSQL with automated backups"},
        {"product_name": "TechCorp Storage", "category": "Storage", "price": "0.02", "description": "Object storage with CDN integration"},
        {"product_name": "TechCorp Analytics", "category": "Analytics", "price": "50", "description": "Real-time data analytics and visualization"}
    ]
    return csv_data

def demonstrate_csv_loading():
    """Demonstrate CSV loading with different configurations."""
    print("=== CSV Data Loading ===\n")
    
    # Create sample CSV file
    csv_data = create_sample_csv()
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='') as temp_file:
        writer = csv.DictWriter(temp_file, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)
        temp_csv = temp_file.name
    
    try:
        print(f"Loading CSV with CSVLoader from {temp_csv}")
        
        # Basic CSV loading
        # The CSVLoader is configured to put specific columns (product_name, category, price) into metadata via the metadata_columns parameter. 
        # Only the columns NOT listed in metadata_columns remain in the page_content.
        loader = CSVLoader(
            file_path=temp_csv,
            source_column="product_name",  # Use product name as source identifier. Default file path
             metadata_columns=["category", "price", "product_name"] # excludes from page_content and puts metadata
        )
        documents = loader.load()
        
        print(f"Loaded {len(documents)} CSV records:")
        for i, doc in enumerate(documents, 1):
            print(f"Record {i}:")
            print(doc)
            print("==" * 30)
        print()
        
        return documents
        
    finally:
        os.unlink(temp_csv)

def create_sample_json():
    """Create sample JSON data for demonstration."""
    json_data = {
        "services": [
            {
                "name": "AI Studio",
                "type": "platform",
                "features": ["drag-and-drop", "automated-ml", "one-click-deploy"],
                "pricing": {"basic": 99, "pro": 199, "enterprise": 499}
            },
            {
                "name": "EdgeCompute", 
                "type": "infrastructure",
                "features": ["global-edge", "5ms-latency", "auto-scale"],
                "pricing": {"basic": 0.05, "pro": 0.08, "enterprise": 0.12}
            }
        ],
        "company_info": {
            "support": ["chat", "email", "phone"],
            "hours": "24/7",
            "response_time": "< 1 hour"
        }
    }
    return json_data

def demonstrate_json_loading():
    """Demonstrate JSON loading with fallback handling."""
    print("=== JSON Data Loading ===\n")
    
    # Create sample JSON file
    json_data = create_sample_json()
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(json_data, temp_file, indent=2)
        temp_json = temp_file.name
    
    try:
        print(f"Attempting JSONLoader from {temp_json}")
        
        
        # metadata_func allows custom metadata extraction from each JSON record
        # It receives the parsed JSON object and existing metadata, returns updated metadata
        def metadata_func(record: dict, metadata: dict) -> dict:
            """Extract additional metadata from JSON records."""
            metadata["service_name"] = record.get("name", "unknown")
            metadata["service_type"] = record.get("type", "unknown")
            metadata["feature_count"] = len(record.get("features", []))
            return metadata
        
        # Try using JSONLoader (requires jq package)
        loader = JSONLoader(
            file_path=temp_json,
            jq_schema='.services[]',  # Extract services array
            text_content=False,
            metadata_func=metadata_func
        )
        documents = loader.load()
        print("JSONLoader successful!")

        print(f"Loaded {len(documents)} JSON records:")
        for i, doc in enumerate(documents, 1):
            print(f"Record {i}:")
            print(doc)
            print("==" * 30)
        print()

        return documents
    
    except Exception as e:
        print(f"JSONLoader not available ({e})")
             
    finally:
        os.unlink(temp_json)

if __name__ == "__main__":
    
    #demonstrate_csv_loading()
    demonstrate_json_loading()