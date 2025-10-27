import csv
import pandas as pd
import ast

def parse_list_string(value):
    """
    Parse a string that looks like a Python list into actual list items.
    Handles formats like: "['item1', 'item2']" or just "item1, item2"
    """
    value = value.strip()
    
    # Try to parse as a Python literal (list)
    if value.startswith('[') and value.endswith(']'):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
        except (ValueError, SyntaxError):
            pass
    
    # If not a list format, check if it's comma-separated
    if ',' in value:
        return [item.strip() for item in value.split(',')]
    
    # Otherwise, return as single item
    return [value]

def csv_to_triples(csv_file, relationship_mapping, multi_value_columns=None):
    """
    Convert CSV to triples format.
    
    Args:
        csv_file: Path to CSV file
        relationship_mapping: Dictionary mapping column names to relationships
        multi_value_columns: List of column names that may contain multiple values
    
    Returns:
        List of triples (subject, relationship, object)
    """
    if multi_value_columns is None:
        multi_value_columns = []
    
    triples = []
    
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader) 
        
        for row in reader:
            subject = row[0].strip()  # First column as subject
            
            for i in range(1, len(row)):
                column_name = headers[i].strip()
                if column_name in relationship_mapping:  # Skip columns not in mapping
                    relationship = relationship_mapping[column_name]
                    obj = row[i].strip()
                    
                    if obj:
                        # Check if this column should be split into multiple triples
                        if column_name in multi_value_columns:
                            # Parse and create separate triple for each value
                            values = parse_list_string(obj)
                            for value in values:
                                if value:  # Skip empty strings
                                    triples.append((subject, relationship, value))
                        else:
                            # Single triple for this value
                            triples.append((subject, relationship, obj))
    
    return triples

def process_multiple_csvs(csv_files, relationship_mapping, output_csv, multi_value_columns=None):
    """
    Process multiple CSV files and combine their triples into a single output file.
    
    Args:
        csv_files: List of CSV file paths
        relationship_mapping: Dictionary mapping column names to relationships
        output_csv: Path for the output triples CSV
        multi_value_columns: List of column names that may contain multiple values
    
    Returns:
        List of all triples from all files
    """
    all_triples = []
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        triples = csv_to_triples(
            csv_file, 
            relationship_mapping, 
            multi_value_columns
        )
        all_triples.extend(triples)
        print(f"  - Added {len(triples)} triples")
    
    # Write all triples to output file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Subject", "Relationship", "Object"]) 
        writer.writerows(all_triples)
    
    return all_triples

# Custom relationships
relationship_mapping = {
    "Main_Services": "main_services_are",
    "Other_Services": "other_services_are",
    "Phone_Number": "phone_number_is",
    "Website": "website_is",
    "Location_Address": "is_located_at",
    "Location_URL_Map": "url_is_located_at",
    "Eligibility": "eligibility_is",
    "Availability": "has_an_availability_status_of",
    "Description": "description_is",
    "Languages": "speaks",
    "Cost": "costs",
    "Google_Review": "has_Google_Reviews",
    "Facebook_URL": "facebook_is",
    "Twitter_URL": "twitter_is",
    "Serving": "serves",
    "Coverage": "covers",
    "Latitude": "latitude_is",
    "Longitude": "longitude_is",
    "Zipcode": "zipcode_is",
    "24hour": "24hours_status_is",
    "Monday": "Monday's_time_open",
    "Tuesday": "Tuesday's_time_open",
    "Wednesday": "Wednesday's_time_open",
    "Thursday": "Thursday's_time_open",
    "Friday": "Friday's_time_open",
    "Saturday": "Saturday's_time_open",
    "Sunday": "Sunday's_time_open",
    "Google_Rating": "has_a_Google_Rating_of",
    "Service_Type": "offers"
}

# Specify which columns contain multiple values that should be split
multi_value_columns = [
    "Other_Services",
    "Main_Services",
    "Serving",
    "Eligibility",
    "Languages"
]

if __name__ == "__main__":
    # List all your CSV files here
    csv_files = [
        '../Leo/Riverside/FindHelp_extracted_data_riv_mental_health.csv',
        '../Leo/Riverside/FindHelp_extracted_data_riv_shelter.csv',
        '../Leo/Riverside/FindHelp_extracted_data_riv_food_pantry.csv',
        '../Leo/LA/FindHelp_extracted_data_la_mental_health.csv',
        '../Leo/LA/FindHelp_extracted_data_la_shelter.csv',
        '../Leo/LA/FindHelp_extracted_data_la_food_pantry.csv'
    ]
    
    output_csv = "Triples.csv"
    
    all_triples = process_multiple_csvs(
        csv_files,
        relationship_mapping,
        output_csv,
        multi_value_columns
    )
    
    print(f"\n{'='*50}")
    print(f"Successfully created {len(all_triples)} total triples from {len(csv_files)} files!")
    print(f"Output saved to: {output_csv}")
    print(f"{'='*50}\n")