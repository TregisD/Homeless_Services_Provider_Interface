import os
import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher
from pathlib import Path

def find_files(directory, filetype='csv'):
    csv_files = []
    sub_paths = []
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(f".{filetype}"):
                csv_files.append(filename)
                sub_paths.append(dirpath)
    
    return csv_files, sub_paths

def store_triples_into_neo4j(csv_file, first_flag):
    file_path = Path(csv_file)  # Convert string to Path object
    
    if not file_path.exists():
        raise ValueError(f'Triples file {file_path} does not exist')
    
    graph = Graph("bolt://localhost:7687", auth=("neo4j", "put_db_password_here"))
    
    if first_flag:
        graph.delete_all()
    
    df = pd.read_csv(file_path)
    
    if df.shape[1] != 3:
        raise ValueError(f"CSV file {csv_file} does not have exactly three columns.")
    
    for _, row in df.iterrows():
        triple_subject, triple_relation, triple_object = map(str.strip, row[:3])
        
        matcher = NodeMatcher(graph)
        subject_list = list(matcher.match('node', name=triple_subject))
        object_list = list(matcher.match('node', name=triple_object))
        
        if subject_list:
            if object_list:
                relation = Relationship(subject_list[0], triple_relation, object_list[0])
                graph.create(relation)
            else:
                object_node = Node('node', name=triple_object)
                relation = Relationship(subject_list[0], triple_relation, object_node)
                graph.create(relation)
        else:
            if object_list:
                subject_node = Node('node', name=triple_subject)
                relation = Relationship(subject_node, triple_relation, object_list[0])
                graph.create(relation)
            else:
                subject_node = Node('node', name=triple_subject)
                object_node = Node('node', name=triple_object)
                relation = Relationship(subject_node, triple_relation, object_node)
                graph.create(relation)

def main():
    csv_file = "Triples.csv"  # This is just a single filename, not a list
    first_flag = True
    
    # Remove the loop since we're only processing one file
    if csv_file != 'requirements.csv':
        print(f'Writing triples from {csv_file} into Neo4j...')
        store_triples_into_neo4j(csv_file, first_flag)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()