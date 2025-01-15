from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
import json

load_dotenv()
driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
session = driver.session()

def generateClusters(outputFile:str)->None:
    community_dict = {}

    response_text = session.run("""
                            MATCH (n:Word)-[r:text]->(:Website)
                            WITH n.name as name, max(r.count) as Maxcount
                            MATCH (n:Word)-[r:text]->(m:Website)
                            WHERE n.name = name AND r.count = Maxcount AND r.count > 1
                            RETURN n.name as word,m.name as website
                        """)

    temp = {}
    for record in response_text:
        if record[0] not in temp:
            temp[record[0]] = []
        temp[record[0]].append(record[1])
        
    for key, value in temp.items():
        new_key = "|".join(value)
        if new_key not in community_dict:
            community_dict[new_key] = []
        community_dict[new_key].append(key)
        
    with open(outputFile, "w") as file:
        json.dump(community_dict, file)