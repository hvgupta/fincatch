import dotenv
from neo4j import GraphDatabase
import os

dotenv.load_dotenv()
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

def convertProperties_to_string(properties:dict)->str:
    return ', '.join([f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}" for key, value in properties.items()])

def checkIfNodeExist(label:str, properties:dict)->bool:
    properties_str = convertProperties_to_string(properties)
    return driver.run(f"MATCH (n:{label} {{{properties_str}}}) RETURN n").single() != None

def checkIfRelationshipExist(label1:str, label2:str, id1:str, id2:str, propertyName:str)->bool:
    return driver.run(f"""
                            MATCH (a:{label1} {{{id1}}})-[r:{propertyName}]->(b:{label2} {{{id2}}}) 
                            RETURN r
                        """).single() != None

def create_node(label:str, properties:dict):
    properties_str = convertProperties_to_string(properties)
    if checkIfNodeExist(label, properties):
        return 0
    
    status = driver.run(f"CREATE (:{label} {{{properties_str}}})")
    return status.summary().counters.nodes_created 

def update_node_property(label:str, id:str, property_key:str, new_value):
    if isinstance(new_value, str):
        new_value = f"'{new_value}'"
    status = driver.run(f"""
                            MATCH (n:{label} {{id: '{id}'}}) 
                            SET n.{property_key} = {new_value} RETURN n
                        """)
    return status.single()

def createRelationship(label1:str, label2:str, id1:dict, id2:dict, propertyName:str ,properties:dict):
    properties_str = convertProperties_to_string(properties)
    label1Ids = convertProperties_to_string(id1)
    label2Ids = convertProperties_to_string(id2)
    if checkIfRelationshipExist(label1, label2, label1Ids, label2Ids, propertyName):
        return 0
    status = driver.run(f"""
                            MATCH (a:{label1} {{{label1Ids}}}), (b:{label2} {{id: '{{{label2Ids}}}'}}) 
                            CREATE (a)-[:{propertyName} {{{properties_str}}}]->(b)
                        """)
    return status.summary().counters.relationships_created
    
    
    
def printTest(label, properties):
    print("CREATE (:{label} {{{properties}}})".format(label=label, properties=convertProperties_to_string(properties)))
