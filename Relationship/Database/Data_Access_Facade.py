import dotenv
from neo4j import GraphDatabase
import os

dotenv.load_dotenv()
driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

def convertProperties_to_string(properties:dict)->str:
    return ', '.join([f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}" for key, value in properties.items()])

def getWhereClause(prefixName:str,properties:dict)->str:
    return ', '.join([f"{prefixName}.{key} = '{value}'" if isinstance(value, str) else f"n.{key} = {value}" for key, value in properties.items()])

def checkIfNodeExist(label:str, properties:dict)->bool:
    session = driver.session()
    return session.run(f"""
                            MATCH (n:{label})
                            WHERE {getWhereClause("n",properties)}
                            return n
                        """).single() != None

def checkIfRelationshipExist(label1:str, label2:str, id1:dict, id2:dict, propertyName:str)->bool:
    session = driver.session()
    return session.run(f"""
                            MATCH (a:{label1})-[r:{propertyName}]->(b:{label2}) 
                            WHERE {getWhereClause("a",id1)} AND {getWhereClause("b",id2)}
                            RETURN r
                        """).single() != None

def createNode(label:str, properties:dict):
    properties_str = convertProperties_to_string(properties)
    if checkIfNodeExist(label, properties):
        return 0
    session = driver.session()
    return session.run(f"CREATE (:{label} {{{properties_str}}})")

def updateNode(label:str, id:dict, property_key:str, new_value):
    session = driver.session()
    status = session.run(f"""
                            MATCH (n:{label})
                            WHERE {getWhereClause("n",id)} 
                            SET n.{property_key} = {new_value} RETURN n
                        """)
    return status.single()

def createRelationship(label1:str, label2:str, id1:dict, id2:dict, propertyName:str ,properties:dict):
    properties_str = convertProperties_to_string(properties)
    label1Ids = convertProperties_to_string(id1)
    label2Ids = convertProperties_to_string(id2)
    if checkIfRelationshipExist(label1, label2, label1Ids, label2Ids, propertyName):
        return 0
    
    session = driver.session()
    status = session.run(f"""
                            MATCH (a:{label1}), (b:{label2})
                            WHERE {getWhereClause("a",id1)} AND {getWhereClause("b",id2)} 
                            CREATE (a)-[:{propertyName} {{{properties_str}}}]->(b)
                        """)
    return status
    
def printTest(label, properties):
    print("CREATE (:{label} {{{properties}}})".format(label=label, properties=convertProperties_to_string(properties)))
