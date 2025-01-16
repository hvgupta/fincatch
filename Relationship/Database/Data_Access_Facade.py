import dotenv
from neo4j import GraphDatabase
import os

# Load environment variables from a .env file into the system's environment variables.
dotenv.load_dotenv()

# Initialize a Neo4j database driver to connect to the database.
# The connection parameters are retrieved from environment variables for security,
# ensuring sensitive information like URI, username, and password are not hardcoded.
driver = GraphDatabase.driver(
    uri=os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# Create a new session with the Neo4j database driver.
session = driver.session()

def convertProperties_to_string(properties:dict)->str:
      """
    Convert a dictionary of properties into a formatted string suitable for Cypher queries.

    This function takes a dictionary where keys are property names and values 
    are the corresponding property values, and formats it into a string that 
    can be used in a Cypher query. String values are wrapped in single quotes, 
    while numeric values are left as is.

    Parameters:
    properties (dict): A dictionary containing properties to convert.

    Returns:
    str: A formatted string representing the properties, suitable for use in 
         a Cypher query.
    """
    return ', '.join([f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}" for key, value in properties.items()])

def getWhereClause(prefixName:str,properties:dict)->str:
     """
    Generate a WHERE clause for a Cypher query based on given properties.

    This function constructs a WHERE clause that can be used in a Cypher 
    query by iterating over a dictionary of properties. It formats the 
    properties into conditions that compare each property to its value, 
    using the specified prefix for the node.

    Parameters:
    prefixName (str): The prefix to use for the node in the WHERE clause.
    properties (dict): A dictionary containing the properties to include in 
                       the WHERE clause.

    Returns:
    str: A formatted string representing the WHERE conditions for the 
         Cypher query.
    """
    return ' AND '.join([f"{prefixName}.{key} = '{value}'" if isinstance(value, str) else f"n.{key} = {value}" for key, value in properties.items()])

def checkIfNodeExist(label:str, properties:dict)->bool:
     """
    Check if a node exists in the graph database based on its label and properties.

    This function executes a Cypher query to determine whether a node with 
    the specified label and matching properties exists in the database. 
    It uses the `getWhereClause` function to construct the WHERE conditions.

    Parameters:
    label (str): The label of the node to check for existence.
    properties (dict): A dictionary of properties that the node must have.

    Returns:
    bool: Returns True if the node exists; otherwise, returns False.
    """
    return session.run(f"""
                            MATCH (n:{label})
                            WHERE {getWhereClause("n",properties)}
                            return n
                        """).single() != None

def checkIfRelationshipExist(label1:str, label2:str, id1:dict, id2:dict, propertyName:str)->bool:
     """
    Check if a relationship exists between two nodes in the graph database.

    This function executes a Cypher query to determine whether a relationship 
    of the specified type exists between two nodes identified by their labels 
    and properties. It uses the `getWhereClause` function to construct the 
    WHERE conditions for both nodes.

    Parameters:
    label1 (str): The label of the first node.
    label2 (str): The label of the second node.
    id1 (dict): A dictionary representing the identifier(s) for the first node.
    id2 (dict): A dictionary representing the identifier(s) for the second node.
    propertyName (str): The name of the relationship type to check.

    Returns:
    bool: Returns True if the relationship exists; otherwise, returns False.
    """
    return session.run(f"""
                            MATCH (a:{label1})-[r:{propertyName}]->(b:{label2}) 
                            WHERE {getWhereClause("a",id1)} AND {getWhereClause("b",id2)}
                            RETURN r
                        """).single() != None

def updateNode(label:str, id:dict, property_key:str, new_value):
     """
    Update a property of a node in the graph database.

    This function executes a Cypher query to update a specified property of 
    a node identified by its label and properties. It uses the `getWhereClause` 
    function to construct the WHERE conditions for locating the node.

    Parameters:
    label (str): The label of the node to update.
    id (dict): A dictionary representing the identifier(s) for the node.
    property_key (str): The property key of the node to update.
    new_value: The new value to set for the specified property. This can be 
               of any type (string, number, etc.).

    Returns:
    The updated node if the operation is successful; otherwise, returns None.
    """
    status = session.run(f"""
                            MATCH (n:{label})
                            WHERE {getWhereClause("n",id)} 
                            SET n.{property_key} = {new_value} RETURN n
                        """)
    return status.single()

def createNode(label:str, properties:dict):
      """
    Create a new node in the graph database with specified properties.

    This function checks if a node with the specified label and properties 
    already exists. If it does, the function returns 0. If not, it constructs 
    a Cypher query to create a new node with the given properties.

    Parameters:
    label (str): The label for the new node.
    properties (dict): A dictionary of properties to set on the new node.

    Returns:
    int: Returns 0 if the node already exists; otherwise, returns the result 
         of the CREATE operation.
    """
    if checkIfNodeExist(label, properties):
        return 0
    
    properties_str = convertProperties_to_string(properties)
    return session.run(f"CREATE (:{label} {{{properties_str}}})")

def createRelationship(label1:str, label2:str, id1:dict, id2:dict, propertyName:str ,properties:dict):
      """
    Create a relationship between two nodes in a graph database.

    This function checks if a relationship already exists between two nodes 
    identified by their labels and IDs. If the relationship does not exist, 
    it creates a new relationship with specified properties.

    Parameters:
    label1 (str): The label of the first node.
    label2 (str): The label of the second node.
    id1 (dict): A dictionary representing the identifier(s) for the first node.
    id2 (dict): A dictionary representing the identifier(s) for the second node.
    propertyName (str): The name of the relationship type to create.
    properties (dict): A dictionary of properties to set on the relationship.

    Returns:
    int: Returns 0 if the relationship already exists; otherwise, returns 
         the status of the creation operation.
    """
    if checkIfRelationshipExist(label1, label2, id1, id1, propertyName):
        return 0
    
    properties_str = convertProperties_to_string(properties)
    status = session.run(f"""
                            MATCH (a:{label1}), (b:{label2})
                            WHERE {getWhereClause("a",id1)} AND {getWhereClause("b",id2)} 
                            CREATE (a)-[:{propertyName} {{{properties_str}}}]->(b)
                        """)
    return status
    
def printTest(label, properties):
    print("CREATE (:{label} {{{properties}}})".format(label=label, properties=convertProperties_to_string(properties)))
