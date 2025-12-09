"""Neo4j database connection and utilities."""
from neo4j import GraphDatabase
import config


class Neo4jConnector:
    """Manages Neo4j database connections and queries."""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def execute_query(self, query: str, parameters: dict = None):
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters dictionary
            
        Returns:
            List of records
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def test_connection(self):
        """Test the Neo4j connection."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def create_vector_index(self, index_name: str, label: str, property_name: str, dimension: int):
        """
        Create a vector index for embedding-based search.
        
        Args:
            index_name: Name of the index
            label: Node label to index
            property_name: Property containing the vector
            dimension: Vector dimension
        """
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON n.{property_name}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimension},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        try:
            with self.driver.session() as session:
                session.run(query)
                print(f"Vector index '{index_name}' created successfully")
        except Exception as e:
            print(f"Error creating vector index: {e}")

