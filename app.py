from flask import Flask, request, jsonify, send_from_directory
import os
from neo4j import GraphDatabase
import requests
import json
from typing import Dict, List, Any, Tuple, Optional

class Neo4jConnection:
    def __init__(self, uri, user, password, database=None):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.connect()
    
    def connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 AS result")
                record = result.single()
                if record and record["result"] == 1:
                    print("✅ Successfully connected to Neo4j database.")
                else:
                    print("❌ Connection test failed.")
        except Exception as e:
            print(f"❌ Connection error: {e}")
    
    def close(self):
        if self.driver:
            self.driver.close()
            print("Connection to Neo4j database closed.")
    
    def run_query(self, query, params=None):
        if not self.driver:
            print("No active connection. Please connect first.")
            return []
        
        params = params or {}
        
        try:
            with self.driver.session(database=self.database) as session:
                results = session.run(query, params)
                return [record.data() for record in results]
        except Exception as e:
            print(f"Query error: {e}")
            print(f"Query: {query}")
            return []


class EmbeddingGenerator:
    """Class to generate embeddings for text using the intfloat/e5-base-v2 model."""
    
    def __init__(self, model_name="intfloat/e5-base-v2"):
        """Initialize with the model name but DON'T load the model yet."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_loading = False
        print(f"✅ EmbeddingGenerator initialized (model will be loaded on first use)")
    
    def _load_model(self):
        """Load the embedding model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            # Model already loaded
            return True
            
        if self.is_loading:
            print("Model is already being loaded...")
            return False
            
        self.is_loading = True
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            print(f"Loading embedding model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("Using GPU for embeddings")
            else:
                print("Using CPU for embeddings")
                
            print("✅ Embedding model loaded successfully")
            self.is_loading = False
            return True
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            print("Make sure you have installed the required packages:")
            print("pip install transformers torch")
            self.is_loading = False
            return False
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding vector for the given text using e5-base-v2."""
        # Try to load the model if not already loaded
        if self.model is None or self.tokenizer is None:
            print("Model not loaded. Loading now...")
            success = self._load_model()
            if not success or self.model is None or self.tokenizer is None:
                print("Failed to load model.")
                return []
        
        try:
            import torch
            
            # E5 models expect a prefix for different tasks
            # For retrieval tasks, the prefix is "query: " or "passage: "
            # For semantic similarity, we use "query: "
            text = f"query: {text}"
            
            # Tokenize and prepare for the model
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Move to the same device as the model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # E5 uses mean pooling for sentence embeddings
                # Get attention mask to avoid padding tokens
                attention_mask = inputs["attention_mask"]
                
                # Mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to list and return
                return embeddings[0].cpu().tolist()
                
        except Exception as e:
            print(f"Error generating embedding: {e}")
            import traceback
            traceback.print_exc()
            return []


class LLMCypherGenerator:
    def __init__(self, api_key=None):
        """
        Initialize the LLM-based Cypher query generator.
        
        Args:
            api_key: API key for the LLM service (optional)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-3.5-turbo"
        
        # Define the schema explicitly based on your Neo4j database
        self.schema_description = """
Database Schema:
Nodes (30,065):
- Article
- Author
- Entity
- Person
- Topic

Relationships (49,209):
- HAS_TOPIC (connecting Articles to Topics)
- MENTIONS_ENTITY (connecting Articles to Entities)
- MENTIONS_PERSON (connecting Articles to Persons)
- WROTE (connecting Authors to Articles)

Property keys:
- content
- data
- embedding
- id
- meta
- name
- nodes
- relationships
- style
- visualisation

IMPORTANT NOTES:
1. The main property for node identification is 'name', except for author where it's 'id'
2. 'content' may contain article text
3. Do NOT use properties that aren't in the list above (e.g., don't use 'title', 'date', etc.)
4. The database has a vector index named 'article_content_index' on Article nodes
"""
    
    def generate_cypher(self, query_text: str, vector_search: bool = False, top_k: int = 5) -> str:
        """
        Generate a Cypher query from natural language.
        
        Args:
            query_text: Natural language query text
            vector_search: Whether to use vector search
            top_k: Number of top results to return for vector search
            
        Returns:
            Generated Cypher query
        """
        if vector_search:
            # Get keyword search terms for additional filtering
            keywords = self._extract_keywords(query_text)
            
            # For vector search, construct a Cypher query that:
            # 1. Uses the vector index on Article nodes
            # 2. Only applies embeddings to Article nodes
            # 3. Optionally filters based on entities/topics mentioned
            
            template = f"""
            // Vector similarity search on Article nodes using the article_content_index
            CALL db.index.vector.queryNodes('article_content_index', {top_k}, $embedding) 
            YIELD node, score
            """
            
            # Add filters based on query keywords
            if "topic" in query_text.lower() or any(kw in query_text.lower() for kw in ["about", "related to", "concerning", "regarding"]):
                template += """
                // Get topics associated with the articles
                WITH node, score
                MATCH (node)-[:HAS_TOPIC]->(t:Topic)
                RETURN node.content AS article, collect(t.name) AS topics, score
                ORDER BY score DESC
                """
            elif any(kw in query_text.lower() for kw in ["person", "people", "who", "mentions"]):
                template += """
                // Get people mentioned in the articles
                WITH node, score
                MATCH (node)-[:MENTIONS_PERSON]->(p:Person)
                RETURN node.content AS article, collect(p.name) AS mentioned_people, score
                ORDER BY score DESC
                """
            elif any(kw in query_text.lower() for kw in ["entity", "organization", "company"]):
                template += """
                // Get entities mentioned in the articles
                WITH node, score
                MATCH (node)-[:MENTIONS_ENTITY]->(e:Entity)
                RETURN node.content AS article, collect(e.name) AS mentioned_entities, score
                ORDER BY score DESC
                """
            else:
                # Default return format with additional information
                template += f"""
                // Default return with topics and entities
                WITH node, score
                OPTIONAL MATCH (node)-[:HAS_TOPIC]->(t:Topic)
                OPTIONAL MATCH (node)-[:MENTIONS_ENTITY]->(e:Entity)
                RETURN node.content AS article, 
                       collect(DISTINCT t.name) AS topics,
                       collect(DISTINCT e.name) AS entities,
                       score
                ORDER BY score DESC
                LIMIT {top_k}
                """
            
            return template
        else:
            # For regular queries, use the LLM
            prompt = self._create_prompt(query_text, vector_search, top_k)
            response = self._call_llm_api(prompt)
            cypher_query = self._extract_cypher_from_response(response)
            return cypher_query
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from the query text for filtering."""
        # Simple keyword extraction
        common_words = ["find", "show", "get", "what", "are", "the", "list", "articles", "about"]
        words = text.lower().split()
        keywords = [w for w in words if w not in common_words and len(w) > 3]
        return keywords[:3]  # Return top 3 keywords
    
    def _create_prompt(self, query_text: str, vector_search: bool, top_k: int) -> str:
        """
        Create a prompt for the LLM to generate a Cypher query.
        
        Args:
            query_text: Natural language query text
            vector_search: Whether to use vector search
            top_k: Number of top results to return for vector search
            
        Returns:
            Prompt for the LLM
        """
        base_prompt = f"""You are an expert in converting natural language queries into Neo4j Cypher queries.
Your task is to generate a valid Cypher query for Neo4j based on the user's question and the provided database schema.

{self.schema_description}

User's natural language query: "{query_text}"

IMPORTANT: Only use property names listed above. The primary property for all nodes is 'name'. Do NOT use properties like 'title', 'date_published', or anything not listed."""

        if vector_search:
            vector_instructions = f"""
This database has vector embeddings and supports vector search. The user wants to use vector similarity search with the following requirements:

1. Use the vector index 'article_content_index' for searching Articles
2. Restrict the vector search to only match articles that are related to the entities mentioned in the query
3. Return the top {top_k} most similar results

Use the vector index with syntax like:
CALL db.index.vector.queryNodes('article_content_index', {top_k}, $embedding) YIELD node, score
WHERE ... (additional filters to match entities)

The $embedding parameter will be supplied separately when executing the query.
"""
            return base_prompt + vector_instructions
        
        return base_prompt
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call the LLM API to generate a response.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            LLM response text
        """
        if not self.api_key:
            print("No API key provided. Using fallback rule-based approach.")
            return self._fallback_generate(prompt)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1  # Lower temperature for more deterministic output
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(data)
            )
            response.raise_for_status()
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                return response_data['choices'][0]['message']['content'].strip()
            else:
                print("Unexpected API response format:", response_data)
                return self._fallback_generate(prompt)
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return self._fallback_generate(prompt)
    
    def _fallback_generate(self, prompt: str) -> str:
        """Fallback method to generate simple Cypher queries."""
        # Extract key terms from the prompt
        query_text = prompt.split('User\'s natural language query: "')[1].split('"')[0]
        
        # Simple keyword-based matching
        if "article" in query_text.lower() and "author" in query_text.lower():
            return "MATCH (a:Author)-[:WROTE]->(art:Article) RETURN a.id, art.name LIMIT 10"
        elif "article" in query_text.lower():
            return "MATCH (art:Article) RETURN art.name LIMIT 10"
        elif "topic" in query_text.lower():
            return "MATCH (t:Topic) RETURN t.name LIMIT 10"
        else:
            # Very basic fallback
            return "MATCH (n) RETURN n.name LIMIT 10"
    
    def _extract_cypher_from_response(self, response: str) -> str:
        """Extract the Cypher query from the LLM response."""
        # Remove any markdown code block syntax
        query = response.replace('```cypher', '').replace('```', '').strip()
        
        # If there are explanations before or after the query, try to extract just the query
        if 'MATCH' in query or 'RETURN' in query:
            lines = query.split('\n')
            query_lines = []
            capturing = False
            
            for line in lines:
                # Start capturing at MATCH, RETURN, WITH, etc.
                if any(keyword in line.upper() for keyword in ['MATCH', 'RETURN', 'WITH', 'WHERE', 'CALL']):
                    capturing = True
                
                if capturing:
                    query_lines.append(line)
            
            if query_lines:
                query = '\n'.join(query_lines)
        
        return query


class NLQueryInterface:
    """Interface for natural language queries to Neo4j with vector search capabilities."""
    
    def __init__(self, uri, user, password, api_key=None, embedding_model="intfloat/e5-base-v2"):
        """Initialize with connection details."""
        self.neo4j = Neo4jConnection(uri, user, password)
        self.generator = LLMCypherGenerator(api_key)
        # Initialize but don't load the model
        self.embedding_generator = EmbeddingGenerator(embedding_model)
    
    def query(self, text, use_vector_search=False, top_k=5):
        """Convert natural language to Cypher and execute the query."""
        print(f"Processing query: {text}")
        
        # Generate embedding if vector search is requested
        embedding = None
        if use_vector_search:
            embedding = self.embedding_generator.generate_embedding(text)
            if not embedding:
                print("⚠️ Warning: Failed to generate embedding. Falling back to standard search.")
                use_vector_search = False
        
        # Generate Cypher query
        cypher = self.generator.generate_cypher(text, vector_search=use_vector_search, top_k=top_k)
        print(f"Generated Cypher query:\n{cypher}")
        
        # Execute query
        params = {}
        if use_vector_search and embedding:
            params["embedding"] = embedding
            params["top_k"] = top_k
            
            # Print embedding info for debugging
            print(f"Embedding vector generated successfully.")
            print(f"Vector dimensions: {len(embedding)}")
        
        results = self.neo4j.run_query(cypher, params)
        print(f"Found {len(results)} results")
        
        return results, cypher
    
    def close(self):
        """Close the Neo4j connection."""
        self.neo4j.close()


class VectorSearchInterface:
    """Specialized interface for vector similarity searches in Neo4j."""
    
    def __init__(self, uri, user, password, api_key=None, embedding_model="intfloat/e5-base-v2"):
        """Initialize with connection details."""
        self.neo4j = Neo4jConnection(uri, user, password)
        # Initialize but don't load the model
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.index_name = "article_content_index"  # The actual vector index name
    
    def search(self, query_text: str, entity_filter: Optional[str] = None, top_k: int = 5):
        """
        Perform a vector similarity search with optional entity filters.
        
        Args:
            query_text: The text to search for
            entity_filter: Optional entity name to filter results
            top_k: Number of top results to return
            
        Returns:
            Search results
        """
        # Generate embedding for the query
        embedding = self.embedding_generator.generate_embedding(query_text)
        if not embedding:
            print("Failed to generate embedding for query")
            return []
        
        print(f"Generated embedding vector with {len(embedding)} dimensions")
        
        # Build the Cypher query for vector search
        cypher_query = self._build_vector_search_query(entity_filter, top_k)
        
        # Execute the search
        params = {
            "embedding": embedding,
            "top_k": top_k
        }
        
        if entity_filter:
            params["entity_name"] = entity_filter
        
        results = self.neo4j.run_query(cypher_query, params)
        print(f"Vector search found {len(results)} results")
        
        return results
    
    def _build_vector_search_query(self, entity_filter: Optional[str], top_k: int) -> str:
        """Build a Cypher query for vector search with optional filtering."""
        # Base query - Article nodes only since only they have embeddings
        base_query = f"""
        CALL db.index.vector.queryNodes('{self.index_name}', $top_k, $embedding) 
        YIELD node, score
        """
        
        if entity_filter:
            # Filter by specific entity - assuming node is an Article
            filter_clause = """
            WITH node, score
            MATCH (node)-[:MENTIONS_ENTITY]->(e:Entity)
            WHERE e.name = $entity_name
            """
            return f"{base_query} {filter_clause} RETURN node.content AS article, score ORDER BY score DESC"
        else:
            # No entity filter - return more information about the articles
            return f"""
            {base_query}
            WITH node, score
            OPTIONAL MATCH (node)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (a:Author)-[:WROTE]->(node)
            RETURN node.content AS article, 
                   collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT a.id) AS authors,
                   score
            ORDER BY score DESC
            """
    
    def close(self):
        """Close the Neo4j connection."""
        self.neo4j.close()


# Initialize Flask app
app = Flask(__name__)

# Global variables to track initialization state
interface = None
initialization_status = {
    "db_initialized": False,
    "error": None
}

@app.before_first_request
def initialize_interface():
    global interface, initialization_status
    
    # If interface is already initialized, don't do it again
    if interface is not None:
        return
    
    try:
        # Get environment variables for connection
        neo4j_uri = os.environ.get("NEO4J_URI")
        neo4j_user = os.environ.get("NEO4J_USER")
        neo4j_password = os.environ.get("NEO4J_PASSWORD")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        # Check if essential environment variables are present
        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            error_msg = "Missing required environment variables for Neo4j connection"
            print(f"❌ Error: {error_msg}")
            initialization_status["error"] = error_msg
            return
        
        # Initialize interface
        interface = NLQueryInterface(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            api_key=openai_api_key,
            embedding_model="intfloat/e5-base-v2"
        )
        
        initialization_status["db_initialized"] = True
        print("✅ Query interface initialized successfully (without loading the embedding model)")
    except Exception as e:
        error_msg = f"Error initializing query interface: {e}"
        print(f"❌ {error_msg}")
        initialization_status["error"] = error_msg

@app.route('/', methods=['GET'])
def serve_ui():
    return send_from_directory('static', 'index.html')


@app.route('/query', methods=['GET', 'POST'])
def query_endpoint():
    """
    Query endpoint supporting both GET and POST methods.
    
    Query parameters:
    - query (required): Natural language query text
    - vector_search (optional): Use vector search if "true" (default: false)
    - top_k (optional): Number of top results to return (default: 5)
    - entity_filter (optional): Entity name to filter results by (for vector search)
    
    Returns:
        JSON response with results and query information
    """
    global interface, initialization_status
    
    # Initialize interface if not already done
    if interface is None:
        try:
            initialize_interface()
        except Exception as e:
            return jsonify({
                "error": f"Failed to initialize query interface: {str(e)}",
                "status": initialization_status
            }), 500
    
    # Get parameters (from either GET or POST)
    if request.method == 'POST':
        data = request.get_json() or {}
        query_text = data.get('query', '')
        vector_search = data.get('vector_search', False)
        top_k = int(data.get('top_k', 5))
        entity_filter = data.get('entity_filter')
    else:  # GET
        query_text = request.args.get('query', '')
        vector_search = request.args.get('vector_search', 'false').lower() == 'true'
        top_k = int(request.args.get('top_k', 5))
        entity_filter = request.args.get('entity_filter')
    
    # Validate input
    if not query_text:
        return jsonify({
            "error": "Missing required parameter: query"
        }), 400
    
    # Process the query based on the parameters
    try:
        if vector_search and entity_filter:
            # Use VectorSearchInterface for entity-filtered vector search
            vector_interface = VectorSearchInterface(
                uri=os.environ.get("NEO4J_URI"),
                user=os.environ.get("NEO4J_USER"),
                password=os.environ.get("NEO4J_PASSWORD"),
                api_key=os.environ.get("OPENAI_API_KEY")
            )
            results = vector_interface.search(query_text, entity_filter, top_k)
            cypher = vector_interface._build_vector_search_query(entity_filter, top_k)
            vector_interface.close()
            
            return jsonify({
                "results": results,
                "query": query_text,
                "cypher": cypher,
                "count": len(results),
                "vector_search": True,
                "entity_filter": entity_filter
            })
        else:
            # Use the standard NLQueryInterface
            results, cypher = interface.query(query_text, use_vector_search=vector_search, top_k=top_k)
            
            return jsonify({
                "results": results,
                "query": query_text,
                "cypher": cypher,
                "count": len(results),
                "vector_search": vector_search
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Query execution failed: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with initialization status."""
    global interface, initialization_status
    
    # Initialize interface if not already done
    if interface is None:
        try:
            initialize_interface()
        except Exception as e:
            pass  # We'll report the error in the status
    
    return jsonify({
        "status": "healthy" if initialization_status["db_initialized"] else "initializing",
        "message": "Neo4j Query API is running",
        "details": initialization_status
    })

@app.route('/api', methods=['GET'])
def api_info():
    """API info endpoint showing app is running."""
    return jsonify({
        "status": "healthy",
        "message": "n4jquery API is running",
        "version": "1.0",
        "endpoints": {
            "/query": "Main query endpoint (GET/POST)",
            "/health": "Health check endpoint"
        }
    })
    

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # For local development
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
