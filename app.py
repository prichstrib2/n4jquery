from flask import Flask, request, jsonify, send_from_directory
import os
import time
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
        """Run a Cypher query with improved error handling and logging."""
        if not self.driver:
            print("No active connection. Attempting to reconnect...")
            self.connect()
            if not self.driver:
                print("Reconnection failed. Cannot run query.")
                return []
        
        params = params or {}
        
        # Debug information
        param_keys = list(params.keys())
        print(f"Running query with parameters: {param_keys}")
        
        try:
            with self.driver.session(database=self.database) as session:
                # Add timeout option for long-running queries
                results = session.run(query, params)
                # Collect results with timeout protection
                try:
                    records = [record.data() for record in results]
                    print(f"Query returned {len(records)} records")
                    return records
                except Exception as data_exception:
                    print(f"Error collecting query results: {data_exception}")
                    # Return empty list on data collection error
                    return []
        except Exception as e:
            print(f"❌ Query execution error: {e}")
            print(f"Query: {query}")
            
            # Check for specific Neo4j error types
            error_str = str(e).lower()
            if "syntax" in error_str:
                print("Syntax error detected in Cypher query")
            elif "timeout" in error_str:
                print("Query timeout detected")
            elif "constraint" in error_str:
                print("Constraint violation detected")
            elif "connection" in error_str:
                print("Connection issue detected - attempting to reconnect")
                self.connect()
            
            # Return empty result
            return []
    
    def test_query(self, query, params=None):
        """Test if a query is valid without fully executing it."""
        if not self.driver:
            return False
        
        params = params or {}
        
        try:
            with self.driver.session(database=self.database) as session:
                # Using EXPLAIN to check query validity without execution
                explain_query = f"EXPLAIN {query}"
                session.run(explain_query, params)
                return True
        except Exception as e:
            print(f"Query validation error: {e}")
            return False


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
- Article (primary identifier: 'name', contains 'content' property with article text)
- Author (primary identifier: 'id')
- Entity (primary identifier: 'name')
- Person (primary identifier: 'name')
- Topic (primary identifier: 'name')

Relationships (49,209):
- HAS_TOPIC (connecting Articles to Topics)
- MENTIONS_ENTITY (connecting Articles to Entities)
- MENTIONS_PERSON (connecting Articles to Persons)
- WROTE (connecting Authors to Articles)

Property keys:
- content - contains text content for articles
- data
- embedding - vector embeddings for semantic search
- id - primary identifier for Author nodes
- meta
- name - primary identifier for Article, Entity, Person, and Topic nodes
- nodes
- relationships
- style
- visualisation

IMPORTANT NOTES:
1. The main property for node identification is 'name', except for Author where it's 'id'
2. 'content' property contains article text
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
            # Handle vector search query generation
            template = self._generate_vector_search_template(query_text, top_k)
            return template
        else:
            # Try LLM-based generation with fallback to rules
            if not self.api_key:
                print("No API key provided. Using rule-based approach.")
                return self._rule_based_generate(query_text, top_k)
            
            try:
                # Try LLM generation
                prompt = self._create_prompt(query_text, vector_search, top_k)
                response = self._call_llm_api(prompt)
                cypher_query = self._extract_cypher_from_response(response)
                
                # Validate the generated query
                if self._validate_cypher_query(cypher_query):
                    print("Using LLM-generated Cypher query")
                    return cypher_query
                else:
                    print("LLM-generated query validation failed. Using rule-based approach.")
                    return self._rule_based_generate(query_text, top_k)
            except Exception as e:
                print(f"Error in LLM query generation: {e}")
                return self._rule_based_generate(query_text, top_k)
    
    def _generate_vector_search_template(self, query_text: str, top_k: int) -> str:
        """Generate a template for vector similarity search."""
        # Base vector search query
        template = f"""
        // Vector similarity search on Article nodes using the article_content_index
        CALL db.index.vector.queryNodes('article_content_index', {top_k}, $embedding) 
        YIELD node, score
        """
        
        # Check query intent to customize return clause
        lower_query = query_text.lower()
        
        if any(kw in lower_query for kw in ["topic", "about", "related to", "concerning", "regarding"]):
            template += """
            // Get topics associated with the articles
            WITH node, score
            MATCH (node)-[:HAS_TOPIC]->(t:Topic)
            RETURN node.content AS article, collect(DISTINCT t.name) AS topics, score
            ORDER BY score DESC
            """
        elif any(kw in lower_query for kw in ["author", "written by", "wrote"]):
            template += """
            // Get authors of the articles
            WITH node, score
            MATCH (a:Author)-[:WROTE]->(node)
            RETURN node.content AS article, collect(DISTINCT a.id) AS authors, score
            ORDER BY score DESC
            """
        elif any(kw in lower_query for kw in ["person", "people", "who", "mentions"]):
            template += """
            // Get people mentioned in the articles
            WITH node, score
            MATCH (node)-[:MENTIONS_PERSON]->(p:Person)
            RETURN node.content AS article, collect(DISTINCT p.name) AS mentioned_people, score
            ORDER BY score DESC
            """
        elif any(kw in lower_query for kw in ["entity", "organization", "company"]):
            template += """
            // Get entities mentioned in the articles
            WITH node, score
            MATCH (node)-[:MENTIONS_ENTITY]->(e:Entity)
            RETURN node.content AS article, collect(DISTINCT e.name) AS mentioned_entities, score
            ORDER BY score DESC
            """
        else:
            # Default comprehensive return
            template += f"""
            // Default return with topics, authors, and entities
            WITH node, score
            OPTIONAL MATCH (node)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (a:Author)-[:WROTE]->(node)
            OPTIONAL MATCH (node)-[:MENTIONS_ENTITY]->(e:Entity)
            RETURN node.content AS article, 
                   collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT a.id) AS authors,
                   collect(DISTINCT e.name) AS entities,
                   score
            ORDER BY score DESC
            LIMIT {top_k}
            """
        
        return template
    
    def _rule_based_generate(self, query_text: str, top_k: int) -> str:
        """Generate Cypher query using rule-based approach for standard search."""
        print("Using rule-based Cypher generation")
        query_lower = query_text.lower()
        
        # Extract key terms for content filtering
        key_terms = self._extract_search_terms(query_text)
        content_filter = self._build_content_filter(key_terms)
        
        # Check for name pattern in query (FirstName LastName) - likely a Person query
        name_pattern = self._extract_name_pattern(query_text)
        if name_pattern and "author" not in query_lower:
            return f"""
            MATCH (art:Article)
            {content_filter}
            OPTIONAL MATCH (art)-[:MENTIONS_PERSON]->(p:Person)
            WHERE toLower(p.name) CONTAINS toLower('{name_pattern}')
            OPTIONAL MATCH (art)-[:HAS_TOPIC]->(t:Topic)
            RETURN art.content AS article, 
                   collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT p.name) AS people
            LIMIT {top_k}
            """
        
        # Check query intent based on keywords
        if "author" in query_lower and "article" in query_lower:
            # Articles by author - note that authors use ID, not name
            return f"""
            MATCH (a:Author)-[:WROTE]->(art:Article)
            {content_filter}
            RETURN art.content AS article, a.id AS author
            LIMIT {top_k}
            """
        elif "topic" in query_lower and "article" in query_lower:
            # Articles by topic
            return f"""
            MATCH (art:Article)-[:HAS_TOPIC]->(t:Topic)
            {content_filter}
            RETURN art.content AS article, collect(DISTINCT t.name) AS topics
            LIMIT {top_k}
            """
        elif "entity" in query_lower and "article" in query_lower:
            # Articles mentioning entities
            return f"""
            MATCH (art:Article)-[:MENTIONS_ENTITY]->(e:Entity)
            {content_filter}
            RETURN art.content AS article, collect(DISTINCT e.name) AS entities
            LIMIT {top_k}
            """
        elif "person" in query_lower and "article" in query_lower:
            # Articles mentioning people (persons are separate from authors)
            return f"""
            MATCH (art:Article)-[:MENTIONS_PERSON]->(p:Person)
            {content_filter}
            RETURN art.content AS article, collect(DISTINCT p.name) AS people
            LIMIT {top_k}
            """
        elif "article" in query_lower:
            # Just return articles with associated data
            return f"""
            MATCH (art:Article)
            {content_filter}
            OPTIONAL MATCH (art)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (a:Author)-[:WROTE]->(art)
            OPTIONAL MATCH (art)-[:MENTIONS_PERSON]->(p:Person)
            RETURN art.content AS article, 
                   collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT a.id) AS authors,
                   collect(DISTINCT p.name) AS people
            LIMIT {top_k}
            """
        elif "topic" in query_lower:
            # Just return topics
            return f"""
            MATCH (t:Topic)
            WHERE toLower(t.name) CONTAINS toLower('{query_text}') 
            RETURN t.name AS topic
            LIMIT {top_k}
            """
        elif "author" in query_lower:
            # Just return authors (using ID)
            return f"""
            MATCH (a:Author)
            RETURN a.id AS author
            LIMIT {top_k}
            """
        elif any(name in query_lower for name in ["person", "people"]):
            # Just return people
            return f"""
            MATCH (p:Person)
            WHERE toLower(p.name) CONTAINS toLower('{query_text}')
            RETURN p.name AS person
            LIMIT {top_k}
            """
        else:
            # Fallback to a comprehensive query
            return f"""
            MATCH (art:Article)
            {content_filter}
            OPTIONAL MATCH (art)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (a:Author)-[:WROTE]->(art)
            OPTIONAL MATCH (art)-[:MENTIONS_PERSON]->(p:Person)
            RETURN art.content AS article, 
                   collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT a.id) AS authors, 
                   collect(DISTINCT p.name) AS people
            LIMIT {top_k}
            """

    def _extract_name_pattern(self, query_text: str) -> str:
        """
        Extract potential person name from query (e.g., "John Smith", "Senator Jones")
        Returns the potential name or empty string if none found
        """
        # Look for capitalized words that might be names
        words = query_text.split()
        potential_name = []
        
        # Common titles that might precede a name
        titles = ["senator", "governor", "mayor", "president", "representative", "rep", "sen", "gov", "congressman", "congresswoman"]
        
        for i, word in enumerate(words):
            # Check for title followed by capitalized word
            if i < len(words) - 1 and word.lower() in titles and words[i+1][0].isupper():
                return " ".join(words[i:i+2])
                
            # Check for consecutive capitalized words (potential first/last name)
            if word[0].isupper() and i < len(words) - 1 and words[i+1][0].isupper():
                return " ".join(words[i:i+2])
        
        return ""

    def _extract_search_terms(self, query_text: str) -> list:
        """Extract key search terms from the query text."""
        # Remove common stop words
        stop_words = ["the", "and", "or", "a", "an", "in", "on", "at", "to", "for", "with", "by", "about", "from"]
        words = query_text.lower().split()
        terms = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Limit to most significant terms (max 5)
        return terms[:5]
    
    def _build_content_filter(self, terms: list) -> str:
        """Build a Cypher WHERE clause to filter article content by terms."""
        if not terms:
            return ""
            
        # Build OR conditions for each term
        conditions = [f"toLower(art.content) CONTAINS toLower('{term}')" for term in terms]
        where_clause = "WHERE " + " OR ".join(conditions)
        
        return where_clause
    
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

IMPORTANT RULES:
1. Only use property names listed above
2. The primary property for all nodes is 'name', except for Author where it's 'id'
3. Do NOT use properties like 'title', 'date_published', or anything not listed
4. Always include a LIMIT clause, limiting to {top_k} results
5. Only return properties that exist in the database schema
6. Provide a clean, executable Cypher query with no explanations or comments
7. If you're not sure about a property, use the safer OPTIONAL MATCH
8. If finding articles, include their content using node.content AS article"""

        if vector_search:
            vector_instructions = f"""
This database has vector embeddings. Use the vector index 'article_content_index' with syntax:
CALL db.index.vector.queryNodes('article_content_index', {top_k}, $embedding) YIELD node, score

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
                raise ValueError("Invalid API response")
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            raise
    
    def _extract_cypher_from_response(self, response: str) -> str:
        """Extract the Cypher query from the LLM response."""
        # Remove any markdown code block syntax
        query = response.replace('```cypher', '').replace('```', '').strip()
        
        # If there are explanations before or after the query, try to extract just the query
        if 'MATCH' in query or 'RETURN' in query or 'CALL' in query:
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
    
    def _validate_cypher_query(self, query: str) -> bool:
        """Validate the generated Cypher query for common errors."""
        # Check if query is empty
        if not query or query.strip() == "":
            return False
        
        # Check if query contains basic Cypher components
        if not any(keyword in query.upper() for keyword in ['MATCH', 'CALL', 'RETURN']):
            return False
        
        # Check for properties that aren't in our schema
        invalid_props = ['title', 'date', 'published', 'text', 'body', 'description']
        for prop in invalid_props:
            if f".{prop}" in query.lower():
                return False
        
        # Check for LIMIT clause
        if 'LIMIT' not in query.upper() and not 'TOP' in query.upper():
            return False
        
        return True


class NLQueryInterface:
    """Interface for natural language queries to Neo4j with improved error handling and logging."""
    
    def __init__(self, uri, user, password, api_key=None, embedding_model="intfloat/e5-base-v2"):
        """Initialize with connection details."""
        self.neo4j = Neo4jConnection(uri, user, password)
        self.generator = LLMCypherGenerator(api_key)
        # Initialize but don't load the model
        self.embedding_generator = EmbeddingGenerator(embedding_model)
    
    def query(self, text, use_vector_search=False, top_k=5):
        """Convert natural language to Cypher and execute the query with robust error handling."""
        print(f"Processing query: {text}")
        
        # Generate embedding if vector search is requested
        embedding = None
        if use_vector_search:
            try:
                embedding = self.embedding_generator.generate_embedding(text)
                if not embedding:
                    print("⚠️ Warning: Failed to generate embedding. Falling back to standard search.")
                    use_vector_search = False
            except Exception as e:
                print(f"⚠️ Error generating embedding: {e}")
                print("Falling back to standard search.")
                use_vector_search = False
        
        # Generate Cypher query with multiple attempts and fallbacks
        try:
            cypher = self.generator.generate_cypher(text, vector_search=use_vector_search, top_k=top_k)
            print(f"Generated Cypher query:\n{cypher}")
            
            # Validate query has minimum components needed
            if not self._validate_query_basics(cypher):
                print("⚠️ Warning: Generated query appears invalid, using fallback.")
                cypher = self._generate_fallback_query(text, top_k)
                print(f"Using fallback query:\n{cypher}")
        except Exception as e:
            print(f"⚠️ Error generating Cypher: {e}")
            cypher = self._generate_fallback_query(text, top_k)
            print(f"Using fallback query:\n{cypher}")
        
        # Execute query with error handling
        params = {}
        if use_vector_search and embedding:
            params["embedding"] = embedding
            params["top_k"] = top_k
            print(f"Embedding vector generated successfully. Dimensions: {len(embedding)}")
        
        try:
            results = self.neo4j.run_query(cypher, params)
            print(f"Query executed successfully. Found {len(results)} results")
            return results, cypher
        except Exception as e:
            print(f"⚠️ Query execution error: {e}")
            # Try again with simpler fallback query
            try:
                simple_query = self._generate_simple_fallback()
                print(f"Trying simple fallback query:\n{simple_query}")
                results = self.neo4j.run_query(simple_query, {})
                return results, simple_query
            except Exception as e2:
                print(f"⚠️ Fallback query also failed: {e2}")
                # Return empty results and the original cypher
                return [], cypher
    
    def _validate_query_basics(self, query):
        """Check if a query has the basic required components."""
        if not query or len(query.strip()) < 10:
            return False
        
        # Check for essential Cypher keywords
        if not ('MATCH' in query.upper() or 'CALL' in query.upper()):
            return False
            
        if 'RETURN' not in query.upper():
            return False
            
        return True
    
    def _generate_fallback_query(self, text, top_k):
        """Generate a fallback query based on the user's natural language query."""
        text_lower = text.lower()
        
        # Extract key terms for content filtering
        terms = []
        for word in text.split():
            word = word.lower()
            if len(word) > 3 and word not in ["the", "and", "with", "about", "for"]:
                terms.append(word)
        
        # Build content filter
        content_filter = ""
        if terms:
            conditions = [f"toLower(art.content) CONTAINS toLower('{term}')" for term in terms[:5]]
            content_filter = "WHERE " + " OR ".join(conditions)
        
        # Detect query intent from keywords
        if any(word in text_lower for word in ["author", "wrote", "written"]):
            return f"""
            MATCH (a:Author)-[:WROTE]->(art:Article)
            {content_filter}
            RETURN art.content AS article, a.id AS author
            LIMIT {top_k}
            """
        elif any(word in text_lower for word in ["topic", "about", "subject"]):
            return f"""
            MATCH (art:Article)-[:HAS_TOPIC]->(t:Topic)
            {content_filter}
            RETURN art.content AS article, collect(DISTINCT t.name) AS topics
            LIMIT {top_k}
            """
        elif any(word in text_lower for word in ["entity", "organization", "company"]):
            return f"""
            MATCH (art:Article)-[:MENTIONS_ENTITY]->(e:Entity)
            {content_filter}
            RETURN art.content AS article, collect(DISTINCT e.name) AS entities
            LIMIT {top_k}
            """
        elif any(word in text_lower for word in ["person", "people", "who"]):
            return f"""
            MATCH (art:Article)-[:MENTIONS_PERSON]->(p:Person)
            {content_filter}
            RETURN art.content AS article, collect(DISTINCT p.name) AS people
            LIMIT {top_k}
            """
        else:
            # Default general query
            return f"""
            MATCH (art:Article)
            {content_filter}
            OPTIONAL MATCH (art)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (a:Author)-[:WROTE]->(art)
            RETURN art.content AS article, collect(DISTINCT t.name) AS topics, collect(DISTINCT a.id) AS authors
            LIMIT {top_k}
            """
    
    def _generate_simple_fallback(self):
        """Generate the simplest possible query as last resort."""
        return """
        MATCH (art:Article)
        RETURN art.content AS article
        LIMIT 5
        """
    
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
    Enhanced query endpoint with better error handling and logging.
    
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
    
    # Check if database is initialized
    if not initialization_status["db_initialized"]:
        return jsonify({
            "error": "Database connection not initialized",
            "status": initialization_status
        }), 503  # Service Unavailable
    
    # Get parameters (from either GET or POST)
    try:
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
        
        # Validate and sanitize input
        if not query_text:
            return jsonify({
                "error": "Missing required parameter: query"
            }), 400
        
        # Limit top_k to reasonable values
        top_k = max(1, min(top_k, 100))  # Ensure between 1 and 100
        
        # Log the request
        print(f"Query request: '{query_text}', vector_search={vector_search}, top_k={top_k}, entity_filter={entity_filter}")
    except Exception as e:
        return jsonify({
            "error": f"Invalid request parameters: {str(e)}"
        }), 400
    
    # Process the query based on the parameters
    try:
        # Import time module for tracking execution time
        import time
        
        if vector_search and entity_filter:
            # Use VectorSearchInterface for entity-filtered vector search
            vector_interface = VectorSearchInterface(
                uri=os.environ.get("NEO4J_URI"),
                user=os.environ.get("NEO4J_USER"),
                password=os.environ.get("NEO4J_PASSWORD"),
                api_key=os.environ.get("OPENAI_API_KEY")
            )
            
            start_time = time.time()
            results = vector_interface.search(query_text, entity_filter, top_k)
            cypher = vector_interface._build_vector_search_query(entity_filter, top_k)
            vector_interface.close()
            query_time = time.time() - start_time
            
            return jsonify({
                "results": results,
                "query": query_text,
                "cypher": cypher,
                "count": len(results),
                "vector_search": True,
                "entity_filter": entity_filter,
                "execution_time_ms": round(query_time * 1000, 2)
            })
        else:
            # Use the standard NLQueryInterface with timeout protection
            start_time = time.time()
            
            # Set a timeout for the whole operation
            results, cypher = interface.query(query_text, use_vector_search=vector_search, top_k=top_k)
            query_time = time.time() - start_time
            
            # Clean up results to ensure they're serializable
            sanitized_results = []
            for result in results:
                clean_result = {}
                for k, v in result.items():
                    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                        clean_result[k] = v
                    else:
                        # Convert non-serializable objects to strings
                        clean_result[k] = str(v)
                sanitized_results.append(clean_result)
            
            return jsonify({
                "results": sanitized_results,
                "query": query_text,
                "cypher": cypher,
                "count": len(results),
                "vector_search": vector_search,
                "execution_time_ms": round(query_time * 1000, 2)
            })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Query execution failed: {str(e)}",
            "query": query_text,
            "vector_search": vector_search
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
