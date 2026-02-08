#!/usr/bin/env python3
"""
iText2KG Web Server

A FastAPI-based web server that provides HTTP API endpoints for the iText2KG
knowledge graph construction library.

Endpoints:
- /health: Health check endpoint
- /api/kg/build: Build a knowledge graph from atomic facts
- /api/kg/entities: Get all entities from the knowledge graph
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import iText2KG modules
from itext2kg.logging_config import get_logger
from itext2kg.atom import Atom
from itext2kg.itext2kg_star import iText2KG_Star
from itext2kg.graph_integration import Neo4jStorage
from itext2kg.documents_distiller import DocumentsDistiller

logger = get_logger(__name__)

# FastAPI application
app = FastAPI(
    title="iText2KG API",
    description="API for building and managing Knowledge Graphs using LLMs",
    version="1.0.0"
)

# In-memory storage for knowledge graph (in production, use Neo4j or similar)
kg_storage: Optional[Any] = None
kg_lock = asyncio.Lock()

# Request/Response models
class BuildKGRequest(BaseModel):
    atomic_facts: List[str]
    obs_timestamp: str
    existing_knowledge_graph: Optional[bool] = None
    ent_threshold: float = 0.8
    rel_threshold: float = 0.7
    entity_name_weight: float = 0.8
    entity_label_weight: float = 0.2
    max_workers: int = 8

class BuildKGResponse(BaseModel):
    success: bool
    message: str
    entities_count: int
    relationships_count: int
    knowledge_graph_id: str

class EntityResponse(BaseModel):
    entity_id: str
    name: str
    label: str
    description: Optional[str] = None

class EntitiesResponse(BaseModel):
    success: bool
    entities: List[EntityResponse]
    count: int


def get_llm_model():
    """Get LLM model from environment or return a mock for testing."""
    # Try to get OpenAI API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openai_api_key:
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            llm_model = ChatOpenAI(
                api_key=openai_api_key,
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

            embeddings_model = OpenAIEmbeddings(
                api_key=openai_api_key,
                model=os.environ.get("EMBEDDINGS_MODEL", "text-embedding-3-small"),
            )

            return llm_model, embeddings_model
        except ImportError as e:
            logger.warning(f"Could not import langchain_openai: {e}")

    # Return None if no API key available
    return None, None


def get_neo4j_storage():
    """Get Neo4j storage from environment variables."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")

    if password:
        return Neo4jStorage(uri=uri, username=username, password=password)
    return None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "itext2kg",
        "version": "1.0.0"
    }


@app.post("/api/kg/build", response_model=BuildKGResponse)
async def build_knowledge_graph(request: BuildKGRequest):
    """
    Build a knowledge graph from atomic facts.

    This endpoint uses the ATOM or iText2KG_Star model to extract entities
    and relationships from the provided atomic facts and build a knowledge graph.
    """
    global kg_storage

    # Get LLM model
    llm_model, embeddings_model = get_llm_model()

    if llm_model is None:
        raise HTTPException(
            status_code=500,
            detail="LLM model not configured. Please set OPENAI_API_KEY environment variable."
        )

    try:
        # Initialize the ATOM pipeline
        atom = Atom(llm_model=llm_model, embeddings_model=embeddings_model)

        # Build the knowledge graph
        kg = await atom.build_graph(
            atomic_facts=request.atomic_facts,
            obs_timestamp=request.obs_timestamp,
            ent_threshold=request.ent_threshold,
            rel_threshold=request.rel_threshold,
            entity_name_weight=request.entity_name_weight,
            entity_label_weight=request.entity_label_weight,
            max_workers=request.max_workers
        )

        # Store the knowledge graph
        kg_storage = kg

        logger.info(f"Knowledge graph built successfully: {len(kg.entities)} entities, {len(kg.relationships)} relationships")

        return BuildKGResponse(
            success=True,
            message="Knowledge graph built successfully",
            entities_count=len(kg.entities),
            relationships_count=len(kg.relationships),
            knowledge_graph_id="kg_1"
        )

    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kg/entities", response_model=EntitiesResponse)
async def get_entities():
    """
    Get all entities from the knowledge graph.
    """
    global kg_storage

    if kg_storage is None:
        return EntitiesResponse(
            success=True,
            entities=[],
            count=0
        )

    try:
        entities = [
            EntityResponse(
                entity_id=str(e.entity_id) if hasattr(e, 'entity_id') else str(hash(e.name)),
                name=e.name,
                label=e.label if hasattr(e, 'label') else e.entity_type if hasattr(e, 'entity_type') else "Entity"
            )
            for e in kg_storage.entities
        ]

        return EntitiesResponse(
            success=True,
            entities=entities,
            count=len(entities)
        )

    except Exception as e:
        logger.error(f"Error getting entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kg/relationships")
async def get_relationships():
    """
    Get all relationships from the knowledge graph.
    """
    global kg_storage

    if kg_storage is None:
        return {"success": True, "relationships": [], "count": 0}

    try:
        relationships = [
            {
                "name": r.name,
                "start_entity": r.startEntity.name if hasattr(r, 'startEntity') else str(r.start_entity),
                "end_entity": r.endEntity.name if hasattr(r, 'endEntity') else str(r.end_entity),
                "properties": r.properties.model_dump() if hasattr(r, 'properties') else {}
            }
            for r in kg_storage.relationships
        ]

        return {
            "success": True,
            "relationships": relationships,
            "count": len(relationships)
        }

    except Exception as e:
        logger.error(f"Error getting relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kg/visualize")
async def visualize_knowledge_graph():
    """
    Visualize the knowledge graph in Neo4j (if configured).
    """
    global kg_storage

    if kg_storage is None:
        raise HTTPException(
            status_code=400,
            detail="No knowledge graph available. Please build one first."
        )

    neo4j_storage = get_neo4j_storage()

    if neo4j_storage is None:
        raise HTTPException(
            status_code=500,
            detail="Neo4j not configured. Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD environment variables."
        )

    try:
        neo4j_storage.visualize_graph(knowledge_graph=kg_storage)
        return {
            "success": True,
            "message": "Knowledge graph visualized in Neo4j",
            "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        }
    except Exception as e:
        logger.error(f"Error visualizing knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 11380))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting iText2KG server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
