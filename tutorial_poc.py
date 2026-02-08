#!/usr/bin/env python3
"""
Tutorial / PoC for iText2KG HTTP API

This script demonstrates how to interact with the iText2KG Docker container
via HTTP API.

Usage:
    python tutorial_poc.py
"""

import requests
import json

# Configuration
BASE_URL = "http://localhost:11380"

def test_health_check():
    """Test the health endpoint"""
    print("=" * 60)
    print("Testing Health Check")
    print("=" * 60)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to iText2KG server.")
        print("Make sure the Docker container is running:")
        print("  ./invoke.sh")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_build_knowledge_graph():
    """Test building a knowledge graph from text"""
    print("\n" + "=" * 60)
    print("Testing Knowledge Graph Construction")
    print("=" * 60)

    # Example: Build KG from sample atomic facts
    sample_facts = [
        "Steve Jobs was the CEO of Apple Inc. on January 9, 2007",
        "Steve Jobs passed away on October 5, 2011",
        "Tim Cook became the CEO of Apple Inc. on August 24, 2011"
    ]

    print("Submitting atomic facts for knowledge graph construction...")
    print(f"Sample facts: {json.dumps(sample_facts, indent=2)}")

    try:
        response = requests.post(
            f"{BASE_URL}/api/kg/build",
            json={
                "atomic_facts": sample_facts,
                "obs_timestamp": "2024-01-01"
            },
            timeout=60
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to iText2KG server.")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_get_entities():
    """Test getting entities from the knowledge graph"""
    print("\n" + "=" * 60)
    print("Testing Entity Retrieval")
    print("=" * 60)

    try:
        response = requests.get(
            f"{BASE_URL}/api/kg/entities",
            timeout=30
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to iText2KG server.")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main function to run all tests"""
    print("iText2KG Docker Container API Tutorial/PoC")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print("=" * 60)

    # Test health first
    if not test_health_check():
        print("\n" + "!" * 60)
        print("WARNING: Could not connect to iText2KG server.")
        print("!" * 60)
        print("\nTo start the iText2KG server:")
        print("1. Make sure Docker is running")
        print("2. Run: ./invoke.sh")
        print("3. Wait for the server to start")
        print("\nThen run this script again.")
        return 1

    # Run other tests
    test_build_knowledge_graph()
    test_get_entities()

    print("\n" + "=" * 60)
    print("Tutorial Complete")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    exit(main())
