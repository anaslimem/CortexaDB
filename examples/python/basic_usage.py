"""
CortexaDB Python Example - Basic Usage

Demonstrates core features:
- Opening a database
- Storing memories with embeddings
- Smart chunking strategies
- Loading files
- Hybrid search
- Graph relationships
- Namespaces
"""

from cortexadb import CortexaDB, HashEmbedder
import os


def main():
    db_path = "example_agent.mem"

    # Cleanup old db
    for f in [db_path, f"{db_path}.wal", f"{db_path}.checkpoint"]:
        if os.path.exists(f):
            os.remove(f)

    print("=== CortexaDB Python Example ===\n")

    # Open database with embedder (auto-embeds text)
    # HashEmbedder generates deterministic embeddings from text
    db = CortexaDB.open(db_path, embedder=HashEmbedder(dimension=128))
    print(f"Opened: {db}")

    # -----------------------------------------------------------
    # 1. Simple remember (stores a memory)
    # -----------------------------------------------------------
    print("\n[1] Remembering information...")
    m1 = db.remember(
        "The user lives in Paris and loves baguette.", metadata={"category": "personal"}
    )
    m2 = db.remember("Paris is the capital of France.", metadata={"category": "fact"})
    m3 = db.remember(
        "The weather in Paris is often rainy in autumn.",
        metadata={"category": "weather"},
    )
    print(f"   Stored 3 memories: IDs {m1}, {m2}, {m3}")

    # -----------------------------------------------------------
    # 2. Ingest text with smart chunking
    # -----------------------------------------------------------
    print("\n[2] Ingesting text with chunking...")

    # Recursive (default) - splits paragraphs → sentences → words
    long_text = """
    First paragraph with some content.
    
    Second paragraph with more details.
    
    Third paragraph to complete the example.
    """
    ids = db.ingest(long_text, strategy="recursive", chunk_size=100, overlap=10)
    print(f"   Recursive chunking: {len(ids)} chunks stored")

    # Semantic - split by paragraphs
    ids = db.ingest(long_text, strategy="semantic")
    print(f"   Semantic chunking: {len(ids)} chunks stored")

    # -----------------------------------------------------------
    # 3. Load files (TXT, MD, JSON supported natively)
    # -----------------------------------------------------------
    print("\n[3] Loading files...")

    # Create a test markdown file
    test_file = "example_doc.md"
    with open(test_file, "w") as f:
        f.write("""# Example Document

## Introduction

This is an introduction paragraph with some content.

## Features

- Feature one
- Feature two
- Feature three

## Conclusion

This is the conclusion.
""")

    # Load with markdown strategy (preserves headers)
    ids = db.load(test_file, strategy="markdown")
    print(f"   Loaded markdown: {len(ids)} chunks stored")
    os.remove(test_file)

    # -----------------------------------------------------------
    # 4. Semantic Search
    # -----------------------------------------------------------
    print("\n[4] Searching memories...")
    results = db.ask("Where does the user live?")
    print(f"   Query: 'Where does the user live?'")
    for res in results:
        print(f"   - ID: {res.id}, Score: {res.score:.4f}")

    # -----------------------------------------------------------
    # 5. Graph Relationships
    # -----------------------------------------------------------
    print("\n[5] Creating graph connections...")
    db.connect(m1, m2, "related_to")
    db.connect(m2, m3, "mentioned_in")
    print(f"   Connected memories: {m1} → {m2} → {m3}")

    # -----------------------------------------------------------
    # 6. Namespaces (Multi-agent isolation)
    # -----------------------------------------------------------
    print("\n[6] Using namespaces...")

    travel_db = db.namespace("travel_agent")
    travel_db.remember("Flight to Tokyo booked for June.")
    travel_db.remember("Hotel reservation confirmed.")

    results = travel_db.ask("Tokyo")
    print(f"   Travel namespace: {len(results)} results for 'Tokyo'")

    # -----------------------------------------------------------
    # 7. Stats
    # -----------------------------------------------------------
    print("\n[7] Database stats...")
    stats = db.stats()
    print(f"   Total entries: {stats.entries}")
    print(f"   Indexed embeddings: {stats.indexed_embeddings}")

    # Close database first (releases file locks)
    del db

    # Cleanup
    for f in [db_path, f"{db_path}.wal", f"{db_path}.checkpoint"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except PermissionError:
                pass  # File may be locked

    print("\n=== Example Complete! ===")


if __name__ == "__main__":
    main()
