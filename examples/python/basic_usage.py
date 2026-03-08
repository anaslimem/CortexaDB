"""
CortexaDB Python Example - Basic Usage

Demonstrates core features:
- Opening a database
- Storing memories with embeddings (Unified .add)
- High-performance .ingest (batching)
- Fluent .query (QueryBuilder)
- Scoped .collection support
- Graph relationships
"""

from cortexadb import CortexaDB, HashEmbedder
import os
import shutil


def main():
    db_path = "example_agent.db"

    # Cleanup old db directory
    if os.path.isdir(db_path):
        shutil.rmtree(db_path)

    print("=== CortexaDB Python Example (v0.1.7) ===\n")

    # 1. Open database with embedder (auto-embeds text)
    # HashEmbedder generates deterministic embeddings for testing
    db = CortexaDB.open(db_path, embedder=HashEmbedder(dimension=128))
    print(f"Opened: {db}")

    # -----------------------------------------------------------
    # 2. Unified Add (stores a memory)
    # -----------------------------------------------------------
    print("\n[1] Adding information...")
    m1 = db.add(
        "The user lives in Paris and loves baguette.", metadata={"category": "personal"}
    )
    m2 = db.add("Paris is the capital of France.", metadata={"category": "fact"})
    m3 = db.add(
        "The weather in Paris is often rainy in autumn.",
        metadata={"category": "weather"},
    )
    print(f"   Stored 3 memories: IDs {m1}, {m2}, {m3}")

    # -----------------------------------------------------------
    # 3. High-Performance Ingest (100x faster batching)
    # -----------------------------------------------------------
    print("\n[2] Ingesting text with batching...")

    # Recursive (default) - splits paragraphs → sentences → words
    long_text = """
    First paragraph with some content.
    
    Second paragraph with more details.
    
    Third paragraph to complete the example.
    """
    # v0.1.7 uses optimized batch insertion internally
    ids = db.ingest(long_text, strategy="recursive", chunk_size=100, overlap=10)
    print(f"   Recursive batching: {len(ids)} chunks stored in ms")

    # -----------------------------------------------------------
    # 4. Fluent Query Builder
    # -----------------------------------------------------------
    print("\n[3] Using Fluent Query Builder...")
    
    results = db.query("Where does the user live?") \
        .limit(3) \
        .execute()
        
    print(f"   Query: 'Where does the user live?'")
    for res in results:
        print(f"   - ID: {res.id}, Score: {res.score:.4f}")

    # -----------------------------------------------------------
    # 5. Graph Relationships
    # -----------------------------------------------------------
    print("\n[4] Creating graph connections...")
    db.connect(m1, m2, "related_to")
    db.connect(m2, m3, "mentioned_in")
    print(f"   Connected memories: {m1} → {m2} → {m3}")

    # -----------------------------------------------------------
    # 6. Collections (Namespaced isolation)
    # -----------------------------------------------------------
    print("\n[5] Using Collections...")

    travel = db.collection("travel_agent")
    travel.add("Flight to Tokyo booked for June.")
    travel.add("Hotel reservation confirmed.")

    # Search scoped to the collection
    results = travel.search("Tokyo")
    print(f"   Travel collection: {len(results)} results for 'Tokyo'")

    # Or use QueryBuilder from a collection
    scoped_results = travel.query("Tokyo").limit(1).execute()
    print(f"   Scoped QueryBuilder: {len(scoped_results)} result")

    # -----------------------------------------------------------
    # 7. Stats
    # -----------------------------------------------------------
    print("\n[6] Database stats...")
    stats = db.stats()
    print(f"   Total entries: {stats.entries}")
    print(f"   Indexed embeddings: {stats.indexed_embeddings}")

    # Cleanup (database flushes on __exit__ or delete)
    del db
    if os.path.isdir(db_path):
        shutil.rmtree(db_path)

    print("\n=== Example Complete! ===")


if __name__ == "__main__":
    main()
