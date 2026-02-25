from cortexadb import CortexaDB
import os

def main():
    # Database path
    db_path = "example_agent.mem"
    
    # Cleanup old db if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print(f"--- Initializing CortexaDB at {db_path} ---")
    
    # Open database with 1536 dimensions (Standard for OpenAI embeddings)
    db = CortexaDB.open(db_path, dimension=1536)
    
    # 1. Store Memories (Semantic)
    print("\n[Step 1] Remembering information...")
    m1 = db.remember("The user lives in Paris and loves baguette.", metadata={"category": "personal"})
    m2 = db.remember("Paris is the capital of France.", metadata={"category": "fact"})
    m3 = db.remember("The weather in Paris is often rainy in autumn.", metadata={"category": "weather"})
    
    # 2. Hybrid Search (Semantic + Temporal)
    print("\n[Step 2] Asking questions (Semantic Search)...")
    results = db.ask("Where does the user live?")
    for res in results:
        # Note: In a real scenario, you'd use the stored text.
        # Here we just show the metadata and score.
        print(f" >> Found memory ID: {res.id}, Score: {res.score:.4f}")
        
    # 3. Graph Relationships
    print("\n[Step 3] Connecting memories (Graph Relations)...")
    # Connect user location to the fact about Paris
    db.connect(m1, m2, "related_to")
    
    # 4. Namespaces (Isolation)
    print("\n[Step 4] Using namespaces...")
    travel_db = db.namespace("travel_agent")
    travel_db.remember("Flight ticket to Tokyo is booked for June.")
    
    print("\n--- Example Complete! ---")
    
    # Get Stats
    stats = db.stats()
    print(f"Total entries: {stats.entries}")
    print(f"Storage size: {stats.bytes_on_disk} bytes")

if __name__ == "__main__":
    main()
