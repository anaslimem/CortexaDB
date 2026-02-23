from mnemos_client import MnemosClient


def main() -> None:
    with MnemosClient("127.0.0.1:50051", default_namespace="agent1") as client:
        memory_id = client.remember(
            "hello from python",
            importance=0.8,
            metadata={"source": "example"},
        )
        print("memory id:", memory_id)

        hits = client.recall("hello", top_k=5)
        print("hits:", len(hits))
        for hit in hits:
            print("text:", hit.text)
            print("namespace:", hit.memory.namespace if hit.memory else None)
            print("metadata:", hit.memory.metadata if hit.memory else {})
            print(
                "scores:",
                hit.final_score,
                hit.similarity_score,
                hit.importance_score,
                hit.recency_score,
            )


if __name__ == "__main__":
    main()
