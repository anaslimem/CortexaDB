from mnemos_client import MnemosClient


def main() -> None:
    with MnemosClient("127.0.0.1:50051") as client:
        command_id = client.insert_text(
            namespace="agent1",
            text="hello from python",
            importance=0.8,
            metadata={"source": "example"},
        )
        print("insert command id:", command_id)

        result = client.query_text("hello", top_k=5, namespace="agent1")
        print("hits:", len(result.hits))
        for hit in result.hits:
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
