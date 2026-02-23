from mnemos_client import MnemosMemory


def main() -> None:
    memory = MnemosMemory.from_env()
    try:
        memory.store("The customer prefers email updates.", importance=0.7)
        memory.store("The trial expires on March 15.", importance=0.9)

        answers = memory.ask("When does the trial expire?", top_k=3)
        print("Answers:")
        for i, text in enumerate(answers, start=1):
            print(f"{i}. {text}")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
