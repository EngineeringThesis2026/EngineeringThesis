import vector_database


def main():
    print("Testing Qdrant connection")
    client = vector_database._vector_database_client
    print(client.get_collections())


if __name__ == "__main__":
    main()