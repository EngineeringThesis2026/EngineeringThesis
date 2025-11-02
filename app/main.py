import process_data
import vector_database
import llm_model
from langchain_core.output_parsers import StrOutputParser

def main():

    load_data_needs_to_be_done = False  # Set to True if data loading is required

    if load_data_needs_to_be_done:
        # LOADING DATA
        data_folder_path = process_data.get_data_folder_path()
        all_data_from_pdfs = process_data.load_data_from_pdf(file_path=data_folder_path)

        # SPLITTING DATA
        all_splits = process_data.split_docks_into_chunks(documents=all_data_from_pdfs)

        # CREATING EMBEDDINGS
        embeddings = process_data.create_embeddings_with_metadata(sentences=all_splits,embedding_model=process_data._embedding_model)

        # VECTOR DATABASE OPERATIONS
        vector_size = len(embeddings[0]['embedding'])

        vector_database.create_collection_if_not_exists(vector_database_client=vector_database._vector_database_client,
                                                        collection_name=vector_database._collection_name, vector_size=vector_size)
    
        points = vector_database.create_points_from_embeddings(embeddings=embeddings)
        vector_database.upload_to_qdrant(vector_database_client=vector_database._vector_database_client,
                                        collection_name=vector_database._collection_name, points=points)

    # LLM OPERATIONS
    # Creating LLM instance
    llm = llm_model.create_llm()

    # Creating Chat Prompt Template
    prompt_template = llm_model.create_chat_prompt_template(
        system_template="Jesteś pomocnym i profesjonalnym asystentem AI specjalizującym się w doradztwie prawnym. Odpowiadaj wyłącznie na pytania związane z prawem, dostarczając dokładne i zwięzłe informacje.",
        human_template="Pytanie użytkownika: {question}\n\nKontekst:\n{context}"
    )
    print(prompt_template.format_messages(question="Tutaj trafia przykładowe pytanie użytkownika.", context="Tutaj trafia kontekst pytania."))

    retriever = vector_database.create_retriever()

    # RAG chain
    def combine_docs(docs):
        """Combine documents' page_content into a single string"""
        return "\n\n".join([f"{d.metadata}\n{d.page_content}" for d in docs])

    def rag_chain_fn(question: str):
        """RAG: retrieves documents, combines context, and invokes LLM"""
        docs = retriever.invoke(question)  # retrieve relevant documents
        context = combine_docs(docs)       # combine docks page_content's into a string
        messages = prompt_template.format_messages(question=question, context=context)
        return llm.invoke(messages)

    # RAG usage
    response = rag_chain_fn("Czego dotyczy sprawa III AUa 1002/23?")
    print(response.content)


if __name__ == '__main__':
    main()