import llm_model

if __name__ == '__main__':
    llm = llm_model.create_llm()

    response = llm.invoke("Hello world")
    print(response.content)
