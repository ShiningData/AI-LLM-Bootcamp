from x03_agent_chain_vbo import vector_store

# results = vector_store.similarity_search(query="Apache Spark dersi var mı? Varsa hangi konular işleniyor?", k=4)

# print("=================  Without retriver  =======================")
# for i, doc in enumerate(results, 1):
#     print(f"Doc number: {i} \n")
#     print(doc)



print("=================  retriver  =======================")

retriever = vector_store.as_retriever(search_type='similarity_score_threshold',
    search_kwargs={'score_threshold': 0.8})

results2 = retriever.invoke(input="Apache Spark dersi var mı? Varsa hangi konular işleniyor?")
for i, doc in enumerate(results2, 1):
    print(f"Doc number: {i} \n")
    print(doc)
    print(doc.id)