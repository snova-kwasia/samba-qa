PROMPT = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> \nYou are an AI assistant specializing in accurate information retrieval and analysis. Your primary goal is to provide reliable answers based solely on the given context. Follow these guidelines:\n\n1. Answer Step-by-Step:\n   - Break down your reasoning process.\n   - Explain how each piece of information relates to the question.\n\n2. Source Attribution:\n   - ALWAYS cite sources for any information you use in your response.\n   - Use the following format for citations:\n     a) For direct quotes: [Quote from {source type}: \"exact text\"]\n     b) For paraphrased information: [Paraphrased from {source type}: summary of information]\n   - {source type} can be: website, document, article, book, report, or any other relevant source identifier provided in the context.\n   - If the source type is not clear, use [Source: \"exact text\" or summary]\n   - When possible, include any additional source details provided, such as author, date, or title.\n\n3. Accuracy Check:\n   - After formulating your answer, review it against the context.\n   - Ensure every statement is supported by the provided information.\n\n4. Handling Uncertainty:\n   - If the context doesn't contain enough information, say \"The given context does not provide sufficient information to answer this question fully.\"\n   - For partial answers, clearly state what you can and cannot answer based on the context.\n\n5. No External Knowledge:\n   - Do not use any information beyond the given context.\n   - If tempted to add external information, stop and reassess.\n\n6. Avoid Assumptions:\n   - Do not infer or extrapolate beyond what's explicitly stated in the context.\n   - If the question requires assumptions, clearly state them as such.\n\nRemember: It's better to provide a partial answer or admit lack of information than to give an inaccurate or unsupported response.\n\nContext: {context}\n\nQuestion: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
QUERY_WITH_VECTOR_STORE_RETRIEVER_SIMILARITY = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "vectorstore",
    "retriever_config": {"search_type": "similarity", "search_kwargs": {"k": 10}},
    "stream": False,
}

QUERY_WITH_VECTOR_STORE_RETRIEVER_PAYLOAD = {
    "summary": "search with similarity",
    "description": """
        Requires k in search_kwargs for similarity search.
        search_type can either be similarity or mmr or similarity_score_threshold.""",
    "value": QUERY_WITH_VECTOR_STORE_RETRIEVER_SIMILARITY,
}
#######

QUERY_WITH_VECTOR_STORE_RETRIEVER_MMR = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "vectorstore",
    "retriever_config": {
        "search_type": "mmr",
        "search_kwargs": {
            "k": 5,
            "fetch_k": 7,
        },
    },
    "stream": False,
}

QUERY_WITH_VECTOR_STORE_RETRIEVER_MMR_PAYLOAD = {
    "summary": "search with mmr",
    "description": """
        Requires k and fetch_k in search_kwargs for mmr support depends on vector db.
        search_type can either be similarity or mmr or similarity_score_threshold""",
    "value": QUERY_WITH_VECTOR_STORE_RETRIEVER_MMR,
}
#######

QUERY_WITH_VECTOR_STORE_RETRIEVER_SIMILARITY_SCORE = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "vectorstore",
    "retriever_config": {
        "search_type": "similarity_score_threshold",
        "search_kwargs": {"score_threshold": 0.7},
    },
    "stream": False,
}

QUERY_WITH_VECTOR_STORE_RETRIEVER_SIMILARITY_SCORE_PAYLOAD = {
    "summary": "search with threshold score",
    "description": """
        Requires score_threshold float (0~1) in search kwargs.
        search_type can either be similarity or mmr or similarity_score_threshold.""",
    "value": QUERY_WITH_VECTOR_STORE_RETRIEVER_SIMILARITY_SCORE,
}
#######

QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "contextual-compression",
    "retriever_config": {
        "compressor_model_provider": "mixedbread-ai",
        "compressor_model_name": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "top_k": 10,
        "search_type": "similarity",
        "search_kwargs": {"k": 15},
    },
    "stream": False,
}

QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_PAYLOAD = {
    "summary": "similarity search + re-ranking",
    "description": """
        Requires k in search_kwargs for similarity search.
        search_type can either be similarity or mmr or similarity_score_threshold.""",
    "value": QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER,
}
#####


QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_SEARCH_TYPE_MMR = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "contextual-compression",
    "retriever_config": {
        "compressor_model_provider": "mixedbread-ai",
        "compressor_model_name": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "top_k": 5,
        "search_type": "mmr",
        "search_kwargs": {
            "k": 10,
            "fetch_k": 30,
        },
    },
    "stream": False,
}

QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_SEARCH_TYPE_MMR_PAYLOAD = {
    "summary": "mmr + re-ranking",
    "description": """
        Requires k and fetch_k in search kwargs for mmr.
        search_type can either be similarity or mmr or similarity_score_threshold.
        Currently only support for mixedbread-ai/mxbai-rerank-xsmall-v1 reranker is added.""",
    "value": QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_SEARCH_TYPE_MMR,
}

#####


QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_SEARCH_TYPE_SIMILARITY_WITH_SCORE = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "contextual-compression",
    "retriever_config": {
        "compressor_model_provider": "mixedbread-ai",
        "compressor_model_name": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "top_k": 5,
        "search_type": "similarity_score_threshold",
        "search_kwargs": {"score_threshold": 0.7},
    },
    "stream": False,
}

QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_SEARCH_TYPE_SIMILARITY_WITH_SCORE_PAYLOAD = {
    "summary": "threshold score + re-ranking",
    "description": """
        Requires score_threshold float (0~1) in search kwargs for similarity search.
        search_type can either be similarity or mmr or similarity_score_threshold.
        Currently only support for mixedbread-ai/mxbai-rerank-xsmall-v1 reranker is added""",
    "value": QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_SEARCH_TYPE_SIMILARITY_WITH_SCORE,
}

#####

QUERY_WITH_MULTI_QUERY_RETRIEVER_SIMILARITY = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "multi-query",
    "retriever_config": {
        "search_type": "similarity",
        "search_kwargs": {"k": 5},
        "retriever_llm_configuration": {
            "name": "truefoundry/openai-main/gpt-3-5-turbo",
            "provider": "truefoundry",
            "parameters": {"temperature": 0.9},
        },
    },
    "stream": False,
}

QUERY_WITH_MULTI_QUERY_RETRIEVER_SIMILARITY_PAYLOAD = {
    "summary": "multi-query + similarity search",
    "description": """
        Typically used for complex user queries.
        search_type can either be similarity or mmr or similarity_score_threshold.""",
    "value": QUERY_WITH_MULTI_QUERY_RETRIEVER_SIMILARITY,
}
#######


QUERY_WITH_MULTI_QUERY_RETRIEVER_MMR = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "multi-query",
    "retriever_config": {
        "search_type": "mmr",
        "search_kwargs": {
            "k": 5,
            "fetch_k": 10,
        },
        "retriever_llm_configuration": {
            "name": "truefoundry/openai-main/gpt-3-5-turbo",
            "provider": "truefoundry",
            "parameters": {"temperature": 0.9},
        },
    },
    "stream": False,
}

QUERY_WITH_MULTI_QUERY_RETRIEVER_MMR_PAYLOAD = {
    "summary": "multi-query + mmr",
    "description": """
        Requires k and fetch_k in search_kwargs for mmr.
        search_type can either be similarity or mmr or similarity_score_threshold.""",
    "value": QUERY_WITH_MULTI_QUERY_RETRIEVER_MMR,
}
#######

QUERY_WITH_MULTI_QUERY_RETRIEVER_SIMILARITY_SCORE = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "multi-query",
    "retriever_config": {
        "search_type": "similarity_score_threshold",
        "search_kwargs": {"score_threshold": 0.7},
        "retriever_llm_configuration": {
            "name": "truefoundry/openai-main/gpt-3-5-turbo",
            "provider": "truefoundry",
            "parameters": {"temperature": 0.9},
        },
    },
    "stream": False,
}

QUERY_WITH_MULTI_QUERY_RETRIEVER_SIMILARITY_SCORE_PAYLOAD = {
    "summary": "multi-query + threshold score",
    "description": """
        Typically used for complex user queries.
        Requires score_threshold float (0~1) in search kwargs.
        search_type can either be similarity or mmr or similarity_score_threshold.""",
    "value": QUERY_WITH_MULTI_QUERY_RETRIEVER_SIMILARITY_SCORE,
}
#######


QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_MMR = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "contextual-compression-multi-query",
    "retriever_config": {
        "compressor_model_provider": "mixedbread-ai",
        "compressor_model_name": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "top_k": 5,
        "search_type": "mmr",
        "search_kwargs": {
            "k": 10,
            "fetch_k": 30,
        },
        "retriever_llm_configuration": {
            "name": "truefoundry/openai-main/gpt-3-5-turbo",
            "parameters": {"temperature": 0.9},
        },
    },
    "stream": False,
}

QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_MMR_PAYLOAD = {
    "summary": "multi-query + re-ranking +  mmr",
    "description": """
        Typically used for complex user queries.
        Requires k and fetch_k in search_kwargs for mmr.
        search_type can either be similarity or mmr or similarity_score_threshold.""",
    "value": QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_MMR,
}
#######

QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "contextual-compression-multi-query",
    "retriever_config": {
        "compressor_model_provider": "mixedbread-ai",
        "compressor_model_name": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "top_k": 5,
        "search_type": "similarity",
        "search_kwargs": {"k": 10},
        "retriever_llm_configuration": {
            "name": "truefoundry/openai-main/gpt-3-5-turbo",
            "provider": "truefoundry",
            "parameters": {"temperature": 0.1},
        },
    },
    "stream": False,
}

QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY_PAYLOAD = {
    "summary": "multi-query + re-ranking + similarity ",
    "description": """
        Typically used for complex user queries.
        Requires k in search_kwargs for similarity search.
        search_type can either be similarity or mmr or similarity_score_threshold.""",
    "value": QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY,
}
#######

QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY_SCORE = {
    "collection_name": "creditcard",
    "query": "Explain in detail different categories of credit cards",
    "model_configuration": {
        "name": "truefoundry/openai-main/gpt-3-5-turbo",
        "parameters": {"temperature": 0.1},
    },
    "prompt_template": PROMPT,
    "retriever_name": "contextual-compression-multi-query",
    "retriever_config": {
        "compressor_model_provider": "mixedbread-ai",
        "compressor_model_name": "mixedbread-ai/mxbai-rerank-xsmall-v1",
        "top_k": 5,
        "search_type": "similarity_score_threshold",
        "search_kwargs": {"score_threshold": 0.7},
        "retriever_llm_configuration": {
            "name": "truefoundry/openai-main/gpt-3-5-turbo",
            "provider": "truefoundry",
            "parameters": {"temperature": 0.1},
        },
    },
    "stream": False,
}

QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY_SCORE_PAYLOAD = {
    "summary": "multi-query + re-ranking + threshold score",
    "description": """
        Typically used for complex user queries.
        Requires k in search_kwargs for similarity search.
        search_type can either be similarity or mmr or similarity_score_threshold.""",
    "value": QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY_SCORE,
}
#######
