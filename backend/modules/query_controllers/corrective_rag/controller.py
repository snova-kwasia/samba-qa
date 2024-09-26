import os
import asyncio
import json
from typing import Any, Dict, List, TypedDict

import async_timeout
from fastapi import Body, HTTPException
from fastapi.responses import StreamingResponse
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import load_prompt
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents.base import Document

from langgraph.graph import END, StateGraph
from langgraph.graph import END, StateGraph
from langgraph.checkpoint import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.graph import CompiledGraph

from backend.logger import logger
from backend.modules.metadata_store.client import get_client
from backend.modules.model_gateway.model_gateway import model_gateway
from backend.modules.query_controllers.example.payload import (
    QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY_PAYLOAD,
    QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_PAYLOAD,
    QUERY_WITH_VECTOR_STORE_RETRIEVER_PAYLOAD,
)
from backend.modules.query_controllers.example.types import (
    GENERATION_TIMEOUT_SEC,
    ExampleQueryInput,
)
from backend.modules.rerankers.reranker_svc import InfinityRerankerSvc
from backend.modules.vector_db.client import VECTOR_STORE_CLIENT
from backend.server.decorators import post, query_controller
from backend.settings import settings
from backend.types import Collection, ModelConfig

# Import LangSmith For Tracing, W&B TODO
from langsmith import traceable

current_dir = os.getcwd()

EXAMPLES = {
    "vector-store-similarity": QUERY_WITH_VECTOR_STORE_RETRIEVER_PAYLOAD,
}

if settings.RERANKER_SVC_URL:
    EXAMPLES.update(
        {
            "contextual-compression-similarity": QUERY_WITH_CONTEXTUAL_COMPRESSION_RETRIEVER_PAYLOAD,
        }
    )
    EXAMPLES.update(
        {
            "contextual-compression-multi-query-similarity": QUERY_WITH_CONTEXTUAL_COMPRESSION_MULTI_QUERY_RETRIEVER_SIMILARITY_PAYLOAD,
        }
    )

class RAGGraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    answers: List[str]
    original_question: str
    """
    Represents the state of a RAG (Retrieval Augmented Generation) graph.

    Args:
        question: The question being asked by the user.
        generation: The most recent generated text from the LLM agent.
        documents: A list of relevant documents retrieved from the vectorstore.
        answers: A list of possible answers that have been accumulated by the LLM agents.
        original_question: The original question asked - in case of subquery generation, etc..
    """

@query_controller("/corrective-rag")
class CorrectiveRAGQueryController:
    
    def init_retrieval_grader(self) -> None:
        """
        Initializes the retrieval grader component.

        This method loads the retrieval grader prompt from the repository and
        combines it with the language model and a JSON output parser.

        Args:
            None

        Returns:
            None
        """

        retrieval_grader_prompt: Any = load_prompt(current_dir + "/backend/modules/query_controllers/corrective_rag/prompts/retrieval_grader.yaml")
        self.retrieval_grader = retrieval_grader_prompt | self.llm | JsonOutputParser()

    def init_qa_chain(self) -> None:
        """
        Initializes the QA chain by loading the QA prompt and
        combining it with the large language model and a string
        output parser.

        Args:
            None

        Returns:
            None
        """

        qa_prompt: Any = load_prompt(current_dir + "/backend/modules/query_controllers/corrective_rag/prompts/qa.yaml")
        self.qa_chain = qa_prompt | self.llm | StrOutputParser()

    def init_hallucination_chain(self) -> None:
        """
        Initializes the hallucination chain for the model.

        This method loads the hallucination prompt from the repository and
        combines it with the language model (LLM) and a JSON output parser to
        create the hallucination chain.

        Args:
            None

        Returns:
            None
        """

        hallucination_prompt: Any = load_prompt(current_dir + "/backend/modules/query_controllers/corrective_rag/prompts/hallucination_grader.yaml")
        self.hallucination_chain = hallucination_prompt | self.llm | JsonOutputParser()

    def init_grading_chain(self) -> None:
        """
        Initializes the grading chain by loading the grading prompt and
        combining it with the LLM and a JSON output parser.

        Args:
            None

        Returns:
            None
        """

        grading_prompt: Any = load_prompt(current_dir + "/backend/modules/query_controllers/corrective_rag/prompts/answer_grader.yaml")
        self.grading_chain = grading_prompt | self.llm | JsonOutputParser()

    def init_failure_chain(self) -> None:
        """
        Initializes the failure chain by loading the failure prompt and
        combining it with the language model and a string output parser.

        Args:
            None

        Returns:
            None
        """

        failure_prompt: Any = load_prompt(current_dir + "/backend/modules/query_controllers/corrective_rag/prompts/failure_msg.yaml")
        self.failure_chain = failure_prompt | self.llm | StrOutputParser()

    def init_final_generation(self) -> None:
        """
        Initializes the final generation process by loading the final chain
        prompt and combining it with the language model (LLM) and a string
        output parser.

        Args:
            None

        Returns:
            None
        """

        final_chain_prompt: Any = load_prompt(current_dir + "/backend/modules/query_controllers/corrective_rag/prompts/final_msg.yaml")
        self.final_chain = final_chain_prompt | self.llm | StrOutputParser()
    
    def initialize(self) -> None:
        """
        Initializes all the components of the static CorrectiveRAG app.
        """

        # self._get_llm()
        self.init_qa_chain()
        self.init_retrieval_grader()
        self.init_hallucination_chain()
        self.init_grading_chain()
        self.init_failure_chain()
        self.init_final_generation()

    def initialize_rag(self, state: dict) -> dict:
        """
        Initializes the state of the RAG components for LangGraph.

        Args:
            state: A dictionary containing the state of the RAG components.

        Returns:
            The state dict with the question stored in original question.
        """

        print('---Initializing---')

        print('---INITIAL STATE---')
        print(state)

        question: str = state['question']
        print(question)

        return {'answers': [], 'original_question': question}

    async def retrieve(self, state: dict) -> dict:
        """
        Retrieves relevant documents based on a given question.

        Args:
            state: A dictionary containing the question to be retrieved.

        Returns:
            A dictionary containing the retrieved documents and the original question.
        """

        question: str = state['question']

        print('---RETRIEVING FOR QUESTION---')
        print(question)

        documents: List[Document] = await self.retriever.ainvoke(question)

        return {'documents': documents, 'question': question}

    async def grade_documents(self, state: dict) -> dict:
        """
        Grades a list of documents based on their relevance to a given question.

        Args:
            state: A dictionary containing the question and documents to be graded.

        Returns:
            A dictionary containing the graded documents and the
            original question.
        """

        print('---CHECK DOCUMENT RELEVANCE TO QUESTION---')
        question: str = state['question']
        documents: List[Document] = state['documents']

        # Score each doc
        filtered_docs: List = []
        for d in documents:
            try:
                score: Dict[str, str] = await self.retrieval_grader.ainvoke({'question': question, 'document': d.page_content})
            except Exception as e:
                print(e)
            grade: str = score['score']

            # Document relevant
            if grade.lower() == 'yes':
                print('---GRADE: DOCUMENT RELEVANT---')
                filtered_docs.append(d)

            # Document not relevant
            else:
                print('---GRADE: DOCUMENT NOT RELEVANT---')
                continue

        return {'documents': filtered_docs, 'question': question}
    
    async def check_hallucinations(self, state: dict) -> str:
        """
        Checks if the generated text is grounded in the provided documents and addresses the question.

        Args:
            state: The state dictionary containing the question, documents, and generation, and other variables.

        Returns:
            A string indicating the usefulness of the generated text.
        """

        print('---CHECK FOR HALLUCINATIONS---')
        question: str = state['question']
        documents: List[Document] = state['documents']
        generation: str = state['generation']

        docs: str = self._format_docs(documents)
        score: Dict[str, str] = {}

        try:
            score = await self.hallucination_chain.ainvoke({'documents': docs, 'generation': generation})
        except Exception as e:
            print(e)
        print(score)
        grade: str = score['score']

        # Check hallucination
        if grade == 'yes':
            print('---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---')
            # Check question-answering
            print('---GRADE GENERATION vs QUESTION---')
            score = await self.grading_chain.ainvoke({'question': question, 'generation': generation})
            grade = score['score']
            if grade == 'yes':
                print('---DECISION: GENERATION ADDRESSES QUESTION---')
                routing = 'useful'
            else:
                print('---DECISION: GENERATION DOES NOT ADDRESS QUESTION---')
                routing = 'not useful'
        else:
            print('---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---')
            routing = 'not supported'

        return routing
    
    async def rag_generate(self, state: dict) -> dict:
        """
        Generates an answer to a question using a question answering chain.

        Args:
            state : A dictionary containing the question, documents, answers,
            and other variables.

        Returns:
            The updated state dict containing the generated answer and the
            updated list of answers.
        """

        print('---GENERATING---')
        question: str = state['question']
        documents: List[str] = state['documents']
        answers: List[str] = state['answers']

        print('---ANSWERING---')
        print(question)

        docs: str = self._format_docs(documents)

        print('---DOCS---')
        n_tokens = len(docs.split()) * 1.3
        print('number of approximate tokens (n words *1.3): ', n_tokens)
        print(docs)

        generation: str = await self.qa_chain.ainvoke({'question': question, 'context': docs})

        print('---ANSWER---')
        print(generation)
        if not isinstance(answers, List):
            answers = [answers]

        answers.append(generation)

        return {'generation': generation, 'answers': answers}
    
    async def failure_msg(self, state: dict) -> dict:
        """
        This method generates a failure message based on the given state.

        Args:
            state: A dictionary containing the current state of the system.

        Returns:
            The updated state dictionary containing the failure message.
        """

        question: str = state['question']

        failure_msg: str = await self.failure_chain.ainvoke({'question': question})

        return {'answers': failure_msg}
    
    async def final_answer(self, state: dict) -> dict:
        """
        This method is used to generate the final answer based on the original question and the generated text.

        Args:
            state: The state dictionary containing the original
            question, the generated text, and other variables.

        Returns:
            The updated state dictionary containing the final answer.
        """

        original_question: str = state['original_question']
        generation: str = state['generation']

        print('---Final Generation---')
        print(generation)

        final_answer: str = await self.final_chain.ainvoke({'question': original_question, 'generation': generation})

        return {'generation': final_answer}

    def create_rag_nodes(self) -> StateGraph:
        """
        Creates the nodes for the CorrectiveRAG graph state.

        Args:
            None

        Returns:
            The StateGraph object containing the nodes for the CodeRAG graph state.
        """

        workflow: StateGraph = StateGraph(RAGGraphState)

        # Define the nodes

        workflow.add_node('initialize', self.initialize_rag)
        workflow.add_node('retrieve', self.retrieve)
        workflow.add_node('grade_documents', self.grade_documents)
        workflow.add_node('generate', self.rag_generate)
        workflow.add_node('failure_msg', self.failure_msg)
        workflow.add_node('return_final_answer', self.final_answer)

        return workflow
    
    def build_rag_graph(self, workflow: StateGraph) -> CompiledGraph:
        """
        Builds a graph for the RAG workflow.

        This method constructs a workflow graph that represents the sequence of tasks
        performed by the RAG system. The graph is used to execute the workflow and
        generate code.

        Args:
            workflow: The workflow object (StateGraph containing nodes) to be modified.

        Returns:
            The compiled application object for static CodeRAG
        """

        # checkpointer = MemorySaver()

        workflow.set_entry_point('initialize')
        workflow.add_edge('initialize', 'retrieve')
        workflow.add_edge('retrieve', 'grade_documents')
        workflow.add_edge('grade_documents', 'generate')
        workflow.add_conditional_edges(
            'generate',
            self.check_hallucinations,
            {
                'not supported': 'failure_msg',
                'useful': 'return_final_answer',
                'not useful': 'failure_msg',
            },
        )
        workflow.add_edge('failure_msg', 'return_final_answer')
        workflow.add_edge('return_final_answer', END)

        app: CompiledGraph = workflow.compile()

        return app
    
    async def call_rag(
        self, app: CompiledStateGraph, 
        question: str, 
        kwargs: Dict[str, int] = {'recursion_limit': 50}
    ) -> tuple[dict[str, Any], Dict[str, Any] | Any]:
        """
        Calls the RAG (Reasoning and Generation) app to generate an answer to a given question.

        Args:
            app: The RAG app object.
            question: The question to be answered.
            kwargs: Keyword arguments to be passed to the app.
            Defaults to {"recursion_limit": 50}.
            Recursion limit controls how many runnables to invoke without
            reaching a terminal node.

        Returns:
            response: A dictionary containing the answer and source documents.
                - "answer" (str): The generated answer to the question.
                - "source_documents" (List[str]): A list of source documents used
                to generate the answer.
        """

        # runnable = RunnableConfig(configurable={'question': question})

        # response = {}
        # output = app.invoke(runnable, kwargs=kwargs)
        output = await app.ainvoke({'question': question})
        # response['answer'] = output['generation']
        # response['context'] = output['documents']

        return output

    def _format_docs(self, docs):
        formatted_docs = list()
        for doc in docs:
            doc.metadata.pop("image_b64", None)
            formatted_docs.append(
                {"page_content": doc.page_content, "metadata": doc.metadata}
            )
        return "\n\n".join([f"{doc['page_content']}" for doc in formatted_docs])

    def _format_docs_for_stream(self, docs):
        formatted_docs = list()
        for doc in docs:
            doc.metadata.pop("image_b64", None)
            formatted_docs.append(
                {"page_content": doc.page_content, "metadata": doc.metadata}
            )
        return formatted_docs

    def _get_llm(self, model_configuration: ModelConfig, stream=False):
        """
        Get the LLM
        """
        return model_gateway.get_llm_from_model_config(model_configuration, stream)

    async def _get_vector_store(self, collection_name: str):
        """
        Get the vector store for the collection
        """
        client = await get_client()
        collection = await client.aget_collection_by_name(collection_name)
        if collection is None:
            raise HTTPException(status_code=404, detail="Collection not found")

        if not isinstance(collection, Collection):
            collection = Collection(**collection.dict())

        return VECTOR_STORE_CLIENT.get_vector_store(
            collection_name=collection.name,
            embeddings=model_gateway.get_embedder_from_model_config(
                model_name=collection.embedder_config.model_config.name
            ),
        )

    def _get_vector_store_retriever(self, vector_store, retriever_config):
        """
        Get the vector store retriever
        """
        return VectorStoreRetriever(
            vectorstore=vector_store,
            search_type=retriever_config.search_type,
            search_kwargs=retriever_config.search_kwargs,
        )

    def _get_contextual_compression_retriever(self, vector_store, retriever_config):
        """
        Get the contextual compression retriever
        """
        try:
            if settings.RERANKER_SVC_URL:
                retriever = self._get_vector_store_retriever(
                    vector_store, retriever_config
                )
                logger.info("Using MxBaiRerankerSmall th' service...")
                compressor = InfinityRerankerSvc(
                    top_k=retriever_config.top_k,
                    model=retriever_config.compressor_model_name,
                )

                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=retriever
                )

                return compression_retriever
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Reranker service is not available",
                )
        except Exception as e:
            logger.error(f"Error in getting contextual compression retriever: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error in getting contextual compression retriever",
            )

    def _get_multi_query_retriever(
        self, vector_store, retriever_config, retriever_type="vectorstore"
    ):
        """
        Get the multi query retriever
        """
        if retriever_type == "vectorstore":
            base_retriever = self._get_vector_store_retriever(
                vector_store, retriever_config
            )
        elif retriever_type == "contextual-compression":
            base_retriever = self._get_contextual_compression_retriever(
                vector_store, retriever_config
            )
        else:
            raise ValueError(f"Unknown retriever type `{retriever_type}`")

        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=self._get_llm(retriever_config.retriever_llm_configuration),
        )

    async def _get_retriever(self, vector_store, retriever_name, retriever_config):
        """
        Get the retriever
        """
        if retriever_name == "vectorstore":
            logger.debug(
                f"Using VectorStoreRetriever with {retriever_config.search_type} search"
            )
            retriever = self._get_vector_store_retriever(vector_store, retriever_config)

        elif retriever_name == "contextual-compression":
            logger.debug(
                f"Using ContextualCompressionRetriever with {retriever_config.search_type} search"
            )
            retriever = self._get_contextual_compression_retriever(
                vector_store, retriever_config
            )

        elif retriever_name == "multi-query":
            logger.debug(
                f"Using MultiQueryRetriever with {retriever_config.search_type} search"
            )
            retriever = self._get_multi_query_retriever(vector_store, retriever_config)

        elif retriever_name == "contextual-compression-multi-query":
            logger.debug(
                f"Using MultiQueryRetriever with {retriever_config.search_type} search and "
                f"retriever type as {retriever_name}"
            )
            retriever = self._get_multi_query_retriever(
                vector_store, retriever_config, retriever_type="contextual-compression"
            )

        else:
            raise HTTPException(status_code=404, detail="Retriever not found")
        return retriever

    async def _stream_answer(self, rag_chain, query):
        async with async_timeout.timeout(GENERATION_TIMEOUT_SEC):
            try:
                async for chunk in rag_chain.astream(query):
                    if "context" in chunk:
                        yield json.dumps(
                            {"docs": self._format_docs_for_stream(chunk["context"])}
                        )
                        await asyncio.sleep(0.1)
                    elif "answer" in chunk:
                        # print("Answer: ", chunk['answer'])
                        yield json.dumps({"answer": chunk["answer"]})
                        await asyncio.sleep(0.1)

                yield json.dumps({"end": "<END>"})
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Stream timed out")

    @traceable()
    @post("/answer")
    async def answer(
        self,
        request: ExampleQueryInput = Body(
            openapi_examples=EXAMPLES,
        ),
    ):
        """
        Sample answer method to answer the question using the context from the collection
        """
        try:
            print(f"This is the request {request}")
            # Get the vector store
            vector_store = await self._get_vector_store(request.collection_name)

            # Get the LLM
            self.llm = self._get_llm(request.model_configuration, request.stream)

            print(f"This is the LLM {self.llm}")

            # get retriever
            self.retriever = await self._get_retriever(
                vector_store=vector_store,
                retriever_name=request.retriever_name,
                retriever_config=request.retriever_config,
            )

            self.initialize()

            self.workflow = self.create_rag_nodes()

            self.app = self.build_rag_graph(self.workflow)



            # rag_chain_with_source = RunnableParallel(
            #     {"context": retriever, "question": RunnablePassthrough()}
            # ).assign(answer=rag_chain_from_docs)

            if request.stream:
                 return StreamingResponse(
                    self._stream_answer(self.call_rag(self.app, request.query), request.query),
                    media_type="text/event-stream",
                )

            else:
                outputs = await self.call_rag(self.app, request.query)
                # outputs = await self.app.ainvoke({"question": request.query})
                print("---OUTPUTS---")
                print(outputs)
                
                print("---REQUEST---")
                print(request.query)
                docs =  await self.retriever.ainvoke(request.query)
                print(docs)

                # Intermediate testing
                # Just the retriever
                # setup_and_retrieval = RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
                # outputs = await setup_and_retrieval.ainvoke(request.query)
                # print(outputs)

                # Retriever and QA
                # outputs = await (setup_and_retrieval | QA_PROMPT).ainvoke(request.query)
                # print(outputs)

                # Retriever, QA and LLM
                # outputs = await (setup_and_retrieval | QA_PROMPT | llm).ainvoke(request.query)
                # print(outputs)

                return {
                    "answer": outputs["generation"],
                    "docs": outputs["documents"] if outputs["documents"] else [],
                }

        except HTTPException as exp:
            raise exp
        except Exception as exp:
            logger.exception(exp)
            raise HTTPException(status_code=500, detail=str(exp))


#######
# Streaming Client

# import httpx
# from httpx import Timeout

# from backend.modules.query_controllers.example.types import ExampleQueryInput

# payload = {
#   "collection_name": "pstest",
#   "query": "What are the features of Diners club black metal edition?",
#   "model_configuration": {
#     "name": "openai-devtest/gpt-3-5-turbo",
#     "parameters": {
#       "temperature": 0.1
#     },
#     "provider": "truefoundry"
#   },
#   "prompt_template": "Answer the question based only on the following context:\nContext: {context} \nQuestion: {question}",
#   "retriever_name": "vectorstore",
#   "retriever_config": {
#     "search_type": "similarity",
#     "search_kwargs": {
#       "k": 20
#     },
#     "filter": {}
#   },
#   "stream": True
# }

# data = ExampleQueryInput(**payload).dict()
# ENDPOINT_URL = 'http://localhost:8000/retrievers/example-app/answer'


# with httpx.stream('POST', ENDPOINT_URL, json=data, timeout=Timeout(5.0*60)) as r:
#     for chunk in r.iter_text():
#         print(chunk)
#######
