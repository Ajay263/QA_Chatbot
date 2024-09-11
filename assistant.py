import os
import time
import json
import logging
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
from groq import Groq
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

GROQ_MODELS = ["mixtral-8x7b-32768", "llama-3.1-70b-versatile", "gemma2-9b-it", "llama3-70b-8192"]

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.debug(f"Elastic URL: {ELASTIC_URL}")
logging.debug("Groq API Key loaded successfully.")

es_client = Elasticsearch(ELASTIC_URL)
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def calculate_openai_cost(tokens: int, model: str) -> float:
    # This is a placeholder function. You should implement the actual cost calculation
    # based on OpenAI's pricing for the equivalent models.
    return tokens * 0.00002  # Example cost calculation

def create_groq_client():
    logging.debug("Creating Groq client...")
    if not GROQ_API_KEY:
        raise ValueError("Groq API key (GROQ_API_KEY) is not set. Please check your .env file or environment variables.")
    return Groq(api_key=GROQ_API_KEY)

def elastic_search_text(query: str, course: str, index_name: str = "course-questions") -> list:
    logging.debug(f"Performing text search with query: {query} and course: {course}")
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields",
                    }
                },
                "filter": {"term": {"course": course}},
            }
        },
    }

    response = es_client.search(index=index_name, body=search_query)
    search_results = [hit["_source"] for hit in response["hits"]["hits"]]
    logging.debug(f"Text search results: {search_results}")
    return search_results

def elastic_search_knn(field: str, vector: list, course: str, index_name: str = "course-questions") -> list:
    logging.debug(f"Performing KNN search with vector: {vector} and course: {course}")
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {"term": {"course": course}},
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"],
    }

    es_results = es_client.search(index=index_name, body=search_query)
    knn_results = [hit["_source"] for hit in es_results["hits"]["hits"]]
    logging.debug(f"KNN search results: {knn_results}")
    return knn_results

def build_prompt(query: str, search_results: list) -> str:
    logging.debug(f"Building prompt for query: {query}")
    prompt_template ="""
You are a knowledgeable and helpful course teaching assistant. Your task is to provide clear, direct, and easy-to-follow answers to student questions based only on the information given in the CONTEXT.

CONTEXT: {context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer the question directly as if you were a human assistant, without mentioning the CONTEXT explicitly.
2. Structure your answer in a way that is easy to follow:
   - Use bullet points or numbered lists for steps or key points.
   - Include headings if necessary to break down the answer into sections.
   - Highlight important terms or concepts for clarity.
3. If additional details or links are required, provide them naturally at the end of the response, but ensure that they are drawn only from the CONTEXT.
4. Ensure that your response is concise and easy to understand, staying focused on the question asked.
5. Avoid unnecessary details or overly complex explanations.

Provide your answer below:
""".strip()

    context = "\n\n".join(
        [
            f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}"
            for doc in search_results
        ]
    )
    prompt = prompt_template.format(question=query, context=context).strip()
    logging.debug(f"Built prompt: {prompt}")
    return prompt

def llm(prompt: str, model: str = "llama3-70b-8192") -> Tuple[str, Dict[str, int], float, float]:
    logging.debug("Generating LLM response...")
    try:
        groq_client = create_groq_client()
        start_time = time.time()
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
        )
        answer = response.choices[0].message.content
        end_time = time.time()
        response_time = end_time - start_time
        
        tokens = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens
        }
        
        openai_cost = calculate_openai_cost(tokens['total_tokens'], model)
        
        logging.debug(f"LLM response: {answer}")
        logging.debug(f"Response time: {response_time} seconds")
        
        return answer, tokens, response_time, openai_cost
    except Exception as e:
        logging.error(f"Error with Groq API request: {str(e)}")
        raise ValueError(f"Error with Groq API request: {str(e)}")


def evaluate_relevance(question: str, answer: str) -> Tuple[str, str, Dict[str, int]]:
    logging.debug("Evaluating relevance of generated answer...")
    prompt_template = """
    You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
    Your task is to analyze the relevance of the generated answer to the given question.
    Based on the relevance of the generated answer, you will classify it
    as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

    Here is the data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Please analyze the content and context of the generated answer in relation to the question
    and provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
      "Explanation": "[Provide a brief explanation for your evaluation]"
    }}
    """.strip()

    prompt = prompt_template.format(question=question, answer=answer)
    evaluation, tokens, _, _ = llm(prompt)  # Note the change here
    
    try:
        json_eval = json.loads(evaluation)
        logging.debug(f"Evaluation result: {json_eval}")
        return json_eval['Relevance'], json_eval['Explanation'], tokens
    except json.JSONDecodeError:
        logging.error("Failed to parse evaluation JSON.")
        return "UNKNOWN", "Failed to parse evaluation", tokens

def get_answer(query: str, course: str, search_type: str, model: str = "llama3-70b-8192") -> Dict[str, Any]:
    logging.debug(f"Getting answer for query: {query} with search type: {search_type} and model: {model}")
    try:
        if search_type == 'Vector':
            vector = model.encode(query)
            search_results = elastic_search_knn('question_text_vector', vector, course)
        else:
            search_results = elastic_search_text(query, course)

        prompt = build_prompt(query, search_results)
        answer, tokens, response_time, openai_cost = llm(prompt, model)
        
        relevance, explanation, eval_tokens = evaluate_relevance(query, answer)

        result = {
            'answer': answer,
            'response_time': response_time,
            'relevance': relevance,
            'relevance_explanation': explanation,
            'model_used': model,
            'prompt_tokens': tokens['prompt_tokens'],
            'completion_tokens': tokens['completion_tokens'],
            'total_tokens': tokens['total_tokens'],
            'eval_prompt_tokens': eval_tokens['prompt_tokens'],
            'eval_completion_tokens': eval_tokens['completion_tokens'],
            'eval_total_tokens': eval_tokens['total_tokens'],
            'openai_cost': openai_cost,
        }
        logging.debug(f"Final result: {result}")
        return result
    except ValueError as e:
        logging.error(f"ValueError: {str(e)}")
        return {
            'error': str(e),
            'model_used': model,
        }
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return {
            'error': f"An unexpected error occurred: {str(e)}",
            'model_used': model,
        }
