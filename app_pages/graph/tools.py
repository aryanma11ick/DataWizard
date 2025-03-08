from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

from langchain_core.messages import AIMessage
from typing import Annotated, Tuple
from langgraph.prebuilt import InjectedState
import sys
from io import StringIO
import os
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd
import requests
import wikipediaapi
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

wiki_wiki = wikipediaapi.Wikipedia('en')


@tool(parse_docstring=True)
def duckduckgo_search(query: str) -> Tuple[str, dict]:
    """
    Search DuckDuckGo for a query and generate questions and answers.

    Args:
        query (str): The search query string.

    Returns:
        Tuple[str, dict]: The formatted Q&A and state updates.
    """
    try:
        # Add user-agent header to avoid being blocked
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(
            f"https://api.duckduckgo.com/",
            params={
                'q': query,
                'format': 'json',
                't': 'ML_QA_Bot'
            },
            headers=headers
        )

        if response.status_code == 200:
            data = response.json()
            abstract = data.get('AbstractText', '')
            related_topics = data.get('RelatedTopics', [])

            # If no abstract, try to get information from related topics
            if not abstract and related_topics:
                abstract = related_topics[0].get('Text', 'No results found.')
            elif not abstract:
                abstract = 'No results found.'

            # Format the response as Q&A
            output = {
                "thought": f"Searching DuckDuckGo for information about {query}",
                "content": f"""
                    Question: What are the key concepts of {query} according to the search results?
                    Answer: {abstract}
                    """
            }

            return output["content"], {
                "intermediate_outputs": [{
                    "thought": output["thought"],
                    "output": abstract
                }]
            }

        error_msg = f"Error: DuckDuckGo API returned status code {response.status_code}"
        return error_msg, {
            "intermediate_outputs": [{
                "thought": "Error in DuckDuckGo search",
                "output": error_msg
            }]
        }

    except Exception as e:
        error_msg = f"Error in DuckDuckGo search: {str(e)}"
        return error_msg, {
            "intermediate_outputs": [{
                "thought": "Error in DuckDuckGo search",
                "output": error_msg
            }]
        }


@tool(parse_docstring=True)
def arxiv_search(query: str) -> Tuple[str, dict]:
    """
    Search arXiv for a query and generate questions and answers.

    Args:
        query (str): The search query string.

    Returns:
        Tuple[str, dict]: The formatted Q&A and state updates.
    """
    response = requests.get(f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1")
    if response.status_code == 200:
        result = response.text

        # Format the response as Q&A
        output = f"""
        Question: What are the recent research findings about {query} according to arXiv?
        Answer: {result}
        """
        return output, {"intermediate_outputs": [{"thought": f"Searching arXiv for {query}", "output": result}]}

    return 'Error fetching results from arXiv.', {
        "intermediate_outputs": [{"thought": "Error in arXiv search", "output": "Error"}]}


@tool(parse_docstring=True)
def wikipedia_search(query: str) -> Tuple[str, dict]:
    """
    Search Wikipedia for a query and generate questions and answers.

    Args:
        query (str): The search query string.

    Returns:
        Tuple[str, dict]: The formatted Q&A and state updates.
    """
    page = wiki_wiki.page(query)
    if page.exists():
        summary = page.summary

        # Format the response as Q&A
        output = f"""
        Question: What is {query} and what are its main concepts according to Wikipedia?
        Answer: {summary}
        """
        return output, {"intermediate_outputs": [{"thought": f"Searching Wikipedia for {query}", "output": summary}]}

    return 'No Wikipedia page found for this query.', {
        "intermediate_outputs": [{"thought": "Error in Wikipedia search", "output": "No page found"}]}
