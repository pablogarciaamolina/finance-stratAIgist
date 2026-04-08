"""Internet search tool — uses Tavily API for web search."""

import os
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

load_dotenv()


@tool
def internet_search(query: str) -> str:
    """Busca información en internet usando Tavily API. SIEMPRE usa esta herramienta
    para buscar información sobre personas, lugares, tecnología o cualquier dato factual.
    Input: la consulta de búsqueda."""
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return "Error: TAVILY_API_KEY no está configurada en las variables de entorno."

        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(query=query, max_results=1)

        if response and "results" in response and len(response["results"]) > 0:
            first_result = response["results"][0]
            title = first_result.get("title", "")
            content = first_result.get("content", "")
            url = first_result.get("url", "")
            return f"{title}\n{content}\nFuente: {url}"
        else:
            return "No se encontraron resultados relevantes en la búsqueda."
    except Exception as e:
        return f"Error en la búsqueda: {e}"
