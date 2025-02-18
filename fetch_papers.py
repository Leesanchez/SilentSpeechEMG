import requests

# Search Semantic Scholar
def search_semantic_scholar(query, limit=10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,authors,year,abstract",
        "limit": limit
    }
    response = requests.get(url, params=params)
    return response.json()

# Search ArXiv
def search_arxiv(query, max_results=10):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "max_results": max_results
    }
    response = requests.get(url, params=params)
    return response.text  # ArXiv returns XML, needs parsing

# Search Zenodo
def search_zenodo(query, size=10):
    url = "https://zenodo.org/api/records"
    params = {"q": query, "size": size}
    response = requests.get(url, params=params)
    return response.json()

if __name__ == "__main__":
    query = "machine learning"

    print("🔍 Searching Semantic Scholar...")
    papers = search_semantic_scholar(query)
    for paper in papers.get("data", []):
        print(f"- {paper['title']} ({paper['year']})")

    print("\n🔍 Searching ArXiv...")
    print(search_arxiv(query))  # Needs XML parsing

    print("\n🔍 Searching Zenodo Datasets...")
    datasets = search_zenodo(query)
    for dataset in datasets.get("hits", {}).get("hits", []):
        print(f"- {dataset['title']}")
