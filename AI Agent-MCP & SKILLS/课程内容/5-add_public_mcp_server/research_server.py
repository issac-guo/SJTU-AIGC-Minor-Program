import arxiv
import json
import os
import requests
import time
import random
from typing import List
from mcp.server.fastmcp import FastMCP

PAPER_DIR = "papers"

mcp = FastMCP("research")

@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")
    
    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}
    
    # Use arxiv to find the papers with retry mechanism
    max_retries = 3
    retry_delay = 5  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            # Configure client with timeout and retry settings
            client = arxiv.Client(
                page_size=10,
                delay_seconds=3.0,  # Respect arXiv's rate limits
                num_retries=2
            )
            
            # Search for the most relevant articles matching the queried topic
            search = arxiv.Search(
                query=topic,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            papers = client.results(search)
            
            # Process each paper and add to papers_info
            paper_ids = []
            for paper in papers:
                paper_ids.append(paper.get_short_id())
                paper_info = {
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'summary': paper.summary,
                    'pdf_url': paper.pdf_url,
                    'published': str(paper.published.date())
                }
                
                papers_info[paper.get_short_id()] = paper_info
            
            # Save updated papers_info to json file
            with open(file_path, "w") as json_file:
                json.dump(papers_info, json_file, indent=2)
            
            print(f"Results are saved in: {file_path}")
            
            return paper_ids
            
        except (requests.exceptions.HTTPError, arxiv.ArxivError) as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    # Return existing papers if any on final attempt failure
                    if papers_info:
                        existing_ids = list(papers_info.keys())[:max_results]
                        print(f"Maximum retries reached. Returning {len(existing_ids)} previously saved papers for this topic.")
                        return existing_ids
                    else:
                        raise Exception(f"Failed to fetch papers after {max_retries} attempts due to rate limits. Please try again later.")
            else:
                # For other errors, just propagate
                raise


@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.
    
    Args:
        paper_id: The ID of the paper to look for
        
    Returns:
        JSON string with paper information if found, error message if not found
    """
    
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            print(f"Looking for {paper_id} in {file_path}")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."

@mcp.tool()
def download_paper(paper_id: str) -> str:
    """
    Download a paper PDF based on its ID and save it locally.
    
    Args:
        paper_id: The ID of the paper to download
        
    Returns:
        Message indicating success or failure of the download
    """
    # First, find the paper information
    paper_info = None
    topic_dir = None
    
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            paper_info = papers_info[paper_id]
                            topic_dir = item
                            break
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    if not paper_info:
        return f"Paper with ID {paper_id} not found in our database. Please search for it first."
    
    # Create a PDFs directory for this topic
    pdf_dir = os.path.join(PAPER_DIR, topic_dir, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Generate a safe filename
    title = paper_info['title'].replace("/", "-")[:50]  # Remove problematic characters and limit length
    filename = f"{paper_id}_{title}.pdf"
    filepath = os.path.join(pdf_dir, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        return f"Paper already downloaded at: {filepath}"
    
    max_retries = 3
    retry_delay = 5  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            # Method 1: Using arxiv's built-in download with rate limit settings
            client = arxiv.Client(
                page_size=10,
                delay_seconds=3.0,  # Respect arXiv's rate limits
                num_retries=2
            )
            paper = next(client.results(arxiv.Search(id_list=[paper_id])))
            paper.download_pdf(dirpath=pdf_dir, filename=filename)
            return f"Successfully downloaded paper to: {filepath}"
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"Rate limit exceeded. Retrying download in {wait_time:.2f} seconds... (Attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                # If we've exhausted retries for Method 1, try Method 2
            
            try:
                # Method 2: Fallback to direct PDF URL download with rate limit handling
                pdf_url = paper_info['pdf_url']
                # Add headers to mimic a browser request
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # Use a session for better connection handling
                with requests.Session() as session:
                    response = session.get(pdf_url, headers=headers, timeout=30)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                
                return f"Successfully downloaded paper to: {filepath}"
            except Exception as fallback_error:
                if "429" in str(fallback_error) and attempt < max_retries - 1:
                    # Exponential backoff with jitter for Method 2
                    wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 2)
                    print(f"Rate limit exceeded on direct download. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 2}/{max_retries})")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    # For other errors, just retry
                    print(f"Download attempt failed: {str(fallback_error)}. Retrying... (Attempt {attempt + 2}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    return f"Failed to download paper after {max_retries} attempts: {str(fallback_error)}"
    
    return "Unexpected error during paper download"


if __name__ == "__main__":
    mcp.run(transport="stdio")


