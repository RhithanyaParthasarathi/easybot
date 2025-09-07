import requests
from bs4 import BeautifulSoup

def fetch_web_content(url: str) -> str:
    """Fetch and clean the website content into a string, using headers to avoid access denial."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to extract main content for Wikipedia or fallback to full page
        content_div = soup.find('div', {'id': 'bodyContent'})
        if content_div:
            text = content_div.get_text(separator=' ')
        else:
            text = soup.get_text(separator=' ')
        
        # Remove scripts and styles
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Clean text by removing extra spaces and newlines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
        
        return cleaned_text[:5000]
    except Exception as e:
        return f"Error fetching content: {str(e)}"

if __name__ == "__main__":
    print("Website Content Fetcher")
    url = input("Enter the website URL: ")
    content = fetch_web_content(url)
    print("\n--- Fetched Content ---\n")
    print(content)
