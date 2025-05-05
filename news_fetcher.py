import requests

API_KEY = ""  # Replace with your actual API key

def get_top_headlines(category="general", country="us"):
    url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={API_KEY}&pageSize=5&language=en"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for 4xx/5xx responses
        data = response.json()
        
        if data.get("status") == "ok" and data.get("articles"):
            return data["articles"][:5]  # Return only the first 5 articles
        else:
            print(f"No articles found for category '{category}' in {country}.")
            return []  # Return an empty list if no articles found
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return []
