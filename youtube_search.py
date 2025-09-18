from youtubesearchpython import VideosSearch

def fetch_youtube_links(query, max_results=3):
    """
    Fetch top YouTube video links for a given query.
    If no videos are found, fallback to a clickable search URL.
    """
    try:
        search_query = f"{query} cattle breed"
        videos_search = VideosSearch(search_query, limit=max_results)
        results = videos_search.result()
        links = [video["link"] for video in results.get("result", []) if video.get("link")]

        # Fallback if no results
        if not links:
            search_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
            links = [search_url]

        return links

    except Exception as e:
        print("Error fetching YouTube links:", e)
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        return [search_url]
