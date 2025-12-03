"""
Scrape FinViz news data for multiple tickers and save to CSV.
This replaces the need to manually run the sample notebook.
"""

from bs4 import BeautifulSoup
import requests
import pandas as pd
from pathlib import Path
import time

def scrape_finviz_ticker(ticker):
    """
    Scrape news headlines for a specific ticker from FinViz.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'MCD', 'AAPL')
    
    Returns:
        List of [ticker, date, time, headline]
    """
    print(f"Scraping {ticker}...")
    
    try:
        # Make a GET request to fetch the raw HTML content
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        html_content = requests.get(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        ).text
        
        # Parse the html content
        soup = BeautifulSoup(html_content, "lxml")
        
        # Get news table
        news_table = soup.find(id='news-table')
        if news_table is None:
            print(f"  Warning: No news table found for {ticker}")
            return []
        
        table_data = news_table.findAll('tr')
        
        # Hold the parsed news into a list
        parsed_news = []
        date = None  # Keep track of current date
        
        # Iterate through the news
        for x in table_data:
            try:
                # Read the text from the tr tag
                text = x.a.get_text()
                # Split the text in the td tag into a list 
                date_scrape = x.td.text.split()
                
                # If the length of 'date_scrape' is 1, it's just time (use previous date)
                # If not, it's date + time
                if len(date_scrape) == 1:
                    time_str = date_scrape[0]
                else:
                    date = date_scrape[0]
                    time_str = date_scrape[1]
                
                # Append ticker, date, time and headline
                parsed_news.append([ticker, date, time_str, text])
            except Exception as e:
                # Skip rows that don't have the expected format
                continue
        
        print(f"  Found {len(parsed_news)} headlines for {ticker}")
        return parsed_news
        
    except Exception as e:
        print(f"  Error scraping {ticker}: {e}")
        return []


def scrape_multiple_tickers(tickers, output_path='data/raw/finviz_news.csv', delay=2):
    """
    Scrape news for multiple tickers and save to CSV.
    
    Args:
        tickers: List of ticker symbols
        output_path: Path to save CSV file
        delay: Seconds to wait between requests (be nice to the server)
    """
    all_news = []
    
    for ticker in tickers:
        news = scrape_finviz_ticker(ticker)
        all_news.extend(news)
        
        # Be nice to the server - wait between requests
        if ticker != tickers[-1]:  # Don't wait after last ticker
            time.sleep(delay)
    
    # Convert to DataFrame
    columns = ['ticker', 'date', 'time', 'headline']
    df = pd.DataFrame(all_news, columns=columns)
    
    # Fix "Today" dates - replace with actual date
    from datetime import datetime
    today = datetime.now().strftime('%b-%d-%y')
    df['date'] = df['date'].replace('Today', today)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved {len(df)} headlines to {output_path}")
    print(f"  Tickers: {df['ticker'].unique().tolist()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


if __name__ == '__main__':
    # Tickers to scrape (matching the ones in your price data)
    TICKERS = ['MSFT', 'AAPL', 'NVDA']
    
    print("="*60)
    print("FinViz News Scraper")
    print("="*60)
    print(f"\nScraping news for: {', '.join(TICKERS)}")
    print("This may take a minute...\n")
    
    # Scrape and save
    df = scrape_multiple_tickers(TICKERS)
    
    print("\n" + "="*60)
    print("✓ DONE - You can now run notebook 02!")
    print("="*60)
    print("\nPreview of data:")
    print(df.head(10))
