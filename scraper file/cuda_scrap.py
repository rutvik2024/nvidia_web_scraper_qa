#import necessary libaries
import scrapy
from scrapy.spiders import Rule, CrawlSpider
from scrapy.linkextractors import LinkExtractor
from bs4 import BeautifulSoup
import re
import unicodedata
import json

class CudaScraper(CrawlSpider):
    name = 'cuda_scraper' # name of the spider. So that when we start crawler we have to pass the name
    allowed_domains = ['docs.nvidia.com'] # allowed domains for crawler
    start_urls = ['https://docs.nvidia.com/cuda/'] # Startpoint of the crawler

    # # Add rule to filter perticular websites
    rules = (
        Rule(LinkExtractor(allow=r'/cuda/'), callback='parse_item', follow=True),
    )

    def __init__(self, *a, **kw):
        super(CudaScraper, self).__init__(*a, **kw)
        self.depth = 0
        self.max_depth = 5 #parent_link -> Sub_links -> Sub_links -> Sub_links -> Sub_links
        self.cleaned_data = []

    # To perform text cleaning
    def clean_text(self, text):
        # Remove HTML tags
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text()

        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Convert to lowercase
        text = text.lower()

        # Regular expression to match decimal and float numbers
        pattern = r'\b\d+\.?\d*\b'
        
        # Remove all numbers from the text
        cleaned_text = re.sub(pattern, '', text)
        
        # Remove extra spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Remove special characters (except period, comma, and question mark) and numbers
        text = re.sub(r'[^a-z\s\.\,\?]', '', cleaned_text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

        return text

    def parse_item(self, response):
        self.depth += 1

        if self.depth > self.max_depth:
            return
        
        content_div = response.css('div.document')

        if content_div:
            content = content_div.get()
            clean_content = self.clean_text(content)

            self.cleaned_data.append({
                'url': response.url,
                'content': clean_content,
                'depth': self.depth
            })

        # Extract and follow links within the allowed domain
        for link in response.css('a::attr(href)').getall():
            if link.startswith('/') or self.allowed_domains[0] in link:
                yield response.follow(link, self.parse_item)

        self.depth -= 1
    

    def closed(self, reason):
        with open('nvidia_cuda_cleaned_data.json', 'w', encoding='utf-8') as f:
            json.dump(self.cleaned_data, f, ensure_ascii=False, indent=2)
        print("Crawling completed. Cleaned data saved to nvidia_cuda_cleaned_data.json")