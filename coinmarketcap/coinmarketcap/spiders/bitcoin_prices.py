import scrapy


class BitcoinPricesSpider(scrapy.Spider):
    name = 'bitcoin_prices'
    allowed_domains = ['coinmarketcap.com/']
    start_urls = ['https://coinmarketcap.com/currencies/bitcoin/historical-data/']

    def parse(self, response):
        pass 
