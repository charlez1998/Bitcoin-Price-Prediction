import scrapy


class BitcoinPricesSpider(scrapy.Spider):
    name = 'bitcoin_prices'
    allowed_domains = ['coinmarketcap.com/currencies/bitcoin/historical-data']
    start_urls = ['http://coinmarketcap.com/currencies/bitcoin/historical-data/']

    def parse(self, response):
        pass
