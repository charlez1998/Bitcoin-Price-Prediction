import scrapy

class BitcoinPricesSpider(scrapy.Spider):
    name = 'bitcoin_prices'
 
    allowed_domains = ['coinmarketcap.com/']

    start_urls = ['https://coinmarketcap.com/currencies/bitcoin/historical-data/']

    def parse(self, response):
        yield {
            'Date': response.xpath("//td[1]/div/text()").get(),
            'Open': response.xpath("//td[2]/div/text()").get(),
            'High': response.xpath("//td[3]/div/text()").get(),
            'Low': response.xpath("//td[4]/div/text()").get(),
            'Close': response.xpath("//td[5]/div/text()").get(),
            'Volume': response.xpath("//td[6]/div/text()").get(),
            'Market Cap': response.xpath("//td[7]/div/text()").get()
        }
