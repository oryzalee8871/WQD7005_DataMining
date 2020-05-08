#create folder [scrapy startproject demo_project]
#execute to create csv file [scrapy crawl {your_projectname} -o {urcsvfile.json}]

import scrapy 
from demo_news.items import NewsItem
from scrapy.loader import ItemLoader

class NewsSpider(scrapy.Spider):
    name='news'

    start_urls=[
        'https://www.theedgemarkets.com/search-results?keywords=Palm+oil&fromDate=2000-01-01&toDate=2020-02-29']

    def parse(self,response):
        for news in response.xpath("//div[@class='views-row']"):
            l=ItemLoader(item=NewsItem(),selector=news)
            l.add_xpath('news_title',".//div[@class='views-field views-field-title']/span[@class='field-content']/a")
            l.add_xpath('news_date',".//div[@class='views-field views-field-nothing']/span[@class='field-content']/text()")
            l.add_xpath('news_content',".//div[@class='views-field views-field-body']/div[@class='field-content']//text()")
            yield l.load_item()


        next_page= response.xpath("//li[@class='pager-next']/a/@href").extract_first()
        if next_page is not None:
            next_page_link=response.urljoin(next_page)
            yield scrapy.Request(url=next_page_link,callback=self.parse)
