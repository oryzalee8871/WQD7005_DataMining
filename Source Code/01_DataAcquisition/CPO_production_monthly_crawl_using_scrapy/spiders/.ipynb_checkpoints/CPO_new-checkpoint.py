import scrapy 
from demo_CPO_production.items import DemoCpoProductionItem
from scrapy.loader import ItemLoader


class CPOSpider(scrapy.Spider):
    name='CPO'

    start_urls=['http://mpoc.org.my/monthly-palm-oil-trade-statistics-2014/']

   

    def parse(self,response):
         for CPO in response.xpath("//table[@class='tableizer-table'][3]//tr"):
            # //table[@class='tableizer-table'][3]//tr
            #  //div[@class='elementor-element elementor-element-1568df43 elementor-widget elementor-widget-text-editor']//table[1]//tr
            l=ItemLoader(item=DemoCpoProductionItem(),selector=CPO)
            l.add_xpath('Month',".//td[1]/text()")
            l.add_xpath('Production',".//td[2]/text()")
            l.add_xpath('Year',".//th[4]")
            

            yield l.load_item()