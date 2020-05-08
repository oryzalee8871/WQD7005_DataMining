# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy

from scrapy.loader.processors import MapCompose, TakeFirst,Join
from w3lib.html import remove_tags


class DemoCpoProductionItem(scrapy.Item):
    Month= scrapy.Field(
        input_processor=MapCompose(remove_tags),
        
    )

    Production= scrapy.Field(
        input_processor=MapCompose(remove_tags),
       
    )

    Year= scrapy.Field(
        input_processor=MapCompose(remove_tags),
        output_processor=TakeFirst()
       
    )