# -------------------------------------------------------------------
# Title        : Cleansing data
# Description  :
# Writer       : Watcharapong Wongrattanasirikul
# Created date : 25 Jul 2021
# Updated date : 03 Oct 2021
# Version      : 0.0.2
# Remark       : Update logging and unit test
# -------------------------------------------------------------------

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

import itertools
import logging
import logging.config
import re

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(module)s %(funcName)s: %(message)s'

logging.basicConfig(
    filename='app.log', 
    filemode='a', 
    format=log_format,
    level=logging.DEBUG
    )

def calculate_age(birthdate):
    '''
    Calculate age from birth date

    :param datetime birth_date: The birth date.
    :return int age: The ages.
    '''

    if (type(birthdate) is not date):

        error_message = f"The input type isn't datetime.date: {type(birthdate)}"
        
        logging.error(error_message)
        raise TypeError(error_message)

    today = date.today()

    if (birthdate.year > 2400):
        convert_be_to_fe = 543

        #* use relativedelta with get more accurate from timedelta
        birthdate = birthdate - relativedelta(years=convert_be_to_fe)

    if birthdate > today:

        error_message = f"Birth date can't be in the future: {birthdate}"

        logging.error(error_message)
        raise ValueError(error_message)

    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    logging.info(f"The birthday is {birthdate} then age is {age} years")

    return age

def mapping(item, criteria):
    '''
    Mapping item with the criteria

    :param string item: The item that would like to map with the criteria
    :param list criteria: The list of mapping criteria
    :return object criteria_result: The mapping result from item and criteria
    '''
    if item in criteria.keys():
      return criteria[item]
    else:
      logging.error(f"Can't mapping data: {item} with criteria: {criteria}.")
      return -1

def dict_sort(dict, ascending=True):
    '''
    sorting the dictionary by value

    :param dict dict: The dictionary.
    :param bool ascending: Is it sort by ascending
    :return dict dict_sort: The dictionary with sorted
    '''
    if type(dict) != dict:
        logging.error(f"The input {dict} is not dictionary.")
    dict_sort = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=(not ascending))} 
    return dict_sort

# -------------------------------------------------------------------
# description: slide the dictionary from index 0 to n (n = specify number)
# arguments  : dict_item(dict) -> dictionary item
#            : number(int) -> number of item
# return     : dict(dict) -> dict that has item from index 0 to n
# -------------------------------------------------------------------
def dict_slide(dict_items, number):
    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
    if type(dict_items) != dict:
        logging.error(f"The input {dict_items} is not dictionary.")
    return dict(itertools.islice(dict_items.items(), number))
    
# -------------------------------------------------------------------
# description: remove special character from text
# arguments  : text(string) -> text that want to remove special character
# return     : text(string) -> text that already remove special character
# -------------------------------------------------------------------
def cleansing_text(text):

    if(type(text) is not str):
        error_message = f"This input isn't string: {type(text)}"
        logging.error(error_message)
        raise TypeError(error_message)

    text = text.strip().lower()

    # Remvoe special character, number, space, dot
    text = re.sub('[\t\n\xa0\"\'!?\/\(\)%\:\=\-\+\*\_ๆ#$&,<>]', '', text)
    text = re.sub('[0-9]', ' ', text)
    text = re.sub('[\.]', ' ', text)
    text = re.sub('\u200b',' ', text)
    text = re.sub('\s+',' ',text)
    text = text.strip()

    return text