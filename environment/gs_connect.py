import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from df2gspread import df2gspread as d2g

scope = ['https://spreadsheets.google.com/feeds']
# adjust the location and the name of the json keyfile accordingly
credentials = ServiceAccountCredentials.from_json_keyfile_name('/tf/notebooks/environment/W266-2-c0ee0ca856ea.json', scope)
gc = gspread.authorize(credentials)

# first, make sure to share the spreadsheet with service account email
# located in the json file
# this key is coming from the spreadsheet url
spreadsheet_key = '1G4J6uUNVAqjjNz1eza8FTSbAlVjsfS4XIY7voyOtkso'

def get_from_gs(wks_name):
    """
    Gets data from a Google Spreadsheet and returns
    it as a pandas dataframe
    """
    book = gc.open_by_key(spreadsheet_key)
    # enter the name of the spreadsheet to connect to
    worksheet = book.worksheet(wks_name)
    table = worksheet.get_all_values()
    # convert table data into a dataframe and return it
    return pd.DataFrame(table[1:], columns=table[0])


def send_to_gs(df, wks_name):
    """
    Send pandas dataframe to Google Spreadshseet
    """

    d2g.upload(df,
        spreadsheet_key,
        wks_name,
        credentials=credentials,
        row_names=True)

