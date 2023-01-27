# pip install xlrd

import pandas as pd
from sqlalchemy import create_engine, select, MetaData, Table, and_, Column
import mysql.connector

"""
This is a complementary file that performs the following
1. Data Loading from different sources
"""


class DataLoading:
    def __init__(self, file_path, file_name):
        self.file_path = file_path
        self.file_name = file_name

    #     from a csv file
    def csv(self):
        return pd.read_csv(self.file_path + "\\" + self.file_name)

    #     from an excel file
    def excel(self, sheet_name):
        return pd.read_excel(self.file_path + "\\" + self.file_name, sheet_name)


#     from a mysql database
"""
For data loading from MySQL database you will need the
1. username, password, host, port and the database,
2. the query from the user
"""

class DlMysql:
    def __init__(self, user, password, host, port, database):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database

    def mysql(self, user_query):

        try:
            connection = mysql.connector.connect(host=self.host,
                                                 database=self.database,
                                                 user=self.user,
                                                 port=self.port,
                                                 password=self.password)

            cursor = connection.cursor()
            cursor.execute(user_query)
            result = cursor.fetchall()
            #             print(result)
            result = pd.DataFrame(result)
            print(result)
            print("User Query successfully ")

        except mysql.connector.Error as error:
            print("User Query Failed in MySQL: {}".format(error))
        finally:
            if connection.is_connected():
                cursor.close()
                connection.close()
                print("MySQL connection is closed")


"""
The inspection class is used to have an overview of the dataset that was loaded.
to inspect the whole
"""
class Inspect:
    def __init__(self, data):
        self.data = data

    def ins(self, method):
        if method == "n":
            print(data.describe())
        elif method == "y":
            print(data.describe(include="all"))
            print("________________________________________________")
            print(data.info())
        else:
            print("Invalid method")


class PrePro:
    def __init__(self, data):
        self.data = data

    def missing(self, method):
        if method == "dv":
            self.data.dropna(axis=1, inplace=True)
        elif method == "de":
            self.data.dropna(axis=0, inplace=True)
        elif method == "me":
            meann = self.data.mean
            self.data.fillna(meann)
