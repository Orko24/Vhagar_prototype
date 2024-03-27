from sqlalchemy import create_engine
import pandas as pd

# engine = create_engine('postgresql://username:password@localhost:5432/mydatabase')
# df.to_sql('table_name', engine)

'''
https://medium.com/@romina.elena.mendez/sql-to-python-pandas-a-sql-users-quick-guide-2da84cb4ad9e
'''

class postgress_table_add(object):

    def __init__(self, csv, table_name = "table_name", username = "postgres", database = "postgres", password = "pass"):

        '''

        :param csv: csv to be converted into a sql table
        :param table_name: table name of the sql table
        :param username: postgres username
        :param database: postgres database name
        :param password: postgres password
        '''

        self.csv = csv
        self.table_name = table_name
        self.username = username
        self.database = database
        self.password = password


        # engine that connects to postgres server
        self.engine = create_engine('postgresql://{}:{}@localhost:5432/{}'.format(self.username, self.password, self.database))


    def dataframe_to_sql(self):

        # converts csv into pandas dataframe
        df = pd.read_csv(self.csv)

        # converts pandas dataframe to sql table
        df.to_sql('{}'.format(self.table_name), self.engine)
        return


class csv_to_sql(object):

    def __init__(self, csv_path, table_name = 'table_name'):

        self.csv_path = csv_path
        self.table_name = table_name

    def postgress_data_creation(self):

        '''
        rapid execution build in mind
        :return:
        '''

        table_add = postgress_table_add(csv = self.csv_path, table_name= self.table_name)

        table_add.dataframe_to_sql()

        return table_add.table_name

class dataframe_to_csv(object):

    def __init__(self, dataframe, path):

        self.dataframe = dataframe
        self.path = path

    def dataframe_csv(self):

        csv = pd.DataFrame.to_csv(path_or_buf=self.path)

        return self.path

class postgress_table_add_dataframe(object):

    def __init__(self, dataframe, table_name = "table_name", username = "postgres", database = "postgres", password = "pass"):

        '''

        :param csv: csv to be converted into a sql table
        :param table_name: table name of the sql table
        :param username: postgres username
        :param database: postgres database name
        :param password: postgres password
        '''

        self.dataframe = dataframe
        self.table_name = table_name
        self.username = username
        self.database = database
        self.password = password


        # engine that connects to postgres server
        self.engine = create_engine('postgresql://{}:{}@localhost:5432/{}'.format(self.username, self.password, self.database))


    def dataframe_to_sql(self):

        # converts csv into pandas dataframe
        df = self.dataframe

        # converts pandas dataframe to sql table
        df.to_sql('{}'.format(self.table_name), self.engine)
        return






