import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pandas.io.json import json_normalize
pd.set_option("display.max_columns", 0)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from flask import Flask
import json
from psycopg2.extras import Json
import pickle
import os

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
DATABASE_URL = os.getenv("DATABASE_URL")

APP = Flask(__name__)
APP.config['SQLALCHEMY_DATABASE_URI'] = TEST_DATABASE_URL
APP.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

APP.config['extend_existing'] = True
DB = SQLAlchemy(APP)

df = pd.read_csv('somecsv.csv', header='???')

class Example(DB.Model):
    '''
        CookieCutter DB.Model class object
        Instantiate columns to be committed to a table here

        Notes:
        each variable should instantiate a DB.Column() for each column in the dataset.
        i.e. in a dataset with 10 columns there should be 10 variables in this class.
        The first argument in DB.Column() should be the respected columns dtype
        i.e. a column of integers will be DB.Column(DB.Integer),
        a column of strings will be DB.Column(DB.String(max(len(string)))),
        and so on...
    '''
    id = DB.Column(DB.Integer, primary_key=True)
    state = DB.Column(DB.String(2), nullable=False)
    year = DB.Column(DB.Integer, nullable=False)
    colorhex = DB.Column(DB.String(6), nullable=False)
    growth_rate = DB.Column(DB.Float, nullable=False)
    price_per_apple = DB.Column(DB.Float, nullable=False)

DB.create_all()

target = df.shape[0]
print(target)
done = 0
step = 200
# while loop to iterate through all rows of the dataframe
while done < target:
    todo = min(target - done, step)
    # [done:done+todo] is a slice of the dataframe
    for row in np.ascontiguousarray(df[done:done+todo].values):
        EXAMPLE = Example(
                state=row[0],
                year=row[1],
                colorhex=row[2],
                growth_rate=row[3],
                price_per_apple=row[4]
                )

        DB.session.add(EXAMPLE)
    DB.session.commit()
    done += todo
    print(done,end=' ')

print('Table Created with', done, 'additions.')

