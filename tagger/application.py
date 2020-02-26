# Imports
from .classes.new_email_functions import IMap
from .classes.basilica_functions import BasilicaAPI
from db import DB, User

from flask import Flask, request, jsonify
import json
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
from decouple import config
import joblib
from dotenv import load_dotenv
from io import BytesIO
from sqlalchemy import exists

def create_app():
    APP = Flask(__name__)
    APP.config["SQLALCHEMY_DATABASE_URI"] = config("DATABASE_URL")
    APP.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    DB.init_app(APP)
    # This lines can be enabled for testing locally but in production will erase the database every time you start up flask.
    # We opt to use the click package to make a cli that will allow us to run this once.
    # with APP.app_context():
    #     DB.drop_all()
    #     DB.create_all()
    basilica_client = BasilicaAPI()


    # This lines can be enabled for testing locally but in production will erase the database if the web or anyone has access to /reset.
    @APP.route('/resetvbvnpjtzmndypmrosuuvqxqugtaburqm')
    def reset():
        DB.drop_all()
        DB.create_all()
        return "reset database"
    
    @APP.route('/train_model', methods=["POST"])
    def train_model():
        # Get JSON and convert to DataFrame
        data = b""
        while True:
            chunk = request.stream.read(4096)
            if len(chunk) == 0:
                break
            data += chunk
        try:
            j = json.loads(data.decode("utf-8"))
        except Exception as e:
            return "Could not process input stream."
        df = pd.DataFrame(data=j["emails"])

        # Embed emails
        basilica_client.df = df
        df = basilica_client.embed_basilica_to_df()

        # Train model
        X = np.vstack(np.array(df['embedded']))
        y = df["id_email"].values
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X, y)

        # Save/compress model's pickle into the database
        file_obj = BytesIO() 
        joblib.dump(knn, file_obj, compress=3)
        file_obj.seek(0)
        pkl = file_obj.getvalue()

        # Check if user already exists
        db_user = User.query.filter(User.email_address == j["address"]).scalar()
        if db_user: 
            # Update pickle if user exists
            db_user.pickle_file = pkl
        else:
            # Make new user if one doesn't exist
            db_user = User(email_address=j["address"], pickle_file=pkl)
            DB.session.add(db_user)
        DB.session.commit()
        file_obj.close()
        return "Trained a model!"


    return APP
