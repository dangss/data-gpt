from pymongo.mongo_client import MongoClient
from bson.json_util import dumps, loads
import json
import pytz
import datetime
import time
from helpers.constants import MONGODB_URL

utc_time = datetime.datetime.now(pytz.utc)
utc_plus_7 = utc_time.astimezone(pytz.timezone("Asia/Bangkok"))


class MongoHelper:
    def __init__(self):
        self.uri = MONGODB_URL
        self.client = MongoClient(self.uri)

    def connect(self, uri):
        self.uri = uri
        self.client = MongoClient(uri)

        try:
            self.client.admin.command("ping")

        except Exception as e:
            print(e)

    def get_documents_in_collection(self, collection_name: str, database_name: str):
        """
        Get all documents in one collection

        Args:
            database_name: Name of your database
            collection_name: Name of your collection

        Returns:
            list: With all documents in collection
        """

        return loads(
            dumps(
                list(MongoClient(MONGODB_URL)[database_name][collection_name].find({}))
            )
        )

    def create_document_in_collection(
        self, collection_name: str, database_name: str, param: dict
    ):
        """
        Insert one document in one collection

        Args:
            database_name: Name of your database
            collection_name: Name of your collection
            param: include property in one document

        Returns:

        """
        return MongoClient(MONGODB_URL)[database_name][collection_name].insert(
            {
                **param,
                "_id": str(round(time.time() * 1000)),
                "created_date": utc_plus_7,
                "updated_date": utc_plus_7,
            }
        )

    def delete_document_in_collection(
        self, database_name: str, collection_name: str, id: str
    ):
        """
        Delete one document in one collection by id

        Args:
            database_name: Name of your database
            collection_name: Name of your collection
            id: the ID of document

        Returns:
            int: number of document has been deleted
        """

        delete_result = MongoClient(MONGODB_URL)[database_name][
            collection_name
        ].delete_one({"_id": id})

        deleted_count = delete_result.deleted_count
        serialized_deleted_count = json.dumps(deleted_count)
        return serialized_deleted_count

    def update_document_in_collection(
        self, database_name: str, collection_name: str, param: dict
    ):
        """
        Update one document in one collection by id

        Args:
            database_name: Name of your database.
            collection_name: Name of your collection.
            param (dict):
                - Must be have id.
                - Include all property of document.

        Returns:
            int: number of document has been updated.
        """

        filter = {"_id": param["_id"]}
        param.pop("_id")

        update_operation = {"$set": {**param, "updated_date": utc_plus_7}}

        # Perform the update operation using update_one()
        result = MongoClient(MONGODB_URL)[database_name][collection_name].update_one(
            filter, update_operation
        )
        return json.dumps(result.modified_count)

    def get_document_in_collection_by_id(
        self, database_name: str, collection_name: str, id: str
    ):
        """
        Get one document in one collection by id

        Args:
            database_name: Name of your database.
            collection_name: Name of your collection.
            param (dict):
                - Must be have id.
                - Include all property of document.

        Returns:
            dict: the document in collection.

        Raises:
            None: If the document in collection is not found.
        """

        try:
            return loads(
                dumps(
                    list(
                        MongoClient(MONGODB_URL)[database_name][collection_name].find(
                            {"_id": id}
                        )
                    )
                )
            )[0]
        except:
            return None

    def get_document_in_collection_by_unique_value(
        self, database_name: str, collection_name: str, param: dict
    ):
        """
        Get one document in one collection by unique value

        Args:
            database_name: Name of your database.
            collection_name: Name of your collection.
            param (dict):
                - Must be have the value unique.
                - Include all property of document.

        Returns:
            dict: the document in collection.

        Raises:
            None: If the document in collection is not found.
        """

        try:
            return MongoClient(MONGODB_URL)[database_name][collection_name].find_one(
                param
            )

        except:
            return None

    def get_documents_by_property(
        self, database_name: str, collection_name: str, params: dict
    ):
        """
        Get list document in one collection by properties

        Args:
            database_name: Name of your database.
            collection_name: Name of your collection.
            param (dict):
                - Include properties of document that you want to find.

        Returns:
            list: list documents in collection.

        Raises:
            None: If don't have any documents with properties you gave
        """

        try:
            return loads(
                dumps(
                    list(
                        MongoClient(MONGODB_URL)[database_name][collection_name].find(
                            params
                        )
                    )
                )
            )

        except:
            return None

    def update_documents_in_collection_by_property(
        self, database_name: str, collection_name: str, filter: dict, params: dict
    ):
        # filter = {"_id": params["_id"]}

        # params.pop("_id")

        update_operation = {"$set": {**params, "updated_date": utc_plus_7}}

        # Perform the update operation using update_one()
        result = MongoClient(MONGODB_URL)[database_name][collection_name].update_many(
            filter, update_operation
        )

        return json.dumps(result.modified_count)

    def create_collection_in_database(self, database_name: str, collection_name: str):
        collist = MongoClient(MONGODB_URL)[database_name].list_collection_names()
        if "collection_name" in collist:
            print("The collection exists.")
            return None

        return MongoClient(MONGODB_URL)[database_name].create_collection(
            name=collection_name
        )

    def get_list_collection_in_database(self, database_name: str):
        return MongoClient(MONGODB_URL)[database_name].list_collection_names()
