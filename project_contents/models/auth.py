from helpers.constants import (
    ALGORITHM,
    SECRET_KEY,
    DATABASE_NAME,
)
from jose import jwt
from models.mongo_helper import MongoHelper


class Auth:
    def get_current_user_from_token(self, token: str):
        db = MongoHelper()

        # Access the Authorization header
        # Process the authorization header as needed

        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

            return db.get_document_in_collection_by_unique_value(
                collection_name="users",
                database_name=DATABASE_NAME,
                param={"username": payload.get("sub")},
            )
        except:
            print("Failed to get user in DB")
            return None
