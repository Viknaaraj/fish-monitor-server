import firebase_admin
from firebase_admin import credentials, db

# Replace with your actual Firebase Realtime Database URL
db_url = 'https://YOUR-PROJECT.firebaseio.com/'

try:
    cred = credentials.Certificate('firebase-key.json')
    firebase_admin.initialize_app(cred, {'databaseURL': db_url})
    # Attempt a basic read operation to force authentication
    db.reference('/').get()
    print("Success: Key is valid and authenticated.")
except Exception as e:
    print(f"Failed: {e}")
