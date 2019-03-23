import tweepy
import sys
sys.path.insert(0, "..")
from credentials import Credentials

cred = Credentials()

consumer_key = cred.consumer_key
consumer_secret = cred.consumer_secret
access_token = cred.access_token
access_token_secret = cred.access_token_secret

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

user = api.me()
print (user.name)
"""
tweet = "hello world"
api.update_status(tweet)
"""