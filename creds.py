import tweepy
import sys
sys.path.insert(0, "..")
from credentials import Credentials

class twitterUser():
	def __init__(self):
		cred = Credentials()
		consumer_key = cred.consumer_key
		consumer_secret = cred.consumer_secret
		access_token = cred.access_token
		access_token_secret = cred.access_token_secret

		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)
		self.api = tweepy.API(auth)
		self.user = self.api.me()
		print(self.user.name)


	def tweet(self, text):
		self.api.update_status(text)
		return


	def followFollowers(self):
		for follower in tweepy.Cursor(self.api.followers).items():
			try:
				follower.follow()
			except:
				0
		return
