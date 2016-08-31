from twitter import Twitter, OAuth, TwitterHTTPError

# Set token values
ACCESS_TOKEN = '586338090-qYKE4AWln63B0nLI8WnNOXwaDZtNgIS1cZJFuq5N'
ACCESS_SECRET = 'GFpelZCqypWWEq4chgLST7SgoiDo3SxnOUacSGHPzlqDL'
CONSUMER_KEY = 'LoCP15NXjXPzGwJtRCDDh9GJ1'
CONSUMER_SECRET = 'bTeFv1UCdKgYwAS2nvs6XRaC0NV2zWDXRYC4r94jd40JpmAjIx'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
twitter = Twitter(auth=oauth)

def postTweet(text):
    twitter.statuses.update(status = text)
