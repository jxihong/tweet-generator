from twitter import Twitter, OAuth, TwitterHTTPError
from twitter_preprocess import preprocess

# Set token values
ACCESS_TOKEN = '586338090-qYKE4AWln63B0nLI8WnNOXwaDZtNgIS1cZJFuq5N'
ACCESS_SECRET = 'GFpelZCqypWWEq4chgLST7SgoiDo3SxnOUacSGHPzlqDL'
CONSUMER_KEY = 'LoCP15NXjXPzGwJtRCDDh9GJ1'
CONSUMER_SECRET = 'bTeFv1UCdKgYwAS2nvs6XRaC0NV2zWDXRYC4r94jd40JpmAjIx'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
twitter = Twitter(auth=oauth)
username = 'realDonaldTrump'

profile = twitter.users.lookup(screen_name = username)
numTweets = min(3200, profile[0]['statuses_count'])

tweets = twitter.statuses.user_tmeline(screen_name = username, count = 1)
i = 0
interval = 200
id = tweets[0]['id']

while(i < numTweets):
    myCount = min(interval, numTweets - i)
    tweets += twitter.statuses.user_timeline(screen_name = username, count = myCount, max_id = id)
    i += interval
    id = tweets[myCount - 1]['id'] - 1
    with open(username + '_tweets.csv', 'w') as f:
        for tweet in tweets:
            text = tweet['text']
            text = preprocess(text)
            try:
                f.write(text)
                f.write('\n')
            except:
                print('FAILED TO WRITE TWEET TO FILE:')
                print(tweet['text'])
            finally:
                pass
