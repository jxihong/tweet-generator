from twitter import Twitter, OAuth, TwitterHTTPError

# Set token values
ACCESS_TOKEN = '586338090-qYKE4AWln63B0nLI8WnNOXwaDZtNgIS1cZJFuq5N'
ACCESS_SECRET = 'GFpelZCqypWWEq4chgLST7SgoiDo3SxnOUacSGHPzlqDL'
CONSUMER_KEY = 'LoCP15NXjXPzGwJtRCDDh9GJ1'
CONSUMER_SECRET = 'bTeFv1UCdKgYwAS2nvs6XRaC0NV2zWDXRYC4r94jd40JpmAjIx'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
twitter = Twitter(auth=oauth)
username = 'dennisnliao'
numTweets = 20

profile = twitter.users.lookup(screen_name = username)
numTweets = profile[0]['statuses_count']

tweets = []
i = 0
interval = 100
id = 537408199532347393
while(i < numTweets):
    tweets += twitter.statuses.user_timeline(screen_name = username, count = interval, max_id = id)
    i += interval
    currentLen = len(tweets)
    id = tweets[currentLen - 1]['id']
print(len(tweets))

with open(username + '_tweets.csv', 'w') as f:
    for tweet in tweets:
        try:
            f.write(tweet['created_at'])
            f.write(',')
            f.write(tweet['text'])
            f.write('\n')
        except:
            print('FAILED TO WRITE TWEET TO FILE:')
            print(tweet['created_at'])
            print(tweet['text'])
        finally:
            pass
