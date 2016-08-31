import os
import re
from HTMLParser import HTMLParser

import sys, getopt
import codecs

def strip_non_ascii(text):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in text if 0 < ord(c) < 127)
    return ''.join(stripped)

def unescape_text(text):
    html_parser = HTMLParser()
    unescaped = html_parser.unescape(text)
    
    return unescaped


def clean_text(text):
    URLREGEX = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
    HASHREGEX = r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)'
    MENTIONREGEX = r'(?:@[\w_]+)'

    cleaned = re.sub(URLREGEX, '', text)
    cleaned = re.sub(HASHREGEX, '', cleaned)
    cleaned == re.sub(MENTIONREGEX, '', cleaned)
    
    return cleaned


def preprocess(tweet):
    tweet = strip_non_ascii(tweet)

    text = tweet.decode("utf-8", "ignore").encode("ascii", "ignore")    
    text = text.rstrip()
    text = unescape_text(text)
    text = clean_text(text)
    return text


def preprocess_file(input, output):
    out = codecs.open(output, "w", "utf-8")
    
    with open(input) as tweets:
        for tweet in tweets:
            text = tweet.decode("utf-8", "ignore").encode("ascii", "ignore")
            text = text.rstrip()
            text = unescape_text(text)
            text = clean_text(text)
        
            out.write("%s\n" % text)
    
    out.close()
     

def main(argv):
    input = ''
    output = ''

    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["input=", "output="])
    except getopt.GetoptError:
        print 'twitter_preprocess.py -i <input> -o <output>'
        sys.exit(2)
        
    for opt,arg in opts:
        if opt == '-h':
            print 'pan.py -i <input> -o <output>'
            sys.exit()
        elif opt in ('-i', '--input'):
            input = arg
        elif opt in ('-o', '--output'):
            output = arg
            
    print "Input: ", input
    print "Output: ", output
    
    preprocess_file(input, output)

    
if __name__=='__main__':
    main(sys.argv[1:])
