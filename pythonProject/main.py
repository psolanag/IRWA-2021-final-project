import preprocess
import sys
docs_path = "/Users/pausolanagimeno/Downloads/dataset_tweets_WHO.txt"


infile = sys.argv[1]
outfile = sys.argv[2]
if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    tweets = preprocess.preprocessing(infile)
    f = open(outfile,'w')
    f.write(str(tweets))
    f.close()


