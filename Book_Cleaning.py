import re
import numpy as np
import pandas as pd
import pickle
import nltk
from collections import Counter
import os


#Runs through a text and replaces all fancy and single quotes with straight double quotes
def quotinator(raw_text):
    
    #Replace fancy quotes
    raw_text = raw_text.replace('“', '"')
    raw_text = raw_text.replace('”', '"')
    
    #Replace fancy single quotes with normal single quotes
    raw_text = raw_text.replace("‘", "'")
    raw_text = raw_text.replace("’", "'")
    
    #Replace single quotes at the start of quotation
    raw_text = re.sub("[\s]'", ' "', raw_text)
    
    #Replace single quotes at the end of quotation
    raw_text = re.sub("'[\s]", '" ', raw_text)
    
    #Replace other types of quotes
    raw_text = re.sub("\'\'", '"', raw_text)
    raw_text = re.sub("\`\`", '"', raw_text)    
    
    return raw_text    


#Convert a string into a nltk Text object, its word tokens, and its sentence tokens
def TextConvert(raw):
    
    #Convert to tokens, then change back into text
    tokens = nltk.word_tokenize(raw)
    sents = nltk.sent_tokenize(raw)
    text = nltk.Text(tokens)
    
    return (text, tokens, sents)




#Extract the title of a book from a GP book
def BookTitle(path):

    #Open book file and save it as a raw string
    f = open(path, 'r') 
    raw = f.read()
    
    #Get the title line and convert it to tokens
    title_line = re.findall(r'Title:.*\w*.*\n', raw)[0]
    tokes = nltk.word_tokenize(title_line)
    
    #Gets rid of 'Title' label and rejoins the real title
    tokes = tokes[2:]
    title = ' '.join(tokes)
    
    return title
    
     


#Extract the author of a book from a GP book
def BookAuthor(path):

    #Open book file and save it as a raw string
    f = open(path, 'r') 
    raw = f.read()
    
    #Get the author line and convert it to tokens
    author_line = re.findall(r'Author:.*\w*.*\n', raw)[0]
    tokes = nltk.word_tokenize(author_line)
    
    #Gets rid of 'Author' label and rejoins the real title
    tokes = tokes[2:]
    author = ' '.join(tokes)
    
    
    return author
    
    
    
#Strip off the PG header and footer
def TextStrip(path, author):

    #Open book file and save it as a raw string
    f = open(path, 'r') 
    raw = f.read()
    
    #Strip the Header
    text_list = re.split('[Bb][Yy] '+author, raw)
    
    #Rejoins text
    text = ' '.join(text_list[1:])
    
    #Get rid of Chapter labels
    chapters = re.split('[Cc][Hh][Aa][Pp][Tt][Ee][Rr].+\n', text)
    text = ' '.join(chapters)
    
    return text
    



#Take a text and turn it into a data set of excerpts 
def ExcerptDf(tokens, author, title, num):

    #Create an empty DataFrame
    df = pd.DataFrame([], columns = ['Excerpt', 'Author', 'Title'])
    
    ind = 0
    #Loop through to create the dataset
    for i in range(int(len(tokens)/num)):
        exc = ' '.join(tokens[num*i:num*(i+1)])
        df.loc[ind] = [exc, author, title]
        ind += 1
        
        
    return df
        
    
#Strip punctuation from a string of text
def PunctuationStrip(text):

    #Perform a search for everything that's not specified punctuation
    text_list = re.findall('\w+', text)
    tex =  ' '.join(text_list)
    tex = tex.replace(' s ', '')
    
    return tex
    

#Count the number of certain types of punctuation
def OddPuncCount(text):
    
    #Create a dictionary with the punctuation we want
    punc_dict = {'?': 0, '!': 0, ';': 0, ':' : 0, '--': 0, ',': 0, '"': 0, '(': 0}
    
    #Count the number of occurances of each in the text and add it to the dictionary
    punc_dict['?'] = len(re.findall('\?', text))
    punc_dict['!'] = len(re.findall('\!', text))
    punc_dict[';'] = len(re.findall('\;', text))
    punc_dict[':'] = len(re.findall('\:', text))
    punc_dict['--'] = len(re.findall('\-\-', text))
    punc_dict[','] = len(re.findall('\,', text))
    punc_dict['('] = len(re.findall('\(', text))
    punc_dict['"'] += int((len(re.findall('\`\`', text)) + len(re.findall("\'\'", text)) / 2))
        
    return punc_dict

#Adds columns to dataframe based on odd punctuation usage
def PuncCols(df, exc_col):
    
    #Do not loop this - it causes problems later with naming
    df['Q_Count'] = df[exc_col].apply(lambda x: OddPuncCount(x)['?'])
    df['E_Count'] = df[exc_col].apply(lambda x: OddPuncCount(x)['!'])
    df['SC_Count'] = df[exc_col].apply(lambda x: OddPuncCount(x)[';'])
    df['C_Count'] = df[exc_col].apply(lambda x: OddPuncCount(x)[':'])
    df['D_Count'] = df[exc_col].apply(lambda x: OddPuncCount(x)['--'])
    df['Co_Count'] = df[exc_col].apply(lambda x: OddPuncCount(x)[','])
    df['Qu_Count'] = df[exc_col].apply(lambda x: OddPuncCount(x)['"'])
    df['Pa_Count'] = df[exc_col].apply(lambda x: OddPuncCount(x)['('])
        
    return df


#Counts part of Speech and adds the new columns to a DataFrame
def SpeechCols(df, tokens):

    all_speech = df[tokens].apply(lambda x: nltk.pos_tag(x))
    count_speech = all_speech.apply(lambda x: Counter([w[1] for w in x]))
    
    #List of parts of speech
    parts = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']
    
    #Adds a column for each part of speech and reads off the Counter column
    for pos in parts:
        df[pos] = count_speech.apply(lambda x: x.get(pos, 0))
        
    return df

#Creates feature columns for text, gets rid of text-based columns, then outputs a dataframe
def TextToDf(path, num):

    #Get author, title, and raw text
    author = BookAuthor(path)
    title = BookTitle(path)
    raw_text = TextStrip(path, author)
    
    #Changes all quotes to straight quotes
    raw_text = quotinator(raw_text)
    
    #Get text, tokens, and sentences
    text, tokens, sents = TextConvert(raw_text)
    print(len(tokens))
    
    #Create the initial DataFrame
    df = ExcerptDf(tokens, author, title, num)
    
    #Create punctuation columns
    df = PuncCols(df, 'Excerpt')
    
    #Create a sentence counting column
    df['Sent_Count'] = df.Excerpt.apply(lambda x: len(nltk.sent_tokenize(x)))
    df['Hyphen_Count'] = df.Excerpt.apply(lambda x: len(re.findall('\w+\-\w+', x)))
    
    #Create an excerpt column without punctuation
    df['Word_Excerpt'] = df.Excerpt.apply(lambda x: PunctuationStrip(x))
    
    #Create a column of tokens based on new excerpt column
    df['Tokens'] = df.Word_Excerpt.apply(lambda x: nltk.word_tokenize(x))
    df['F_Tokens'] = df.Excerpt.apply(lambda x: nltk.word_tokenize(x))
    
    #Create columns for word number, unique word number, and average unique word length
    df['Word_Num'] = df.Tokens.apply(lambda x: len(x))
    df['Unique_Words'] = df.Tokens.apply(lambda x: len(set(x)))
    df['Word_Length_Av'] = df.Tokens.apply(lambda x: np.array([len(w) for w in set(x) if len(w) > 2]).mean())
    
    #Create the parts of speech columns
    df = SpeechCols(df, 'F_Tokens')
    
    #Drop columns with text information
    df.drop(columns = ['Tokens', 'Word_Excerpt', 'Excerpt', 'F_Tokens'], inplace = True)    

    return df

#Creates a DataFrame from all of the books in the Texts folder
def BookDf(path, num):
    
    #Get current path
    path = os.getcwd() + '/' + path
    
    #Create an empty DataFrame
    df = pd.DataFrame()
    
    #Run through all books and add them to the premade DataFrame
    for root, dirs, files in os.walk(path):
        for name in files:
            if '.txt' in name:
                print(name)
                book_df = TextToDf(os.path.join(root, name), num)
                df = pd.concat([df, book_df])
                
    return df

#Create token sizes for training sets
numbered_list = [600, 800, 1200, 1500, 1800, 2000, 3000, 4000, 8000, 10000]

#Run through each token size, record text statistics, then export to csv
for number in numbered_list:
    df = BookDf('Clean_Texts', number)
    df.to_csv(str(number)+'_Train_Set.csv', index = False)


#Create token sizes for test sets
numbered_list = [600, 800, 1200, 1500, 1800, 2000, 3000, 4000, 8000, 10000]

#Run through each token size, record text statistics, then export to csv
for number in numbered_list:
    df = BookDf('Extra_Clean_Text', number)
    df.to_csv(str(number)+'_Test_Set.csv', index = False)

    
















    
    
    
    
    
