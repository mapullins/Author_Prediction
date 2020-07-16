# Author_Prediction
This is a personal project to see if the author of a text can be predicted using only statistics about a passage of text. Included are the data sets created, the python code used to create the data sets, and the models created to predict the author, along with analysis.

## General Information
All text were taken from Project Gutenberg (http://www.gutenberg.org/).

## Technologies
* Python 3.8 (NumPy, Pandas, NLTK, and Re required)

## Use of Book_Cleaning.py
Book_Cleaning.py does not have code to remove the Project Gutenberg headers and footers, so this must be added or done by hand. If you do want to write code for it, please be aware that while the footers are (relatively) consistent, the headers are extremely diversive in both content and formatting.

To change by hand, erase everything after the end of the book to eliminate the footer. For the header, keep the 'Title:' and 'Author:' parts, then erase everything except the book title where it appears later and 'by (author name)' where 'author name' is written identically to how it is written in 'Author:'. I also took out the table of contents and any text that is not part of the story of the novel.
