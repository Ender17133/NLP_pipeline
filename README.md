[![Open in Jupyter](https://img.shields.io/badge/Open%20in-Jupyter-blue.svg?logo=jupyter)](https://github.com/Ender17133/NLP_pipeline/blob/main/NLP.ipynb)

This assignment will focus primarily on the design, creation, and deployment of `functions`. This assignment will require to create a series of functions together will be used as a crude NLP pipeline that cleans text data, structures the data for analysis, then calculates the `tf-idf` scores for a series of documents. The tf-idf metric is used to calculate the importance of a word to a document. Specifically, it measures the importance of a word in a document while discounting its value if it is used very frequently in all documents. For example the word "the" is probably used a lot in a given document, but it is also used a lot in most documents, so should not be considered important.

The functions should be created for in `both R and Python`.

### Background Vocabulary
- **Document**: A single instance of text
- **Corpus**: A collection of documents
- **Tokenize**: Structure a string into a list of words. Each item is a single token
- **TF-IDF Score**: A formula that tries to measure how important a given word is to a document. This helps in understanding what a document may be about.


### **TF-IDF**
#### Term Frequency: Calculated at the document level
$$
\begin{align}
TF = \frac{Term\ Frequency}{Total\ Number\ of\ Words\ in\ Document}
\end{align}
$$
```
#-- Term Frequency Example
"The dog ate the cat"
Output [0]: {
  "The": 2/5,
  "Dog": 1/5,
  "Ate": 1/5,
  "Cat": 1/5
}
```
#### Inverse Document Frequency - Calculated at the corpus level
$$
\begin{align}
IDF = log(\frac{Number\ of\ Documents\ in\ a\ corpus}
          {Number\ of\ documents\ that\ contain\ the\ term}\ \ \ \  )
\end{align}
$$
```
#-- Inverse Document Frequency for 2 documents
[
 "The dog ate the cat",
 "The stars are bright"
]
Output [0]: {
  "The":   log(2 / 2),
  "Dog":   log(2 / 1),
  "Ate":   log(2 / 1),
  "Cat":   log(2 / 1),
  "stars": log(2 / 1),
  "are":   log(2 / 1),
  "bright":log(2 / 1),
}
```
#### TF * IDF
$$
\begin{align}
TFIDF = TF * IDF
\end{align}
$$
```
#-- TFIDF for Doc1 using corpus of Doc1 and Doc2
[
 "The dog ate the cat",
 "The stars are bright"
]
Output [0]: TFIDF for Doc 1 = {
  "The":   (2/5) * log(2 / 2),
  "Dog":   (1/5) * log(2 / 1),
  "Ate":   (1/5) * log(2 / 1),
  "Cat":   (1/5) * log(2 / 1),
  "stars": (0/5) * log(2 / 1),
  "are":   (0/5) * log(2 / 1),
  "bright":(0/5) * log(2 / 1)
}
```


Learn more about TF-IDF and the specifics of the formula [here](https://medium.com/analytics-vidhya/tf-idf-term-frequency-technique-easiest-explanation-for-text-classification-in-nlp-with-code-8ca3912e58c3).




## 1. Create the function clean_text (12.5 pts)
@clean_text(text: str)
  - standardizes text by upper casing, removing non-alphanumeric characters (leave spaces), and removing extra spacing within the text
  - Aguments: text: str representing a string of text data
  - Return: text: str representing the clean string of text data
  ```
  Example
  Input  --> "The dog, ate 2 cats. "
  Output --> "THE DOG ATE 2 CATS"
  ```



Python


```python
#-- Hint. regular expressions can be helpful for string manipulation
#------ The following functions may be useful

#1 Libraries
# re (regular expressions) module that is used for sub function
import re
'''
  #-- removes non-alpha numeric characters, except white space
  text = re.sub(r'[^A-Za-z0-9 ]+', '', text)

  #-- Replace all white space with single space
  text = " ".join(text.split())
'''
#2 Code
def clean_text(text:str):
  # function removes all characters from the text that are non-alpha numeric
  result = re.sub(r'[^A-Za-z0-9 ]+', '', text)
  # after that, using upper(), text casing is changed to upper casing
  result = result.upper()
   # as a last step, extra white spaces are eliminated with split() func. and then words are joined back together by white space with join()
  result = ' '.join(result.split())
  # function returns modified text as an output
  return result


#--- test function with example
#3 Testing
print(clean_text("This is a test    sentence!"))


```

    THIS IS A TEST SENTENCE
    

R


```python
#-- Hint. regular expressions can be helpful for string manipulation
#------ The following functions may be useful

#-- removes non-alpha numeric characters, except white space
# text = gsub('[^A-Za-z0-9 ]+', '', text)

#-- Replace all white space with single space
# str_squish(text) using the stringr library

#1 Libraries
# library that has useful str_squish() function, which will help us to replace all white space with single space
library(stringr)

#2 Code
clean_text = function(text) {
    # gsub eliminates all characters that are non-alpha numeric
    result_clean = gsub('[^A-Za-z0-9 ]+', '', text)
    # toupper() function is used to make upper casing of all words in the text
    result_clean = toupper(result_clean)
    # str_squish() function is then used to replace all white space with single space
    result_clean = str_squish(result_clean)
    # clean_text() function returns modified text as an output
    return(result_clean)
}



#--- test function with example
print(clean_text("This is a test    sentence!"))
```

    [1] "THIS IS A TEST SENTENCE"
    

## 2. Create the function tokenize (12.5 pts)
@tokenize(text: str)
  - Turns a string of text data into a list of words
  - Arguments: text str representing a string of text data
  - Returns: list of words
  ```
  Example
  Input  --> "THE DOG ATE 2 CATS"
  Output --> ["THE", "DOG", "ATE", "2", "CATS"]
  ```

Python


```python
#1. Code
def tokenize(text:str):
   # function divides words in the text into the elements of the list by using split()
  result_tokenize = text.split()
   # function then returns text variable result_tokenize as an output
  return result_tokenize



#--- test function with example
#2. Testing
tokenize("This is a test    sentence!")

```




    ['This', 'is', 'a', 'test', 'sentence!']



R


```python
#1 Code
tokenize = function(text) {
    # strsplit() function splits words by the space and makes a list
    result_tokenize = strsplit(text, ' ')[[1]]
    # function returns list of words
    return(result_tokenize)
}


#2 Test
#--- test function with example
tokenize("THIS IS A TEST SENTENCE")

```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>'THIS'</li><li>'IS'</li><li>'A'</li><li>'TEST'</li><li>'SENTENCE'</li></ol>



## 3. Create the function clean_corpus (12.5 pts)

@clean_corpus(list_of_text: list)
  - Takes a list of strings and calls clean_text() and tokenize() functions to each string in a list
  - Arguments: text_list list represents a list of text.
  - Returns: list of lists with string tokens
  ```
Example
  Input  --> ["The dog, ate 2 cats. ",
              "The stars- are so     bright!"]
  Output --> [["THE", "DOG", "ATE", "2", "CATS"],
              ["THE","STARS","ARE","SO","BRIGHT"]]
  ```

Python


```python
#1 Code
def clean_corpus(list_of_text:list):
  # function creates a blank clean_corpus list
  clean_corpus = []
  # for loop that iterates over element in the clean_corpus list
  for document in list_of_text:
    # function uses previously created clean_text() function and cleans elements from non-alpha numeric characters
    text = clean_text(document)
    # function then uses another created tokenize() function and makes a tokes list from each element
    tokens = tokenize(text)
     # function appends tokens list to the clean_corpus list to create a lists of lists
    clean_corpus.append(tokens)

  # function returns clean_corpus list
  return clean_corpus # function returns clean_corpus list


#--- test function with example
#2 Testing
test_corp = [
    "This is a test    sentence!",
    "here&^% is a 2nd example...."
]
print(clean_corpus(test_corp))

```

    [['THIS', 'IS', 'A', 'TEST', 'SENTENCE'], ['HERE', 'IS', 'A', '2ND', 'EXAMPLE']]
    

R


```python
#1 Code
clean_corpus = function(list_of_text) {
    # blank list called clean_corpus
    clean_corpus = list()
    # for loop that iterates over each element in the list
    for (document in list_of_text) {
        # clean_text() function cleans input from uneccessary white spaces and characters, assigns the result to variable text
        text = clean_text(document)
        # text is then tokenized using function tokenize()
        tokens = tokenize(text)
        # final result is then appended to clean_corpus blank list
        clean_corpus = c(clean_corpus, list(tokens))
    }
    # function returns clean_corpus list as a result
    return(clean_corpus)
}



#2 Test
#--- test function with example
doc1 <- "This is a test    sentence!"
doc2 <- "here&^% is a 2nd example...."
test_corp <- list(doc1, doc2)

corp <- clean_corpus(test_corp)
print(corp)
```

    [[1]]
    [1] "THIS"     "IS"       "A"        "TEST"     "SENTENCE"
    
    [[2]]
    [1] "HERE"    "IS"      "A"       "2ND"     "EXAMPLE"
    
    

## 4. Create the function term_frequency (12.5 pts)


@term_frequency(list_of_text: list)
  - Takes a list of list (string tokens) output from clean_corpus and computes a word count/total words in each document saved in a list of dictionaries. The dictionaries should have keys that relate to a token (e.g. word) and the value is the count. Note this should relate to the TF in TFIDF. Therefore it should be the count of word divided by the the total number of words in a document.
  - Arguments: list_of_text str representing a list of strings. This should be the output of text_to_tokens.
  - Returns: list of dictionaries featuring word counts
  ```
  Illustrative Examples - Values may not be correct
  Input  --> [["THE", "DOG", "ATE", "2", "CATS"],
              ["THE","STARS","ARE","SO","BRIGHT"]]
  Output --> [{"THE":0.2, "DOG":0.2, "ATE":0.2, "2":0.2, "CATS":0.2},
              {"THE":0.2,"STARS":0.2,"ARE":0.2,"SO":0.2,"BRIGHT":0.2}]
  ```



Python


```python
#1 Code
def term_frequency(list_of_text:list):
  # function makes a blank list that will have 2 dictionaries inside later
  dictionary_list = []
  # outer for loop iterates over each list in the list of lists
  for document1 in list_of_text:
    # blank dictionary for elements in each list
    elements_dictionary = {}
     # function then further iterates over each element of the list in the inner for loop
    for element in document1:
       # total length variable that computes total number of words in the list
        total_length = len(document1)
         # word_count variable that compute number of instances of specific word in the list
        word_count = document1.count(element)
        # we calculate term frequency inside of the dictionary by dividing number of instances of word in dictionary by total number of words there
        elements_dictionary[element] = (word_count / total_length)
    # dictionary is then appended to dictionary_list
    dictionary_list.append(elements_dictionary)
  # dictionary list is returned in the end of the function
  return dictionary_list




#2 Testing
print(term_frequency([['THIS', 'IS', 'A', 'TEST', 'SENTENCE', 'IS'], ['HERE', 'IS', 'A', '2ND', 'EXAMPLE']]))
```

    [{'THIS': 0.16666666666666666, 'IS': 0.3333333333333333, 'A': 0.16666666666666666, 'TEST': 0.16666666666666666, 'SENTENCE': 0.16666666666666666}, {'HERE': 0.2, 'IS': 0.2, 'A': 0.2, '2ND': 0.2, 'EXAMPLE': 0.2}]
    

R


```python
#-- token %in% names(tf_dict) may be useful to check if word is in dictionary
#1 Code
term_frequency = function(list_of_text) {
    # blank list called dictionary_list
    dictionary_list = list()
    # for loop that iterates over each document in the input list
    for (document1 in list_of_text) {
        # temporary dictionary called elements_dictionary is created inside of the outer for loop
        elements_dictionary = list()
        # inner for loop that iterates over each word in the document
        for (element in document1) {
            # function records length of the document and assigns it to the total_length variable
            total_length = length(document1)
            # function counts number of instances of word in the document
            word_count = sum(document1 == element)
            # function then calculates a term frequency of word inside of the elements_dictionary by formula word_count / total_length
            elements_dictionary[[element]] = (word_count / total_length)

        }
        # after innter loop finishes, the temporary dictionary is appended to the dictionary_list
        dictionary_list = c(dictionary_list, list(unlist(elements_dictionary)))
    }
    # function returns dictionary_list as a result
    return(dictionary_list)
}


#2 Testing
l1 <- c('THIS', 'IS', 'A', 'TEST', 'SENTENCE', 'IS')
l2 = c('HERE', 'IS', 'A', '2ND', 'EXAMPLE')
corp = list(l1, l2)
tf <- term_frequency(corp)
print(tf)

```

    [[1]]
         THIS        IS         A      TEST  SENTENCE 
    0.1666667 0.3333333 0.1666667 0.1666667 0.1666667 
    
    [[2]]
       HERE      IS       A     2ND EXAMPLE 
        0.2     0.2     0.2     0.2     0.2 
    
    

## 5. Create the function inverse_document_frequency (12.5 pts)

@inverse_document_frequency(list_of_text: list)
  - Takes a list of strings and computes a word count saved in a  dictionary. The dictionary should have keys that relate to a token (e.g. word) and the value should be the inverse document frequency of that word
  - Arguments: list_of_text str representing a list of strings. This should be the output of text_to_tokens.
  - Returns: dictionary featuring word counts
  ```
  Illustrative Examples - Values may not be correct
  Input  --> [["THE", "DOG", "ATE", "2", "CATS"],
              ["THE","STARS","ARE","SO","BRIGHT"]]
  Output --> {"THE":0, "DOG":0, "ATE":0, "2":0, "CATS":0,
              "STARS":0,"ARE":0,"SO":0,"BRIGHT":0}
  ```




Python


```python
#1 Libraries
# important math library to calculate IDF
import math

#2 Code
def inverse_document_frequency(list_of_text:list):
  # list_length variable that calculates the length of input list
  list_length = len(list_of_text)
  # blank dictionary for elements in each list
  elements_dictionary2 = {}
  # outer for loop that iterates over each list (document) in the input
  for document2 in list_of_text:
    # inner for loop that uses set() function on list in the outer for loop which makes list count duplicate words only once.
    for element1 in set(document2):
      # if element is in the dictionary, add 1 to the value of the element in the dictionary
      if element1 in elements_dictionary2:
        elements_dictionary2[element1] += 1
        # in any other scenario assign 1 to the value of the word in the dictionary
      else:
        elements_dictionary2[element1] = 1

  # another outer for loop that iterates over names as element3 and value of elements as value from dictionary
  for element3, value in elements_dictionary2.items():
      # for loop then calculates idf
      elements_dictionary2[element3] = math.log(list_length / value)

  # function returns a dictionary
  return elements_dictionary2






#3 Testing
test_corpus = [
    ["IT", "IS", "GOING", "TO", "RAIN", "TODAY"],
    ["TODAY", "I", "AM", "NOT", "GOING", "OUTSIDE"],
    ["I", "AM", "GOING", "TO", "WATCH", "THE", "SEASON", "PREMIERE"]
]
for i in inverse_document_frequency(test_corpus).items():
  print(i)
```

    ('TODAY', 0.4054651081081644)
    ('RAIN', 1.0986122886681098)
    ('IS', 1.0986122886681098)
    ('GOING', 0.0)
    ('IT', 1.0986122886681098)
    ('TO', 0.4054651081081644)
    ('NOT', 1.0986122886681098)
    ('OUTSIDE', 1.0986122886681098)
    ('I', 0.4054651081081644)
    ('AM', 0.4054651081081644)
    ('WATCH', 1.0986122886681098)
    ('THE', 1.0986122886681098)
    ('PREMIERE', 1.0986122886681098)
    ('SEASON', 1.0986122886681098)
    

R


```python
#1 Code
inverse_document_frequency = function(list_of_text) {
    # list_length variable that records the length of the input list
    list_length = length(list_of_text)
    # blank list that will work as a dictionary in R called elements_dictionary2
    elements_dictionary2 = list()
    # outer for loop that iterates over each document in the list
    for (document2 in list_of_text) {
        # inner for loop that iterates over each element in the document
        # we need unique() function as we want to calculate number of document that contain the term without including the duplicates
        for (element1 in unique(document2)) {
            # if statement that adds +1 to the number of instances of element if the element is already in the elements_dictionary
            if (element1 %in% names(elements_dictionary2)) {
                elements_dictionary2[[element1]] = elements_dictionary2[[element1]] + 1
            }
            # else statement that assigns 1 to number of instances of element in any other scenario
            else {
                elements_dictionary2[[element1]] = 1
            }
        }
    }
    # another outer for loop that calculates the value of the word in the elements_dictionary by the formula log(list_length / number of documents with this word)
    for (word in names(elements_dictionary2)) {
        elements_dictionary2[[word]] = unlist(log(list_length / elements_dictionary2[[word]]))
    }
    # at the end, function returns the dictionary as a result
    return(elements_dictionary2)
}



#2 Testing

doc1 = c("IT", "IS", "GOING", "TO", "RAIN", "TODAY")
doc2 = c("TODAY", "I", "AM", "NOT", "GOING", "OUTSIDE")
doc3 = c("I", "AM", "GOING", "TO", "WATCH", "THE", "SEASON", "PREMIERE")
test_corpus = list(doc1, doc2, doc3)

idf = inverse_document_frequency(test_corpus)
for (name in names(idf)){
 print(paste(name, as.character(idf[name])))
}
```

    [1] "IT 1.09861228866811"
    [1] "IS 1.09861228866811"
    [1] "GOING 0"
    [1] "TO 0.405465108108164"
    [1] "RAIN 1.09861228866811"
    [1] "TODAY 0.405465108108164"
    [1] "I 0.405465108108164"
    [1] "AM 0.405465108108164"
    [1] "NOT 1.09861228866811"
    [1] "OUTSIDE 1.09861228866811"
    [1] "WATCH 1.09861228866811"
    [1] "THE 1.09861228866811"
    [1] "SEASON 1.09861228866811"
    [1] "PREMIERE 1.09861228866811"
    

## 6. Create the tfidf function (12.5 pts)
@tfidf(list_of_text: list)
  - Takes a list of strings and calls the term_frequency and inverse_document_frequency functions to compute the TFIDF.
  - Arguments: list_of_text str representing a list of strings. This should be the output of text_to_tokens.
  - Returns: dictionary featuring word counts
  ```
  Illustrative Examples - Values may not be correct
  Input  --> [["THE", "DOG", "ATE", "2", "CATS"],
              ["THE","STARS","ARE","SO","BRIGHT"]]
  Output --> [{"THE":0, "DOG":0, "ATE":0, "2":0, "CATS":0},
              {"THE":0,"STARS":0,"ARE":0,"SO":0,"BRIGHT":0}]
  ```

Python


```python
#1 Code
def tfidf(list_of_text:list):
  # function applies term_frequency() function on the input list and assigns the result to the tf variable
  tf = term_frequency(list_of_text)
  # function applies inverse_document_frequency function on the input list and assigns the result to the idf variable
  idf = inverse_document_frequency(list_of_text)
  # function creates blank list called tfidf_list
  tfidf_list = []

  # outer loop that iterates over each dictionary in tf variable
  for dictionary_tf in tf:
    # temporary dictionary that is created in outer for loop
    temp_dictionary = {}
    # innter for loop that iterates over each element and value of the element in the dictionary_tf
    for element_tf, value_tf in dictionary_tf.items():
      # if the element is already in idf, calculate tfidf by multiplying the value of idf by value of tf
      if element_tf in idf:
        temp_dictionary[element_tf] =  value_tf * idf[element_tf]
    # temporary dictionary is then appended to tfidf_list
    tfidf_list.append(temp_dictionary)

  # tfidf_list is returned as a result of this function
  return tfidf_list






#2 Testing
test_corpus = [
    ["IT", "IS", "GOING", "TO", "RAIN", "TODAY"],
    ["TODAY", "I", "AM", "NOT", "GOING", "OUTSIDE"],
    ["I", "AM", "GOING", "TO", "WATCH", "THE", "SEASON", "PREMIERE"]
]

for i in tfidf(test_corpus):
  print(i)


```

    {'IT': 0.1831020481113516, 'IS': 0.1831020481113516, 'GOING': 0.0, 'TO': 0.06757751801802739, 'RAIN': 0.1831020481113516, 'TODAY': 0.06757751801802739}
    {'TODAY': 0.06757751801802739, 'I': 0.06757751801802739, 'AM': 0.06757751801802739, 'NOT': 0.1831020481113516, 'GOING': 0.0, 'OUTSIDE': 0.1831020481113516}
    {'I': 0.05068313851352055, 'AM': 0.05068313851352055, 'GOING': 0.0, 'TO': 0.05068313851352055, 'WATCH': 0.13732653608351372, 'THE': 0.13732653608351372, 'SEASON': 0.13732653608351372, 'PREMIERE': 0.13732653608351372}
    

R


```python
#1 Code
tfidf = function(list_of_text) {
    # tf variable that uses term_frequency() function on input list
    tf = term_frequency(list_of_text)
    # idf variable that uses inverse_document_frequency() function on input list
    idf = inverse_document_frequency(list_of_text)
    # blank list called tfidf_list
    tfidf_list = list()

    # outer for loop that iterates over each dictionary in the modified tf list
    for (dictionary_tf in tf) {
        # outer loop creates a blank temporary dictionary
        temp_dictionary = list()
        # innner loop that iterates over each element in the dictionary
        for (word in names(dictionary_tf)) {
            # if the word is present as an element in idf variable, function calculates the value of this word in the temporary dictionary by using formula tf value * idf value
            if (word %in% names(idf)) {
                temp_dictionary[[word]] = dictionary_tf[[word]] * idf[[word]]
            }
          }
        # temporary dictionary is then appended to our tfidf_list
        tfidf_list[[length(tfidf_list) + 1]] = unlist(temp_dictionary)

    }
    # function returns tfidf list as a result
    return(tfidf_list)
}





#2 Testing

doc1 = c("IT", "IS", "GOING", "TO", "RAIN", "TODAY")
doc2 = c("TODAY", "I", "AM", "NOT", "GOING", "OUTSIDE")
doc3 = c("I", "AM", "GOING", "TO", "WATCH", "THE", "SEASON", "PREMIERE")
test_corpus = list(doc1, doc2, doc3)

for (i in tfidf(test_corpus)){
  print(i)
}
```

            IT         IS      GOING         TO       RAIN      TODAY 
    0.18310205 0.18310205 0.00000000 0.06757752 0.18310205 0.06757752 
         TODAY          I         AM        NOT      GOING    OUTSIDE 
    0.06757752 0.06757752 0.06757752 0.18310205 0.00000000 0.18310205 
             I         AM      GOING         TO      WATCH        THE     SEASON 
    0.05068314 0.05068314 0.00000000 0.05068314 0.13732654 0.13732654 0.13732654 
      PREMIERE 
    0.13732654 
    

## 7. Create the function tfidf_pipeline (12.5 pts)
@TFIDF_Pipeline(list_of_text: list)
  - Takes a corpus of text in the form of a list of strings, and runs the above functions to clean, tokenize, and calculate TFIDF for each word and each document
  - Arguments: list_of_text str representing a list of strings.
  - Returns: list of dictionaries featuring TFIDF scores for each word in each document. Each item in the list should represent the scores for that document and each key:value pair in the dictionaries should represent word and TFIDF score
  ```
  Illustrative Examples - Values may not be correct
  Input  --> ["The dog, ate 2 cats. ",
              "The stars- are so     bright!"]
  Output --> [{"THE":0, "DOG":0, "ATE":0, "2":0, "CATS":0},
              {"THE":0,"STARS":0,"ARE":0,"SO":0,"BRIGHT":0}]
  ```


Python


```python
#1 Code
#final tfidf_pipeline function that should be used
def tfidf_pipeline(list_of_text:list):
  # function applies clean_corpus() function on the input list and assigns the result to the final_text variable
  final_text = clean_corpus(list_of_text)
  # function applies tfidf() function on the input list and assigns the result to the tfidf_list variable
  tfidf_list = tfidf(final_text)

  # function returns tfidf_list as a result
  return tfidf_list

text = ["The dog, ate 2 cats. ",
          "The stars- are so     bright!"]

for i in tfidf_pipeline(text):
  print(i)

```

    {'THE': 0.0, 'DOG': 0.13862943611198905, 'ATE': 0.13862943611198905, '2': 0.13862943611198905, 'CATS': 0.13862943611198905}
    {'THE': 0.0, 'STARS': 0.13862943611198905, 'ARE': 0.13862943611198905, 'SO': 0.13862943611198905, 'BRIGHT': 0.13862943611198905}
    

R


```python
#1 Code
tfidf_pipeline = function(list_of_text) {
    # pipeline uses clean_corpus() function on input list and assigns it to a final_text variable
    final_text = clean_corpus(list_of_text)
    # pipeline then applies tfidf() function on final_text variable and assigns the result to a tfidf_list variable
    tfidf_list = tfidf(final_text)

    # function returns tfidf_list as a result
    return(tfidf_list)
}

#2 Testing
text = c("The dog, ate 2 cats. ",
          "The stars- are so     bright!")

for (i in tfidf_pipeline(text)) {
    print(i)
}

```

          THE       DOG       ATE         2      CATS 
    0.0000000 0.1386294 0.1386294 0.1386294 0.1386294 
          THE     STARS       ARE        SO    BRIGHT 
    0.0000000 0.1386294 0.1386294 0.1386294 0.1386294 
    

## 8. Run the pipeline on the following corpus (12.5 pts)
- What are some of the words with the highest tfidf in each document? You do not need to program this, you can merely look at the results. Note that because we have a small number of documents, many words will have the highest scores. Explain why some of these have the highest scores.

Answer:

In document 1, Lane Thomas, Wednesday, Jon Lester have a high value of idf. It is because these are unique names and are not present in other documents. Jon Lester and Lane Thomas - names of people. Wednesday - a name of the day in the week

In document 2, U.S, WSJ, Dollar are some of the words that have a high value of tfidf. Again, it is because these words are not present in other documents and they are important for the context of the text.

In document 3, Vivek Ramaswamy, splash, are the words that have a high value of tfidf

In document 4, presidential, primary, debate are some of the words that have a high value of tfidf

In document 5, national, polls, states are some of the words that have a high value of tfidf

In document 6, Tulane, group, five are some of the words that have a high value of tfidf

In document 7, San Francisco, Friday are some of the words that have a high value of tfidf

In document 8, Pelosi, lieutenants, Steny Hoyer, Jim Clyburn are some of the words that have high value of tfidf

In document 9, Emerita Pelosi, talented, transformational, lifetime are some of the words that have high value of tfidf

The key idea is that words that are more unique and important to the context of the text have higher number of tfidf





```python
#1 Text (variable) that is going to be used
corpus = [
    "Right fielder Lane Thomas, 28 on Wednesday and playing at an all-star level, was acquired for now-retired Jon Lester. ",
    "Propelled by signs that the U.S. economy is motoring while the rest of the world flags, the WSJ Dollar Index has appreciated more than 2% over the past month.",
    "At last week’s Republican debate, businessman Vivek Ramaswamy arguably made the biggest splash of any candidate.",
    "With the first Republican primary debate in the books, presidential debate season is officially in full swing.",
    "Candidates will need to hit at least 3 percent in two national polls, or 3 percent in one national poll and 3 percent in two polls conducted from separate early nominating states (Iowa, New Hampshire, South Carolina and Nevada), in order to qualify by the Sept. 25 deadline.",
    'Is Tulane once again the best of the Group of Five?',
    "The San Francisco Democrat and first female speaker of the House told volunteers on Friday that she would seek reelection in 2024.",
    "Pelosi’s top lieutenants, Reps. Steny Hoyer (D-Md.) and Jim Clyburn (D-S.C.), similarly stepped down from their top leadership roles.",
    "Speaker Emerita Pelosi is one of the most talented and transformational leaders of our lifetime."
]
```


```python
#1 Text (variable) that is going to be used
doc1 <- c("Right fielder Lane Thomas, 28 on Wednesday and playing at an all-star level, was acquired for now-retired Jon Lester. ")
doc2 <- c("Propelled by signs that the U.S. economy is motoring while the rest of the world flags, the WSJ Dollar Index has appreciated more than 2% over the past month.")
doc3 <- c("At last week’s Republican debate, businessman Vivek Ramaswamy arguably made the biggest splash of any candidate.")
doc4 <- c("With the first Republican primary debate in the books, presidential debate season is officially in full swing.")
doc5 <- c("Candidates will need to hit at least 3 percent in two national polls, or 3 percent in one national poll and 3 percent in two polls conducted from separate early nominating states (Iowa, New Hampshire, South Carolina and Nevada), in order to qualify by the Sept. 25 deadline.")
doc6 <- c('Is Tulane once again the best of the Group of Five?')
doc7 <- c("The San Francisco Democrat and first female speaker of the House told volunteers on Friday that she would seek reelection in 2024.")
doc8 <- c("Pelosi’s top lieutenants, Reps. Steny Hoyer (D-Md.) and Jim Clyburn (D-S.C.), similarly stepped down from their top leadership roles.")
doc9 <- c("Speaker Emerita Pelosi is one of the most talented and transformational leaders of our lifetime.")
corpus <- c(doc1, doc2, doc2, doc4, doc5, doc6, doc7, doc8, doc9)


```

Python


```python
#2 Testing
for i in tfidf_pipeline(corpus):
  print(sorted(i.items(), reverse =  True))

```

    [('WEDNESDAY', 0.11564339880716945), ('WAS', 0.11564339880716945), ('THOMAS', 0.11564339880716945), ('RIGHT', 0.11564339880716945), ('PLAYING', 0.11564339880716945), ('ON', 0.07916196825138284), ('NOWRETIRED', 0.11564339880716945), ('LEVEL', 0.11564339880716945), ('LESTER', 0.11564339880716945), ('LANE', 0.11564339880716945), ('JON', 0.11564339880716945), ('FOR', 0.11564339880716945), ('FIELDER', 0.11564339880716945), ('AT', 0.057821699403584725), ('AND', 0.030936140258006263), ('AN', 0.11564339880716945), ('ALLSTAR', 0.11564339880716945), ('ACQUIRED', 0.11564339880716945), ('28', 0.11564339880716945)]
    [('WSJ', 0.07576636473573171), ('WORLD', 0.07576636473573171), ('WHILE', 0.07576636473573171), ('US', 0.07576636473573171), ('THE', 0.043330073841535546), ('THAT', 0.05186473781987152), ('THAN', 0.07576636473573171), ('SIGNS', 0.07576636473573171), ('REST', 0.07576636473573171), ('PROPELLED', 0.07576636473573171), ('PAST', 0.07576636473573171), ('OVER', 0.07576636473573171), ('OF', 0.020268505686279966), ('MOTORING', 0.07576636473573171), ('MORE', 0.07576636473573171), ('MONTH', 0.07576636473573171), ('IS', 0.027963110904011337), ('INDEX', 0.07576636473573171), ('HAS', 0.07576636473573171), ('FLAGS', 0.07576636473573171), ('ECONOMY', 0.07576636473573171), ('DOLLAR', 0.07576636473573171), ('BY', 0.05186473781987152), ('APPRECIATED', 0.07576636473573171), ('2', 0.07576636473573171)]
    [('WEEKS', 0.13732653608351372), ('VIVEK', 0.13732653608351372), ('THE', 0.015707151767556635), ('SPLASH', 0.13732653608351372), ('REPUBLICAN', 0.09400483729851714), ('RAMASWAMY', 0.13732653608351372), ('OF', 0.03673666655638244), ('MADE', 0.13732653608351372), ('LAST', 0.13732653608351372), ('DEBATE', 0.09400483729851714), ('CANDIDATE', 0.13732653608351372), ('BUSINESSMAN', 0.13732653608351372), ('BIGGEST', 0.13732653608351372), ('AT', 0.06866326804175686), ('ARGUABLY', 0.13732653608351372), ('ANY', 0.13732653608351372)]
    [('WITH', 0.12924850454918937), ('THE', 0.02956640332716543), ('SWING', 0.12924850454918937), ('SEASON', 0.12924850454918937), ('REPUBLICAN', 0.08847514098683966), ('PRIMARY', 0.12924850454918937), ('PRESIDENTIAL', 0.12924850454918937), ('OFFICIALLY', 0.12924850454918937), ('IS', 0.04770177742448993), ('IN', 0.12924850454918937), ('FULL', 0.12924850454918937), ('FIRST', 0.08847514098683966), ('DEBATE', 0.17695028197367932), ('BOOKS', 0.12924850454918937)]
    [('WILL', 0.0457755120278379), ('TWO', 0.0915510240556758), ('TO', 0.0915510240556758), ('THE', 0.005235717255852212), ('STATES', 0.0457755120278379), ('SOUTH', 0.0457755120278379), ('SEPT', 0.0457755120278379), ('SEPARATE', 0.0457755120278379), ('QUALIFY', 0.0457755120278379), ('POLLS', 0.0915510240556758), ('POLL', 0.0457755120278379), ('PERCENT', 0.13732653608351372), ('ORDER', 0.0457755120278379), ('OR', 0.0457755120278379), ('ONE', 0.03133494576617238), ('NOMINATING', 0.0457755120278379), ('NEW', 0.0457755120278379), ('NEVADA', 0.0457755120278379), ('NEED', 0.0457755120278379), ('NATIONAL', 0.0915510240556758), ('LEAST', 0.0457755120278379), ('IOWA', 0.0457755120278379), ('IN', 0.0915510240556758), ('HIT', 0.0457755120278379), ('HAMPSHIRE', 0.0457755120278379), ('FROM', 0.03133494576617238), ('EARLY', 0.0457755120278379), ('DEADLINE', 0.0457755120278379), ('CONDUCTED', 0.0457755120278379), ('CAROLINA', 0.0457755120278379), ('CANDIDATES', 0.0457755120278379), ('BY', 0.03133494576617238), ('AT', 0.02288775601391895), ('AND', 0.024491111037588293), ('3', 0.13732653608351372), ('25', 0.0457755120278379)]
    [('TULANE', 0.19974768884874725), ('THE', 0.04569353241471021), ('ONCE', 0.19974768884874725), ('OF', 0.1068703027094762), ('IS', 0.07372092874693899), ('GROUP', 0.19974768884874725), ('FIVE', 0.19974768884874725), ('BEST', 0.19974768884874725), ('AGAIN', 0.19974768884874725)]
    [('WOULD', 0.09987384442437362), ('VOLUNTEERS', 0.09987384442437362), ('TOLD', 0.09987384442437362), ('THE', 0.022846766207355106), ('THAT', 0.06836715439892156), ('SPEAKER', 0.06836715439892156), ('SHE', 0.09987384442437362), ('SEEK', 0.09987384442437362), ('SAN', 0.09987384442437362), ('REELECTION', 0.09987384442437362), ('ON', 0.06836715439892156), ('OF', 0.02671757567736905), ('IN', 0.04993692221218681), ('HOUSE', 0.09987384442437362), ('FRIDAY', 0.09987384442437362), ('FRANCISCO', 0.09987384442437362), ('FIRST', 0.06836715439892156), ('FEMALE', 0.09987384442437362), ('DEMOCRAT', 0.09987384442437362), ('AND', 0.02671757567736905), ('2024', 0.09987384442437362)]
    [('TOP', 0.2312867976143389), ('THEIR', 0.11564339880716945), ('STEPPED', 0.11564339880716945), ('STENY', 0.11564339880716945), ('SIMILARLY', 0.11564339880716945), ('ROLES', 0.11564339880716945), ('REPS', 0.11564339880716945), ('PELOSIS', 0.11564339880716945), ('LIEUTENANTS', 0.11564339880716945), ('LEADERSHIP', 0.11564339880716945), ('JIM', 0.11564339880716945), ('HOYER', 0.11564339880716945), ('FROM', 0.07916196825138284), ('DSC', 0.11564339880716945), ('DOWN', 0.11564339880716945), ('DMD', 0.11564339880716945), ('CLYBURN', 0.11564339880716945), ('AND', 0.030936140258006263)]
    [('TRANSFORMATIONAL', 0.1464816384890813), ('THE', 0.016754295218727077), ('TALENTED', 0.1464816384890813), ('SPEAKER', 0.1002718264517516), ('PELOSI', 0.1464816384890813), ('OUR', 0.1464816384890813), ('ONE', 0.1002718264517516), ('OF', 0.07837155532028255), ('MOST', 0.1464816384890813), ('LIFETIME', 0.1464816384890813), ('LEADERS', 0.1464816384890813), ('IS', 0.05406201441442192), ('EMERITA', 0.1464816384890813), ('AND', 0.03918577766014127)]
    


```python


```


```python


```

R


```python
#2 Testing
for (i in tfidf_pipeline(corpus)) {
  print(i)}
```

         RIGHT    FIELDER       LANE     THOMAS         28         ON  WEDNESDAY 
    0.11564340 0.11564340 0.11564340 0.11564340 0.11564340 0.07916197 0.11564340 
           AND    PLAYING         AT         AN    ALLSTAR      LEVEL        WAS 
    0.03093614 0.11564340 0.07916197 0.11564340 0.11564340 0.11564340 0.11564340 
      ACQUIRED        FOR NOWRETIRED        JON     LESTER 
    0.11564340 0.11564340 0.11564340 0.11564340 0.11564340 
      PROPELLED          BY       SIGNS        THAT         THE          US 
     0.05186474  0.03788318  0.05186474  0.03788318  0.04333007  0.05186474 
        ECONOMY          IS    MOTORING       WHILE        REST          OF 
     0.05186474  0.02026851  0.05186474  0.05186474  0.05186474  0.02026851 
          WORLD       FLAGS         WSJ      DOLLAR       INDEX         HAS 
     0.05186474  0.05186474  0.05186474  0.05186474  0.05186474  0.05186474 
    APPRECIATED        MORE        THAN           2        OVER        PAST 
     0.05186474  0.05186474  0.05186474  0.05186474  0.05186474  0.05186474 
          MONTH 
     0.05186474 
      PROPELLED          BY       SIGNS        THAT         THE          US 
     0.05186474  0.03788318  0.05186474  0.03788318  0.04333007  0.05186474 
        ECONOMY          IS    MOTORING       WHILE        REST          OF 
     0.05186474  0.02026851  0.05186474  0.05186474  0.05186474  0.02026851 
          WORLD       FLAGS         WSJ      DOLLAR       INDEX         HAS 
     0.05186474  0.05186474  0.05186474  0.05186474  0.05186474  0.05186474 
    APPRECIATED        MORE        THAN           2        OVER        PAST 
     0.05186474  0.05186474  0.05186474  0.05186474  0.05186474  0.05186474 
          MONTH 
     0.05186474 
            WITH          THE        FIRST   REPUBLICAN      PRIMARY       DEBATE 
      0.12924850   0.02956640   0.08847514   0.12924850   0.12924850   0.25849701 
              IN        BOOKS PRESIDENTIAL       SEASON           IS   OFFICIALLY 
      0.12924850   0.12924850   0.12924850   0.12924850   0.03457569   0.12924850 
            FULL        SWING 
      0.12924850   0.12924850 
     CANDIDATES        WILL        NEED          TO         HIT          AT 
    0.045775512 0.045775512 0.045775512 0.091551024 0.045775512 0.031334946 
          LEAST           3     PERCENT          IN         TWO    NATIONAL 
    0.045775512 0.137326536 0.137326536 0.091551024 0.091551024 0.091551024 
          POLLS          OR         ONE        POLL         AND   CONDUCTED 
    0.091551024 0.045775512 0.031334946 0.045775512 0.024491111 0.045775512 
           FROM    SEPARATE       EARLY  NOMINATING      STATES        IOWA 
    0.031334946 0.045775512 0.045775512 0.045775512 0.045775512 0.045775512 
            NEW   HAMPSHIRE       SOUTH    CAROLINA      NEVADA       ORDER 
    0.045775512 0.045775512 0.045775512 0.045775512 0.045775512 0.045775512 
        QUALIFY          BY         THE        SEPT          25    DEADLINE 
    0.045775512 0.022887756 0.005235717 0.045775512 0.045775512 0.045775512 
            IS     TULANE       ONCE      AGAIN        THE       BEST         OF 
    0.05343515 0.19974769 0.19974769 0.19974769 0.04569353 0.19974769 0.10687030 
         GROUP       FIVE 
    0.19974769 0.19974769 
           THE        SAN  FRANCISCO   DEMOCRAT        AND      FIRST     FEMALE 
    0.02284677 0.09987384 0.09987384 0.09987384 0.02671758 0.06836715 0.09987384 
       SPEAKER         OF      HOUSE       TOLD VOLUNTEERS         ON     FRIDAY 
    0.06836715 0.02671758 0.09987384 0.09987384 0.09987384 0.06836715 0.09987384 
          THAT        SHE      WOULD       SEEK REELECTION         IN       2024 
    0.04993692 0.09987384 0.09987384 0.09987384 0.09987384 0.04993692 0.09987384 
        PELOSIS         TOP LIEUTENANTS        REPS       STENY       HOYER 
     0.11564340  0.23128680  0.11564340  0.11564340  0.11564340  0.11564340 
            DMD         AND         JIM     CLYBURN         DSC   SIMILARLY 
     0.11564340  0.03093614  0.11564340  0.11564340  0.11564340  0.11564340 
        STEPPED        DOWN        FROM       THEIR  LEADERSHIP       ROLES 
     0.11564340  0.11564340  0.07916197  0.11564340  0.11564340  0.11564340 
             SPEAKER          EMERITA           PELOSI               IS 
          0.10027183       0.14648164       0.14648164       0.03918578 
                 ONE               OF              THE             MOST 
          0.10027183       0.07837156       0.01675430       0.14648164 
            TALENTED              AND TRANSFORMATIONAL          LEADERS 
          0.14648164       0.03918578       0.14648164       0.14648164 
                 OUR         LIFETIME 
          0.14648164       0.14648164 
    


```python


```


```python

```
