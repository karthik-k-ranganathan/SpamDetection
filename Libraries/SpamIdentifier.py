import pandas as pd # for data manipulation and file operation
import os # for IO Operations

from sklearn.feature_extraction.text import CountVectorizer #CounterVectorizer Convert the text into matrics

# Naive Bayes Have three Classifier(Bernouli,Multinominal,Gaussian) 
# Here we use Multinominal Bayes Because the data is in a discrete form 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline # https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html
from sklearn.model_selection import train_test_split # Splitting the data into training dataset and test dataset



class SpamIdentifier:
    
    def __init__(self) -> None:
        # Create a private variable for the generated file
        self.__generated_file = '../Data/spam_classification.csv'
        self.SelectedModel = 'None Selected'
        self.ModelAccuracyScore = 0.0
        self.Model = None

        if os.path.exists(self.__generated_file):
             self.__df = pd.read_csv(self.__generated_file)
        else:
             # Since the generated transformed file is not present, we will need to load the 
             # original file and do our ETL / transformation / Preparation of data that is needed for model creation
             self.__orig_df = pd.read_csv('../Data/mail_data.csv')

             # Data Preparation
             self.__orig_df['IsSpam'] = self.__orig_df['Category'].map({'spam': 1, 'ham': 0})

             # Setting it to the original
             self.__df = self.__orig_df

             # Saving the updated file
             self.__df.to_csv(self.__generated_file)

    def __ChoosePredictionAlgorithm(self):
         print('TODO: Not Implemented')
         pass
    
    def Algorithm_Naive_Bayes_MultinomialNB(self):
        self.SelectedModel = 'Naive Bayes MultinomialNB'

        pipe = Pipeline([
              ('vectorizer', CountVectorizer()),
              ('nb', MultinomialNB())
            ])
        
        self.Model = pipe
         
        # Calculating the mean of the accuracy so that way we know if this would work for various test data
        meanAccuracy = 0.0
        repeatCount = 20

        for i in range(1, repeatCount, 1):
            x_train, x_test, y_train, y_test = train_test_split(self.__df['Message'], self.__df['IsSpam'], test_size=0.20)

            # fit the data with the pipeline
            pipe.fit(x_train, y_train)
            meanAccuracy += pipe.score(x_test, y_test)
        
        self.ModelAccuracyScore = (meanAccuracy / repeatCount)


    # Returns 0 if this is not a spam
    # returns 1 if this is a spam
    def IsSpam(self, message):
           vectorizedMessage = [message] 
           prediction = self.Model.predict(vectorizedMessage)
           return prediction[0]
        

    
    # Returns Spam if the response of the IsSpam is 1 else returns Not a Spam
    def DecodeSpamOutput(self, response):
          if response == 0:
                return 'Not a Spam'
          else:
               return 'Spam' 

# Code to call the class files          
si = SpamIdentifier()
print("Model Used:", si.SelectedModel)

msg = "This is a test"
print(f"Is Spam Text:", si.IsSpam(msg))
