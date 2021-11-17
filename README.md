# COVID_Symptoms_Extractor
Extracts tweets containing COVID Symptoms

Input: seed texts i.e. tweets containing "cough" as COVID symptom and DataSet of COVID relate Tweets

First run generateStates.py to generate Embeddings. 

Then, run trainADR.py to identify tweets containing COVID Symtoms

As the BERT model learns encoding from the contexts of the token, it also captures spelling mistakes like caugh (cough) or short words like fevr (fever) because these tokens occur in similar context. 
We were able to capture 11 out of 13 confirmed symptoms by WHO. 
The method captured a wide range of symptoms and still missed some rare symptoms like conjunctivitis and loss of speech or movement as it is not reported frequently in tweets. 

This can provide insights to researchers and government bodies to decide their plan of action and also curb any misinformation regarding symptoms of COVID-19 or the side effects of COVID-19 vaccines in social media. 
