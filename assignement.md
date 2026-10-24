This assignment can be completed in teams of three students. Exceptionally, teams of two will be accepted upon my approval.

Given this data set including news, the assignment consists of the following tasks:

1. Implement the following pseudocode to calculate the variable ratio_quality using the TFIDF vectors built from the description fields:

total_goods = 0

For every article (a) on topic (article_section) "Food and Drink":

   Obtain the top-10 most similar articles (top-10) in Corpus to a

   Count how many articles in top-10 are related to topic "Food and Drink" (goods)

   total_goods = total_goods + goods

endFor

ratio_quality = total_goods/(num_articles_food_and_drink*10)

 

And measure the execution times separately for the following two subprocesses: 

    Creating the model (from the program begin to the call similarities.MatrixSimilarity(tfidf_vectors))
    Implementation of the pseudocode above.

 

2. Repeat the previous task with LDA vectors and compare the results. Explain the differences in the results and execution times. Use 30 topics, two passes, and a random state parameter = 30.

3. Repeat the previous two tasks but with the topic "Sports" and compare the results. Why do you think that the quality of the results is worse than the ones obtained with the topic "Food and Drink"?

4. Explain how you can get better results comparing articles by resorting to tagging. Propose a method to calculate the similarity between two articles using their associated tags. Note that the articles in the data set are already tagged.

5. (Only for students of HCID master and Erasmus students of this Master) Explain how you would implement a content-based RS for news using the techniques mentioned in this statement. Note that you would need to work with a user profile/model, so explain briefly which kind of information you would store in this model. Elaborate a low-fidelity prototype including the main screens of this RS.

To do this assignment, you must use the python libraries Gensim and NLTK, which provide implementations for the CB algorithms addressed in the course.

You will have to submit a zip/rar file containing a document explaining the results obtained in the proposed tasks, the discussion of these results, and the source code. 

Only one member of each team must submit the assignment in Moodle.

Use of AI tools: it is allowed for code generation, but the students must make sure of fully understanding the submitted code. If I deem it appropriate, I may ask students to explain their code if I detect a code that is excessively complex.
