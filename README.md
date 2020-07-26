# DS Salary Prediction

[
![sample_image](https://user-images.githubusercontent.com/39486938/88488085-6d168500-cfa4-11ea-9190-2f8a999a1aca.PNG)
](url)

* This is the Repository for Data Science Salary Prediction of Glassdoor's Data Science Job
* First we scrapped Data Science Job from Glassdoor.
* Then We cleaned the data and Did Exploratory Data Analysis and Feature Engineering from different perspectives to know in-detail about the python, excel, aws, and spark jobs.
* For our model building we used are  Linear, Lasso, and Random Forest Regressors using GridsearchCV.


## Resources We Used
* Python version 3.8
* Jupyter Notebook
* We used pandas, numpy, sklearn, matplotlib, seaborn, flask, json, pickle packages.
* Glassdoor Scrapping Article https://towardsdatascience.com/selenium-tutorial-scraping-glassdoor-com-in-10-minutes-3d0915c6d905
* Flask Productionization Article https://towardsdatascience.com/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2


## Data Cleaning and Exploratory Data Analysis
* We analyzed and cleaned this dataset so it can be usable for our model.
* And in EDA part, we simplified our data and analyzed different value counts through graphs also through pivot table


## Model Building
* We transformed our variables into dummy variables
* We used different Models to evaluate
* The Random Forest model performed better than the other approaches on the test set. 

Model Used | MAE | Score
------------ | ------------- | -------------
Random Forest | 11.36 | 95.88%
Linear Regression | 18.93 | 71.65%
Lasso Regression | 19.83 | 64.58%


## Deployed Using Flask

* Here we build Flask API that was hosted on local server with above given tutorial
* This API will take list of values from job and predict the salary.
* To send request to the flask application:
    Run app.py, after installing all the required dependencies.
    In another terminal, run request.py, to get the results.


