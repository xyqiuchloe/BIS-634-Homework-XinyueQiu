For this assignment, exercise 1 and 2 is written in file: assignment5.ipynb. 


Exercise 3 is written in fodler app. Throughout exercise 3, I have built three documents: incd.py, index.html, info.html. In app folder, template folder contains file info.html, index.html. incd.py help builds up the routes for flask to access; index contains information of the home page, where users type in the state name and acquire age-adjusted rate; info.html defines the way to acquire the information and return it.


For the exercise 3, the data being extracted comes from source: https://statecancerprofiles.cancer.gov/incidencerates/index.php 

In order to clean this dataset, I have deleted all of the brackets and numbers present after the state name, under the state column. I have deleted Nevada row because its row only contains invalid numbers. I have also modified the column name of age adjusted rate to be 'age-adjusted rate' for it to better read in py file. Therefore the result csv is clean and good to read into python.

For 3.2, I have tried to optimize the look of the html by css style, so that the websites looks a little bit better.
