The dataset folder contains the dataset used to train crop recommendation system
The Notebooks folder contains the code to train both the crop recommendation and the disease detection models



The website folder contains the working flask application. Here are the steps to run the application

1-Create a python environment
python -m venv venv

2-Run in the python virtual environment
venv\Scripts\activate(on cmd)  or  .\venv\Scripts\Activate.ps1(on powershell)

3-Install all the requirements
pip install -r requirements.txt

---------------------------------One time setup end here---------------------------------------------------------------

(Always run in python venv using the 2nd step if returning)

4- to run the flask application
python app.py

5-Open the browser and go to the url 
127.0.0.1:5000

6- to end session , use ctrl+c in the terminal and then to end venv session type 
deactivate