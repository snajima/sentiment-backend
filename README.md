# Sentiment
An open-source backend for an app showing your mood trends based off of your journal entries.

## iOS Repository
https://github.com/snajima/sentiment-ios

## Installation

This project uses the Django Rest Framework. 
Clone the project with

```
git clone https://github.com/snajima/sentiment-backend.git
```
After cloning the project, `cd` into the new directory and install dependencies with

```
$ python3 -m venv venv
$ . venv/bin/activate
(venv) $ pip install -r requirements.txt
```

You can run the project with

```
(venv) $ python3 manage.py runserver
```
You can update the database schema with
```
(venv) $ python3 manage.py makemigrations
(venv) $ python3 manage.py migrate
```

## AI Algorithms

This project was created for Cornell's class CS 4701: Practicum in Artificial Intelligence, and the AI code is all contained within the folder `sentiment/entries/controllers/algorithm/`.
