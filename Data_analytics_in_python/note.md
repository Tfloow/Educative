# Data Analytics in Python

- [Data Analytics in Python](#data-analytics-in-python)
  - [1. Comparing Wages With Consumer Price Index Data](#1-comparing-wages-with-consumer-price-index-data)
  - [2. Wages and CPI: Reality Check](#2-wages-and-cpi-reality-check)
  - [3. Working With Major US Storm Data](#3-working-with-major-us-storm-data)
  - [4. Property Rights and Economic Development](#4-property-rights-and-economic-development)
  - [5. How Representative Is Your Government?](#5-how-representative-is-your-government)
  - [6. Does Wealth Influence The Prevalence Of Mental Illness?](#6-does-wealth-influence-the-prevalence-of-mental-illness)
  - [7. Do Birthdays Make Elite Athletes?](#7-do-birthdays-make-elite-athletes)
  - [8. Does Literacy Impact The Income of People?](#8-does-literacy-impact-the-income-of-people)
  - [Conclusion](#conclusion)


This course is based on 8 little scenario which makes the learning fun and interactive.

## 1. Comparing Wages With Consumer Price Index Data

The Bureau of Labor Statistics (BLS) is a US government department. They collect and publish huge amounts of employment and economics-related data and have done for decades now.

Their data is well organized and well-supported API.

In this example we will request the CPI or Consumer Price Index in the US.

To get an API key go over [there](https://data.bls.gov/registrationEngine/registerkey). (If you want to try just type ``$Env:BLS_API_KEY='YOUR API KEY'``).

Also here is the blv [website](https://beta.bls.gov/dataQuery/find?removeAll=1&q=computer). We will do request like this to find the right endpoint (series ID). And we have python package with those data called `bls`.

[**Notebook**](Comparing_wages.ipynb)

## 2. Wages and CPI: Reality Check

There are too many variables in play to be able to definitively say why this happened. Possible theories include government policies, business regulation, changing tax rates, or other external factors. Correlation doesn’t equal causation, and the real world is more complex than a two-line graph, but it’s always important to ask questions of your data and the historical context behind it.

We will also compare our data to the **S&P**. (I extracted the csv from educative)

## 3. Working With Major US Storm Data

## 4. Property Rights and Economic Development

## 5. How Representative Is Your Government?

## 6. Does Wealth Influence The Prevalence Of Mental Illness?

## 7. Do Birthdays Make Elite Athletes?

## 8. Does Literacy Impact The Income of People?

## Conclusion