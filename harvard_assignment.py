import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os

### Problem 1

## Problem 1 (a)
# Load in the CSV files from the Sean Lahman's Baseball Database

#zip_ref = zipfile.ZipFile("lahman-csv_2014-02-14.zip", 'r')
#zip_ref.extractall("lahman_data")
#zip_ref.close()
#
#os.chdir("lahman_data")
salaries = pd.read_csv("Salaries.csv") #Reading the dataset in a dataframe using Pandas
teams = pd.read_csv("Teams.csv")

salaries.head()
teams.head()

salaries.describe()  # get summary of numerical variables
teams.describe()


## Problem 1 (b)
# Summarize the Salaries DataFrame to show the total salaries for each team for each year.
summaryDF = salaries.pivot_table(index = "teamID", columns = "yearID", values = "salary", aggfunc= sum)
summaryDF.head()


## Problem 1 (c)
# Merge the new summarized Salaries DataFrame and Teams DataFrame together to create a new DataFrame
# showing wins and total salaries for each team for each year year
ddff = summaryDF.stack()
latestDF = ddff.reset_index()
latestDF.columns = ['teamID', 'yearID','TotalSalary']
finalDF = pd.merge(teams, latestDF, how='inner', on=['yearID', 'teamID'])


## Problem 1 (d)
# graphically display the relationship between total wins and total salaries for a given year
wsDF = finalDF.pivot_table(index = "teamID", columns = "yearID", values = ['W','TotalSalary'], aggfunc= sum)
newWSdf = wsDF.stack()
latestWSdf = newWSdf.reset_index()
#latestWSdf[latestWSdf.teamID == 'OAK']

winBin = pd.cut(latestWSdf.W, 10)
salaryBin = pd.cut(latestWSdf.TotalSalary, 10)

latestWSdf['salaryBin'] = salaryBin
latestWSdf['winBin'] = winBin
latestWSdf.head()

for year in latestWSdf.yearID.unique():
    mapDF = latestWSdf[latestWSdf.yearID == year]
    x = mapDF.W
    y = mapDF.TotalSalary
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, y, '.')
    plt.plot(x, m*x + b, '-')




### Problem 2

## Problem 2(a) - Load data
countries = pd.read_csv("countries.csv")
income = pd.read_excel("indicator gapminder gdp_per_capita_ppp.xlsx")

income_df = pd.melt(income, id_vars=["GDP per capita"], 
                  var_name="Year", value_name="Income")
income_final_df = income_df.pivot_table('Income', ['Year'], 'GDP per capita')

## Problem 2(b)
# Graphically display the distribution of income per person across all
# countries in the world for any given year (e.g. 2000)
income_final_df.ix[2000].plot()
income_final_df.ix[2000].plot(kind='bar')

## Problem 2(c)
# Write a function to merge the countries and income data sets for any given year.
icomeDF = income.rename(columns={"GDP per capita":"Country"})

def mergeByYear(year):
    icomeDF = pd.merge(countries, icomeDF[["Country",year]])
    returnDF = icomeDF.rename(columns={year:"Income"})
    return returnDF
    
## Problem 2(d)
# Use exploratory data analysis tools such as histograms and boxplots to explore
# the distribution of the income per person by region data set
finalData = mergeByYear(1800)
finalData.groupby('Region').boxplot()








