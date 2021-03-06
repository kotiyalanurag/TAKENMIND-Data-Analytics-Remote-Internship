import pandas as pd
#loading the excel file
excelfile = pd.ExcelFile('Input File.xlsx')
#loading the sheets of our excel file in data frames
df = excelfile.parse('DC')
df1 = excelfile.parse('Marvel')
df2 = excelfile.parse('Nationality')
df3 = excelfile.parse('Salary')
df4 = excelfile.parse('DogBreeds')
df5 = excelfile.parse('Wrestlemania')
df6 = excelfile.parse('WSM')
df7 = excelfile.parse('Snooker')
df8 = excelfile.parse('Basketball')
df9 = excelfile.parse('Anime')
#converting the data frames to csv sheets
df.to_csv('Converted_DC.csv')
df1.to_csv('Converted_Marvel.csv')
df2.to_csv('Converted_Nationality.csv')
df3.to_csv('Converted_Salary.csv')
df4.to_csv('Converted_DogBreeds.csv')
df5.to_csv('Converted_Wrestlemania.csv')
df6.to_csv('Converted_WSM.csv')
df7.to_csv('Converted_Snooker.csv')
df8.to_csv('Converted_Basketball.csv')
df9.to_csv('Converted_Anime.csv')
