import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv'
df = pd.read_csv(url)
df['lifeExp'] = df['lifeExp'].astype(int)
df.head()

df_piv = df.pivot('country', 'year', 'lifeExp')
df_piv

plt.figure(figsize=(7,7))
ax = sns.heatmap(df_piv).get_figure().savefig('HeatMap.png')
plt.title('Heatmap for lifeExp value of year vs country')
plt.xlabel('year')
plt.ylabel('country')
plt.show()
