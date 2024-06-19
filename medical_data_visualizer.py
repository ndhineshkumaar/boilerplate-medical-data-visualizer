import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv("medical_examination.csv")

# 2
BMI=df['weight']/((df['height']/100)**2)
df['overweight']=BMI.apply(lambda x: 1 if x>25 else 0)

# 3


# 4
def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Step 4: Group and count data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()

    # Step 5: Draw the catplot
    fig = sns.catplot(x='variable', y='size', hue='value', col='cardio', data=df_cat, kind='bar')
    fig.set_axis_labels('variable', 'total')
    fig._legend.set_title('value')

    # Save the plot
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # Step 10: Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # Step 11: Calculate the correlation matrix
    corr = df_heat.corr()

    # Step 12: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 13: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Step 14: Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    # Save the plot
    fig.savefig('heatmap.png')
    return fig

