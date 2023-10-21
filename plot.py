# import dependent libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import math


# DIRECTORIES
PROJECT_FILE = ""
MULTI_SIZE_FILE = PROJECT_FILE + 'data/HH/multi_size.csv' 
MULTI_FILE = PROJECT_FILE + 'data/HH/multi_data.csv' 



def get_car_type(length):
    
    if 5 < length:
        return "Large"
    if length <= 4.6:
        return "Small"
    if 4.6 < length <= 5:
        return "Medium"
    
    return None

# Read data for both plots
file_path = MULTI_FILE
df = pd.read_csv(file_path)

# First plot
x_lead = df['v_lead'].values
y_lead = df['a_lead'].values

# Second plot
x_follow = df['v_follow'].values
y_follow = df['a_follow'].values

# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# First subplot - Lead Cars
axes[0].set_title('Scatter Heatmap for lead cars velocity-acceleration')
axes[0].set_xlabel('Lead Velocity')
axes[0].set_ylabel('Lead Acceleration')
sns.scatterplot(x=x_lead, y=y_lead, s=5, color=".15", ax=axes[0])
sns.histplot(x=x_lead, y=y_lead, bins=50, pthresh=.1, cmap="mako", ax=axes[0])
sns.kdeplot(x=x_lead, y=y_lead, levels=5, color="w", linewidths=1, ax=axes[0])

# Second subplot - Follow Cars
axes[1].set_title('Scatter Heatmap for follow cars velocity-acceleration')
axes[1].set_xlabel('Follow Velocity')
axes[1].set_ylabel('Follow Acceleration')
sns.scatterplot(x=x_follow, y=y_follow, s=5, color=".15", ax=axes[1])
sns.histplot(x=x_follow, y=y_follow, bins=50, pthresh=.1, cmap="mako", ax=axes[1])
sns.kdeplot(x=x_follow, y=y_follow, levels=5, color="w", linewidths=1, ax=axes[1])

# Adjust the spacing between subplots
plt.tight_layout()

# Show the entire figure
plt.show()