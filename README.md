**<h1>BCG Client Customer Churn Analysis</h1>**
**<h2>Overview</h2>**

This repository contains an analysis of client churn within the BCG (Boston Consulting Group) context. 
The goal is to understand whether price sensitivity is a significant driver of churn. 
I explored historical customer data, pricing information, and churn indicators.

**<h2>Project Components</h2>**
**1. Exploratory Data Analysis (EDA):**
The dataset was explored, uncovering patterns and relationships.
EDA helps to understand the distribution of features and identify potential drivers of churn.

**2. Feature Engineering:**
Engineer relevant features to enhance model performance.
This involved the creation of new variables, transforming existing ones, and selecting impactful features.

**3. Predictive Modeling with Random Forest Classifier:**
A predictive model was built using the Random Forest classifier which aims to predict churn based on historical data and feature engineering insights.

**Data Description**

The dataset includes the following components:

Historical Customer Data: Details about customer usage, sign-up dates, forecasted usage, etc.

Historical Pricing Data: Variable and fixed pricing information.

Churn Indicator: Whether each customer has churned or not.

**Data Source:** client_data.csv and price_data.csv datasets from BCG job simulation on Forage.

**<h3>Exploratory Data Analysis</h3>**
<b>1. Overview of Client Data</b>
The first step was understanding the structure of the client and price dataframe by exploring the first few rows, data types, and summary statistics.

<pre># Display the first 3 rows of client dataframe
client_df.head(3)

# Display the first 3 rows of price dataframe
price_df.head(3)

# Get an overview of data types and non-null values
client_df.info()

price_df.info()</pre>

<b>2. Descriptive Statistics</b>
Summary statistics for relevant columns was calculated to give insights into the distribution of numerical features.

<pre># Summary statistics
client_df.describe()

price_df.describe()</pre>

<b>3. Visualization: Churning Status</b>
A stacked bar chart was created to visualize the churning status of the client's customers, with the bars representing the percentage of retained and churned client customer.
<pre># Calculate the percentage of retained and churned companies
churn_total = churn.groupby(churn['churn']).count()
churn_percentage = churn_total / churn_total.sum() * 100

# Plot the stacked bar chart
plot_stacked_bars(churn_percentage.transpose(), "Churning Status", (5, 5), legend_="lower right")</pre>

![image](https://github.com/Ebenola/Customer-Churn-Analysis/assets/104829299/74cab839-edc0-4005-9eb7-cc7879a9322c)

The distribution of the cons_12m column was plotted to understand the consumption patterns for both retained and churned client customers.

<pre># Select relevant columns for consumption analysis
consumption = client_df[['id', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'imp_cons', 'has_gas', 'churn']]

# Create a subplot
fig, axs = plt.subplots(nrows=1, figsize=(18, 5))

# Plot the distribution of 'cons_12m'
plot_distribution(consumption, 'cons_12m', axs)</pre>

![image](https://github.com/Ebenola/Customer-Churn-Analysis/assets/104829299/11695a07-514c-4d85-b7f8-db18a42c773c)

**<h3>Feature Engineering</h3>**

**1. Off-Peak Price Differences**
Two new features were created:

a. offpeak_diff_dec_january_energy: The difference in off-peak energy prices between December and January.

b. offpeak_diff_dec_january_power: The difference in off-peak power prices between December and January.
<pre># Difference between off-peak prices in December and preceding January

# Group off-peak prices by companies and month

monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()

# Get january and december prices

jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

# Calculate the difference
diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), jan_prices.drop(columns='price_date'), on='id')
diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
diff = diff[['id', 'offpeak_diff_dec_january_energy','offpeak_diff_dec_january_power']]
diff.head()</pre>

**2. Price Changes and Customer Duration**

The mean difference between different price periods (off-peak to peak, peak to mid-peak, etc.) was calculated which provide insights into how price fluctuations impact churn behavior

I also calculated the duration of customer engagement (from date_activ to date_end). The result shows that customers with an active period of 4 months or less are more likely to churn.
<pre># Checking how long a customer has been with the client in relation to churn

df['duration'] = ((df['date_end'] - df['date_activ'])/ np.timedelta64(1, 'Y')).astype(int)

df.groupby(['duration']).agg({'churn': 'mean'}).sort_values(by='churn', ascending=False)</pre>

**3. Categorical Variables and Numerical Transformation**

The has_gas column was transformed from categorical to binary (1 for gas customers, 0 for non-gas customers). This will be useful for predictive modeling.

Dummy variables was created for channel_sales and origin_up. These will help capture the impact of different sales channels and customer origins.

Logarithmic transformation was applied to skewed numerical features (e.g., cons_12m, cons_gas_12m, etc.). This helps handle extreme values and improves model performance.

**4. Histogram Visualization of the Distributions**
<pre># Checking the distributions and plotting histogram

fig, axs = plt.subplots(nrows=3, figsize=(15, 20))
sns.distplot((df["cons_12m"].dropna()), ax=axs[0])
sns.distplot((df[df["has_gas"]==1]["cons_gas_12m"].dropna()), ax=axs[1])
sns.distplot((df["cons_last_month"].dropna()), ax=axs[2])
plt.show()</pre>
![image](https://github.com/Ebenola/Customer-Churn-Analysis/assets/104829299/5ce2249b-63be-4e80-afa9-89282206d62d)

<pre># Setting features that have 0 correlation with all of the independent variables and a high correlation with the 
# target variable (churn).

correlation = df.corr()

plt.figure(figsize=(50, 50))
sns.heatmap(
    correlation, 
    xticklabels=correlation.columns.values,
    yticklabels=correlation.columns.values, 
    annot=True, 
    annot_kws={'size': 30})

# Axis ticks size
plt.xticks(fontsize=50)
plt.yticks(fontsize=50)
plt.show()</pre>
![image](https://github.com/Ebenola/Customer-Churn-Analysis/assets/104829299/28bc93a4-073c-4263-a18e-ad9e2a7db0aa)

Remove variables which exhibit a high correlation with other independent features.

Using correlation of 0.9 and above will result in removing cons_gas_12m, forecast_cons_year, num_years_antig, off_peak_mid_peak_var_mean_diff, and off_peak_mid_peak_fix_mean_diff

**<h3>Predictive Modeling: Random Forest Classifier</h3>**
**1. Data Splitting:**
The dataset was splitted into training and test samples.
<pre># Predictive model using Random Forest classifier

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Using a copy of dataset for training sample

df_training = df.copy()</pre>

**2. Target and Features:**
The target variable (churn) has been separated from the independent features (X).
<pre>y = df['churn']
X = df.drop(columns=['id', 'churn', 'date_activ', 'date_end', 'date_modif_prod', 'date_renewal'])
print(X.shape)
print(y.shape)</pre>

**3. Random Forest Classifier:**
Random Forest classifier was used for churn prediction.

**4. Model Evaluation:**
The Random Forest classifier achieved an accuracy of approximately 90.3%.

Precision (true positive rate) is high (92.9%), indicating that when the model predicts churn, it is often correct.

Recall (sensitivity) is low (3.6%), meaning that the model misses many actual churn cases.

**5. Feature Importance:**
<pre># Using Feature Importance to check the number of times each feature is used for splitting across all trees

feature_importances = pd.DataFrame({
    'features': X_train.columns,
    'importance': rfc.feature_importances_
}).sort_values(by='importance', ascending=True).reset_index()

plt.figure(figsize=(15, 25))
plt.title('Feature Importances')
plt.barh(range(len(feature_importances)), feature_importances['importance'], color='b', align='center')
plt.yticks(range(len(feature_importances)), feature_importances['features'])
plt.xlabel('Importance')
plt.show()</pre>
Consumption over 12 months (cons_12m) is a top driver for churn in this model.

Net margin is also an important feature.

Price sensitivity does not appear to be a significant driver for churn.
![image](https://github.com/Ebenola/Customer-Churn-Analysis/assets/104829299/e1d1478d-e84a-4e76-949a-e72146726f57)

**6. Improvement Opportunities:**

To improve recall, explore additional predictive features or fine-tuning the model.

Investigate other factors beyond the current features to enhance churn prediction.

