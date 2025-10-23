import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.linear_model import LinearRegression
import plotly.express as px

sns.set_theme(style="whitegrid")

# --- Load datasets ---
dfGDP = pd.read_csv('World_GDP (1).csv').set_index('Country')
dfComb = pd.read_csv('df_combined (1).csv').set_index('Country')
dfMaininfo = pd.read_csv('World_MainInfo (3).csv', index_col=0)
dfBo = pd.read_csv('Bonus_Info (3).csv', index_col=0)

# --- Functions ---
def show_country_on_map(country_name):
    try:
        fig = px.choropleth(
            locations=[country_name],
            locationmode="country names",
            color_discrete_sequence=["royalblue"],
            title=f"🌍 Location of {country_name}",
            projection="natural earth"
        )
        fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not display map for {country_name}: {e}")

def get_country_data(df, country_name):
    if country_name not in df.index:
        st.warning(f"No data for {country_name}")
        return None
    return df.loc[[country_name]]

# --- Sidebar ---
User_Country = st.sidebar.selectbox("Select a Country:", sorted(dfGDP.index))
Year_to_predict = st.sidebar.number_input("Year to predict:", min_value=2025, max_value=2050, step=1)

# --- Display map ---
show_country_on_map(User_Country)

# --- Main info ---
st.subheader(f"📋 Main Info: {User_Country}")
st.dataframe(get_country_data(dfMaininfo, User_Country))

if st.button("🔍 Show More Info"):
    st.subheader(f"🌟 Bonus Info: {User_Country}")
    st.dataframe(get_country_data(dfBo, User_Country))

# --- GDP Plot ---
country_gdp = dfGDP.loc[User_Country].transpose()
country_gdp.index = country_gdp.index.astype(int)
country_gdp.name = "GDP"

plt.figure(figsize=(12,6))
plt.plot(country_gdp.index, country_gdp.values, marker='o', color='#2a9d8f', linewidth=2.5, label="Actual GDP")
sns.regplot(x=country_gdp.index, y=country_gdp.values, scatter=False,
            line_kws={"color":"#e76f51","lw":2,"ls":"--","alpha":0.8,"label":"Trend Line"})
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
plt.title(f"{User_Country} GDP (2015–2024)", fontsize=16, fontweight="bold")
plt.xlabel("Year")
plt.ylabel("GDP (millions USD)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
st.pyplot(plt)

# --- GDP Prediction ---
x = country_gdp.index.values
y = country_gdp.values
a, b = np.polyfit(x, y, 1)
Predicted_GDP = a * Year_to_predict + b
st.success(f"💰 Predicted GDP for {User_Country} in {Year_to_predict}: {Predicted_GDP:,.2f} Million USD")

# --- Happiness Plot ---
df_country = dfComb.loc[[User_Country]]
X = df_country["Year"].values.reshape(-1,1)
y = df_country["Happiness_Score"].values
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(df_country["Year"], y, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8, label="Actual Happiness")
ax.plot(df_country["Year"], y_pred, linestyle='--', color='#ff7f0e', linewidth=2.5, label="Trend Line")
ax.set_title(f"Happiness Score for {User_Country} (2015–2024)", fontsize=16, fontweight='bold')
ax.set_xlabel("Year")
ax.set_ylabel("Happiness Score")
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_facecolor('#f7f7f7')
ax.legend()
st.pyplot(fig)

# --- Happiness Prediction ---
a, b = np.polyfit(df_country["Year"], df_country["Happiness_Score"], 1)
predicted_Happiness_score = a * Year_to_predict + b
st.success(f"😊 Predicted Happiness Score for {User_Country} in {Year_to_predict}: {predicted_Happiness_score:.3f}")
