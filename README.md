
<p align="center">
  <img src="images/output_20250526_102343.jpg" alt="Banner" width="600" height="200"/>
</p>

# 📈 Marketing Campaigns Analysis Dashboard

Welcome to the **Marketing Campaigns Analysis Dashboard**! This project provides a comprehensive, interactive, and visually engaging platform for analyzing marketing campaign data, uncovering actionable insights, and optimizing marketing strategies.

An interactive Streamlit web app for analyzing and visualizing marketing campaign performance across various dimensions such as ROI, budget, revenue, conversion rates, channels, campaign types, and target audiences.


---

## 🚀 Project Overview

This repository contains a full data science workflow for marketing campaign analytics, from raw data preprocessing to advanced interactive dashboards. The project leverages Python, Streamlit, Plotly, and modern data science libraries to deliver a robust solution for marketing teams, analysts, and decision-makers.

---

## 🧰 Tech Stack

- **Python**
- **Streamlit** – Interactive frontend
- **Plotly** – Dynamic, interactive charts with dark theme support
- **Seaborn / Matplotlib** – For enhanced statistical visualization
- **Pandas / NumPy / Scikit-learn / Statsmodels** – Data manipulation and modeling
- **Jupyter Notebook (EDA.ipynb)** – Initial exploratory data analysis
 
---

## 💡 Key Features

- **Interactive Dashboard:** Built with Streamlit and Plotly for dynamic data exploration.
- **Comprehensive Filtering:** Slice and dice campaigns by date, channel, type, ROI, budget, revenue, audience, and more.
- **Advanced Visualizations:** Modern bar charts, boxplots, scatterplots, heatmaps, and 3D plots.
- **Statistical Insights:** Correlation analysis, ANOVA, and regression models to uncover drivers of ROI and profitability.
- **Segmentation:** Compare B2B vs B2C, channel performance, campaign types, and budget categories.
- **Seasonality & Trends:** Analyze monthly, quarterly, and yearly patterns to optimize campaign timing.
- **Best Practices & Recommendations:** Executive summary and actionable recommendations for maximizing marketing efficiency.

---

## 🚀 Dashboard Sections

- 📊 **Channel Analysis:** Frequency, ROI, and conversion by marketing channel.
- 💡 **Campaign Type Analysis:** Revenue and conversion by campaign type.
- 💰 **ROI Analysis:** Distribution and drivers of ROI, including outlier detection.
- 🎯 **B2B vs B2C Comparison:** Key metrics and statistical tests between audience segments.
- 🏆 **Most Profitable Campaigns:** Deep dive into top campaigns by net profit.
- 💵 **Budget vs Revenue:** Correlation and regression analysis to find optimal investment levels.
- 📈 **High-Performance Campaigns:** Identify and visualize campaigns with exceptional ROI and revenue.
- 📅 **Seasonality & Temporal Patterns:** Monthly, quarterly, and annual trends.
- 🧠 **General Conclusions:** Executive summary and strategic recommendations.

---

## 🧑‍💻 How to Run

**Clone the repository:**
git clone https://github.com/Victor-Marti/marketing


Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

📝 Data
The dataset marketingcampaigns_clean.csv contains real or simulated campaign information, including:

Budget

Revenue

Conversion Rate

Net Profit

ROI

Channel

Campaign Type

Target Audience

Duration

**Prepare the data:**

Use the notebooks in notebooks/ to clean and preprocess the raw data if needed.
Ensure data/marketingcampaigns_clean.csv exists.
Launch the dashboard:

**📚 Data Science Workflow**

Preprocessing: See notebooks/preprocesamiento.ipynb for detailed data cleaning, outlier handling, and feature engineering.
EDA: Explore notebooks/EDA.ipynb for in-depth exploratory analysis, visualizations, and statistical summaries.
Dashboard: The main dashboard logic and visualizations are implemented in app.py.

**📦 Dependencies**

pandas, numpy, matplotlib, seaborn
scikit-learn, statsmodels, scipy
plotly, streamlit, dash
See requirements.txt for the full list.

**📝 Insights & Recommendations**

Paid channels and social media campaigns consistently deliver the highest ROI.
Efficiency peaks at low-to-mid budget ranges; higher investment does not guarantee better returns.
No significant difference in conversion or ROI between B2B and B2C, but profit margin is higher for B2C.
Seasonal peaks in spring and autumn; Wednesdays and Thursdays are optimal for campaign launches.
Best practices: Focus on operational efficiency, optimize profit margin, and replicate high-performing campaign models.

🤝 Contributing

Pull requests and feedback are welcome. Feel free to fork this repo and propose improvements or new features!

📜 License

This project is licensed under the MIT License.

👨‍💼 Author

Developed by Victor — Data Science Analytics
🔗 [LinkedIn](https://www.linkedin.com/in/victormartic)
