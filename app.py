# --- Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import traceback

# --- Page Config ---
st.set_page_config(
    page_title="Marketing Campaigns Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# --- Title & Description ---
st.title("📈 Marketing Campaigns Analysis Dashboard")
st.markdown("""
This dashboard provides a comprehensive, interactive analysis of marketing campaigns,
including channel performance, campaign types, ROI, audience segmentation, profitability,
budget-revenue relationships, high-performing campaigns, seasonality, and strategic recommendations.
""")

# Global variables for safe initialization
filtered_df = None
df = None

# --- Data Loading ---
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\Victor\Documents\GitHub\marketing\data\marketingcampaigns_clean.csv")
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        st.text(traceback.format_exc())
        return None

# Load data
try:
    with st.spinner('Loading data...'):
        df = load_data()
    st.write(df)

    if df is None or df.empty:
        st.error("No data available.")
        st.stop()

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")
    min_date = df['start_date'].min().date()
    max_date = df['end_date'].max().date()
    date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['start_date'].dt.date >= start_date) & (df['end_date'].dt.date <= end_date)].copy()
    else:
        filtered_df = df.copy()

    # Channel filter
    channels = filtered_df['channel'].unique().tolist()
    selected_channels = st.sidebar.multiselect("Channels", options=channels, default=channels)
    filtered_df = filtered_df[filtered_df['channel'].isin(selected_channels)]

    # Type filter
    types = filtered_df['type'].unique().tolist()
    selected_types = st.sidebar.multiselect("Campaign Types", options=types, default=types)
    filtered_df = filtered_df[filtered_df['type'].isin(selected_types)]

    # ROI filter
    roi_min, roi_max = float(filtered_df['calculated_roi'].min()), float(filtered_df['calculated_roi'].max())
    roi_range = st.sidebar.slider("ROI Range", min_value=roi_min, max_value=roi_max, value=(roi_min, roi_max), step=100.0)
    filtered_df = filtered_df[(filtered_df['calculated_roi'] >= roi_range[0]) & (filtered_df['calculated_roi'] <= roi_range[1])]

    # Budget filter
    min_budget, max_budget = st.sidebar.slider(
    "budget range",
    min_value=float(df['budget'].min()),
    max_value=float(df['budget'].max()),
    value=(float(df['budget'].min()), float(df['budget'].max())),
    step=1000.00
    )
    filtered_df = filtered_df[(filtered_df['budget'] >= min_budget) & (filtered_df['budget'] <= max_budget)]

    # Revenue filter
    min_revenue, max_revenue = st.sidebar.slider(
    "Revenue range",
    min_value=float(df['revenue'].min()),
    max_value=float(df['revenue'].max()),
    value=(float(df['revenue'].min()), float(df['revenue'].max())),
    step=5000.00
    )
    filtered_df = filtered_df[(filtered_df['revenue'] >= min_revenue) & (filtered_df['revenue'] <= max_revenue)]

    # Net Profit filter
    min_net_profit, max_net_profit = st.sidebar.slider(
    "Net Profit range",
    min_value=float(df['net_profit'].min()),
    max_value=float(df['net_profit'].max()),
    value=(float(df['net_profit'].min()), float(df['net_profit'].max())),
    step=5000.00
    )
    filtered_df = filtered_df[(filtered_df['net_profit'] >= min_net_profit) & (filtered_df['net_profit'] <= max_net_profit)]

    # Conversion rate filter
    min_conversion_rate, max_conversion_rate = st.sidebar.slider(
    "Conversion rate range",
    min_value=float(df['conversion_rate'].min()),
    max_value=float(df['conversion_rate'].max()),
    value=(float(df['conversion_rate'].min()), float(df['conversion_rate'].max())),
    step=0.1
    )
    filtered_df = filtered_df[(filtered_df['conversion_rate'] >= min_conversion_rate) & (filtered_df['conversion_rate'] <= max_conversion_rate)]

    # Target Audience filter
    target_audience = df['target_audience'].unique().tolist()
    selected_target_audience = st.sidebar.multiselect(
    "Target Audiences",
    options=target_audience,
    default=target_audience
    )
    if selected_target_audience:
        filtered_df = filtered_df[filtered_df['target_audience'].isin(selected_target_audience)]

    # Revenue category filter
    revenue_category = df['revenue_category'].unique().tolist()
    selected_revenue_category = st.sidebar.multiselect(
    "Revenue category",
    options=revenue_category,
    default=revenue_category
    )
    if selected_revenue_category:
        filtered_df = filtered_df[filtered_df['revenue_category'].isin(selected_revenue_category)]

    st.sidebar.metric("Selected campaigns", len(filtered_df))

    # Advanced options in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("Advanced Options")

    show_clusters = st.sidebar.checkbox("Show Cluster Analysis", value=False)
    show_advanced_charts = st.sidebar.checkbox("Show Advanced Charts", value=False)

    # Check if there is data after filtering
    if len(filtered_df) == 0:
        st.warning("No data available with the selected filters. Please adjust the filters.")
    else:
        # Main tabs to organize the dashboard

        # --- Tabs ---
        tabs = st.tabs([
            "📉Channel Analysis",
            "📊Revenue & Conversion by Campaign Type",
            "💵ROI Analysis",
            "🧩B2B vs B2C Comparison",
            "💰Most Profitable Campaigns",
            "📈Budget vs Revenue Correlation",
            "🏆High ROI & Revenue Campaigns",
            "📅Seasonality & Temporal Patterns",
            "🧠General Conclusion"
        ])

        # --- 1. Channel Analysis ---
        with tabs[0]:
            st.header("Channel-Based Analysis")
            st.markdown("**This section explores the distribution, ROI, and comparative metrics of marketing channels.**")

            # Frequency by channel
            st.subheader("Channel Frequency")
            channel_counts = filtered_df['channel'].value_counts()
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.barplot(x=channel_counts.index, y=channel_counts.values, palette='viridis', ax=ax)
            ax.set_title('Channel Usage Frequency')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Number of Campaigns')
            for i, v in enumerate(channel_counts.values):
                ax.text(i, v + 5, str(v), ha='center')
            st.pyplot(fig)
            st.info("Promotion is the most frequent channel, followed by referral, paid, and organic. "
"Promotion leads in campaign count (221), but paid achieves the highest average ROI. "
"All channels show a balanced distribution, indicating a well-diversified marketing strategy. "
"Referral and organic channels, while less frequent, still contribute significantly to overall performance.")

            # ROI by channel (boxplot)
            st.subheader("ROI Distribution by Channel")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x='channel', y='calculated_roi', data=filtered_df, palette='Set1', ax=ax)
            ax.set_title('ROI Distribution by Channel')
            ax.set_xlabel('Channel')
            ax.set_ylabel('ROI')
            st.pyplot(fig)
            st.info("ROI distribution is homogeneous across channels. "
"Paid stands out with the highest median ROI (894.45), but all channels have similar interquartile ranges. "
"Paid also shows the lowest variability, suggesting more predictable results, while other channels have greater dispersion. "
"This indicates that channel selection alone does not guarantee higher ROI—other factors such as campaign type and targeting are also important.")

            # Channel metrics table
            st.subheader("Channel Performance Metrics")
            channel_metrics = filtered_df.groupby('channel').agg({
                'calculated_roi': 'mean',
                'conversion_rate': 'mean',
                'profit_margin': 'mean',
                'revenue': 'mean',
                'budget': 'mean',
                'campaign_name': 'count'
            }).rename(columns={'campaign_name': 'num_campaigns'}).reset_index()
            st.dataframe(channel_metrics, use_container_width=True)
            st.info("Paid channel has the highest average ROI, but all channels are close. "
"Paid also leads in conversion rate and profit margin, confirming its efficiency. "
"Promotion and organic channels are nearly as effective, while referral lags slightly in ROI and conversion. "
"The top three channels by ROI are paid, promotion, and organic, while referral is the lowest.")

            # ROI bar chart by channel
            st.subheader("Average ROI by Channel")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='channel', y='calculated_roi', data=channel_metrics, palette='viridis', ax=ax)
            for i, v in enumerate(channel_metrics['calculated_roi']):
                ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
            ax.set_title('Average ROI by Channel')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Average ROI')
            st.pyplot(fig)
            st.info("The bar chart confirms paid as the channel with the highest average ROI, closely followed by promotion and organic. "
"All four channels fluctuate between 806 and 894 in average ROI, indicating a balanced performance across the marketing mix.")

            # ROI by channel and type (boxplot)
            st.subheader("ROI by Channel and Campaign Type")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='channel', y='calculated_roi', hue='type', data=filtered_df, showfliers=False, ax=ax)
            ax.set_title('ROI by Channel and Campaign Type')
            ax.set_xlabel('Channel')
            ax.set_ylabel('ROI')
            ax.legend(title='Campaign Type', loc='upper right')
            st.pyplot(fig)
            st.info("Paid channel shows the most stable ROI across types, with 50% of values between 450 and 1400. "
"Other channels display greater variability, but the central 50% of all categories are between ROI 300 and 1500. "
"This suggests that paid campaigns are more consistent, while other channels may offer higher upside but also more risk.")

            # ROI by channel and audience (boxplot)
            st.subheader("ROI by Channel and Target Audience")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(x='channel', y='calculated_roi', hue='target_audience', data=filtered_df, showfliers=False, ax=ax)
            ax.set_title('ROI by Channel and Target Audience')
            ax.set_xlabel('Channel')
            ax.set_ylabel('ROI')
            ax.legend(title='Target Audience')
            st.pyplot(fig)
            st.info("The top 25% ROI values are highly dispersed across audiences for all channels. "
"This means that more than half of the audience-channel combinations fall below the overall mean ROI (849.4). "
"There is no clear audience-channel combination that consistently outperforms others, highlighting the need for tailored strategies.")

            # ROI vs Budget scatter
            st.subheader("ROI vs Budget by Channel")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(x='budget', y='calculated_roi', hue='channel', size='revenue', sizes=(50, 300), alpha=0.7, data=filtered_df, ax=ax)
            ax.set_xlim(0, 105000)
            ax.set_ylim(-300, 5000)
            ax.set_title('Budget vs ROI by Channel')
            ax.set_xlabel('Budget')
            ax.set_ylabel('ROI')
            st.pyplot(fig)
            st.info("There is a slight negative trend between budget and ROI (correlation ≈ -0.13): "
"higher budgets tend to have lower ROI, but the correlation is weak. "
"Smaller budgets (<50,000) show greater ROI variability, with some campaigns reaching up to 5,000 ROI. "
"Larger budgets (>75,000) are more stable but generally below 2,000 ROI. "
"High-ROI campaigns do not necessarily generate the highest revenue, and all channels show similar ROI distributions.")

            # Channel conclusions
            st.markdown("""
            **Conclusions:**  
            - Promotion is the most used channel, but paid has the highest average ROI.  
            - ROI is distributed homogeneously across channels.  
            - Higher budgets do not guarantee higher ROI.  
            - Social media campaigns in paid channels are the most efficient.
            """)

        # --- 2. Revenue & Conversion by Campaign Type ---
        with tabs[1]:
            st.header("Revenue and Conversion by Campaign Type")
            st.markdown("**This section analyzes revenue and conversion rates by campaign type.**")

            # Revenue by type (bar)
            campaign_metrics = filtered_df.groupby('type').agg({
                'revenue': 'mean',
                'conversion_rate': 'mean',
                'calculated_roi': 'mean',
                'net_profit': 'mean',
                'campaign_name': 'count'
            }).reset_index().rename(columns={'campaign_name': 'count'})
            campaign_metrics_by_revenue = campaign_metrics.sort_values('revenue', ascending=False)
            st.subheader("Average Revenue by Campaign Type")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='type', y='revenue', data=campaign_metrics_by_revenue, palette='viridis', ax=ax)
            for i, v in enumerate(campaign_metrics_by_revenue['revenue']):
                ax.text(i, v + 0.01, f'{v:,.2f}', ha='center')
            ax.set_title('Average Revenue by Campaign Type')
            ax.set_xlabel('Campaign Type')
            ax.set_ylabel('Average Revenue')
            st.pyplot(fig)
            st.info("Revenue is homogeneous across most campaign types, with the exception of 'event', which is significantly lower. "
    "Most types achieve average revenues between $450,000 and $480,000. "
    "This suggests that, except for events, all campaign types are capable of generating substantial revenue, providing flexibility in campaign planning.")

            # Conversion rate by type (bar)
            campaign_metrics_by_conv = campaign_metrics.sort_values('conversion_rate', ascending=False)
            st.subheader("Average Conversion Rate by Campaign Type")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='type', y='conversion_rate', data=campaign_metrics_by_conv, palette='viridis', ax=ax)
            for i, v in enumerate(campaign_metrics_by_conv['conversion_rate']):
                ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
            ax.set_title('Average Conversion Rate by Campaign Type')
            ax.set_xlabel('Campaign Type')
            ax.set_ylabel('Conversion Rate')
            st.pyplot(fig)
            st.info("Conversion rates are also homogeneous, ranging from 0.52 to 0.55 across most types. "
    "This indicates that the probability of converting leads is stable regardless of campaign type, with no type showing a clear advantage in conversion efficiency.")

            # Revenue vs Conversion scatter
            st.subheader("Revenue vs Conversion Rate by Campaign Type")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='conversion_rate', y='revenue', size='count', sizes=(100, 500), hue='calculated_roi', data=campaign_metrics, palette='viridis', ax=ax)
            for i, row in campaign_metrics.iterrows():
                ax.text(row['conversion_rate']+0.02, row['revenue'], row['type'], fontsize=11)
            ax.set_title('Revenue vs Conversion Rate by Campaign Type')
            ax.set_xlabel('Conversion Rate')
            ax.set_ylabel('Average Revenue')
            st.pyplot(fig)
            st.info("'Podcast' campaigns achieve the highest average revenue, despite having a similar conversion rate to other types. "
    "'Social media' stands out with the highest average ROI (most intense color), while 'event' campaigns are the lowest in both revenue and conversion. "
    "The size of the circles shows that 'social media' and 'email' are the most common types. "
    "Overall, the distribution is homogeneous, but small differences may indicate specific optimization opportunities.")

            # Key metrics table
            st.subheader("Key Metrics by Campaign Type")
            st.dataframe(campaign_metrics, use_container_width=True)

            # Net profit by type (bar)
            campaign_metrics_by_profit = campaign_metrics.sort_values('net_profit', ascending=False)
            st.subheader("Average Net Profit by Campaign Type")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='type', y='net_profit', data=campaign_metrics_by_profit, palette='viridis', ax=ax)
            for i, v in enumerate(campaign_metrics_by_profit['net_profit']):
                ax.text(i, v, f'{v:,.2f}', ha='center', va='bottom', fontsize=12)
            ax.set_title('Average Net Profit by Campaign Type')
            ax.set_xlabel('Campaign Type')
            ax.set_ylabel('Net Profit')
            st.pyplot(fig)
            st.info("'Social media', 'email', and 'podcast' types are the most homogeneous and profitable in terms of net profit. "
    "'Webinar' is slightly below, while 'event' is much lower, making it the least profitable option.")

            # ROI vs Net Profit scatter
            campaign_metrics_filtered = campaign_metrics[campaign_metrics['type'] != 'event']
            st.subheader("ROI vs Net Profit by Campaign Type")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='calculated_roi', y='net_profit', size='count', sizes=(100, 500), hue='type', data=campaign_metrics_filtered, palette='viridis', ax=ax)
            ax.set_title('ROI vs Net Profit by Campaign Type')
            ax.set_xlabel('Average ROI')
            ax.set_ylabel('Average Net Profit')
            st.pyplot(fig)
            st.info("The top three campaign types show a homogeneous distribution in both ROI and net profit, confirming their consistency and reliability. "
    "This highlights the strategic value of focusing on these types for sustained profitability.")

            # Efficiency: profit per dollar
            campaign_metrics['budget'] = filtered_df.groupby('type')['budget'].mean().reset_index()['budget']
            campaign_metrics['profit_per_dollar'] = campaign_metrics['net_profit'] / campaign_metrics['budget']
            campaign_metrics_by_efficiency = campaign_metrics.sort_values('profit_per_dollar', ascending=False)
            st.subheader("Net Profit per Dollar Invested by Campaign Type")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='type', y='profit_per_dollar', data=campaign_metrics_by_efficiency, palette='viridis', ax=ax)
            for i, v in enumerate(campaign_metrics_by_efficiency['profit_per_dollar']):
                ax.text(i, v, f'{v:,.2f}', ha='center', va='bottom', fontsize=12)
            ax.set_title('Net Profit per Dollar Invested by Campaign Type')
            ax.set_xlabel('Campaign Type')
            ax.set_ylabel('Net Profit per Dollar')
            st.pyplot(fig)
            st.info("'Social media' campaigns generate the highest net profit per dollar invested (7.55), followed by 'email' (7.17) and 'podcast' (7.10). "
    "'Webinar' is lower (6.81), and 'event' is the least efficient (1.40). "
    "This metric reinforces the recommendation to prioritize social media and email campaigns for maximum efficiency.")

            # Complete metrics table
            st.subheader("Complete Metrics by Campaign Type")
            metrics_columns = ['type', 'count', 'budget', 'revenue', 'net_profit', 'calculated_roi', 'conversion_rate', 'profit_per_dollar']
            st.dataframe(campaign_metrics[metrics_columns], use_container_width=True)

            # Correlation heatmap
            st.subheader("Correlation between Key Metrics")
            correlation = campaign_metrics[['budget', 'revenue', 'net_profit', 'calculated_roi', 'conversion_rate', 'profit_per_dollar']].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation, annot=True, cmap='viridis', fmt='.2f', ax=ax)
            ax.set_title('Correlation between Financial and Performance Metrics')
            st.pyplot(fig)
            st.info("Net profit is highly correlated with revenue, and profit per dollar is strongly correlated with calculated ROI. "
    "These relationships provide valuable insights for optimizing budget allocation and maximizing marketing performance.")

            # Conclusions
            st.markdown("""
            **Conclusions:**  
            - 'Social media', 'email', and 'podcast' are the most consistent and profitable types.  
            - Net profit and ROI are highly correlated.  
            - Social media campaigns are the most efficient per dollar invested.
            """)

        # --- 3. ROI Analysis ---
        with tabs[2]:
            st.header("ROI Analysis in Marketing Campaigns")
            st.markdown("**This section explores the distribution and drivers of ROI.**")

            # ROI distribution (histogram, boxplot, QQ plot)
            st.subheader("ROI Distribution")
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            sns.histplot(filtered_df['calculated_roi'], kde=True, bins=30, ax=axs[0, 0])
            axs[0, 0].set_title('ROI Distribution')
            sns.boxplot(y=filtered_df['calculated_roi'], ax=axs[0, 1])
            axs[0, 1].set_title('ROI Boxplot')
            stats.probplot(filtered_df['calculated_roi'], dist="norm", plot=axs[1, 0])
            axs[1, 0].set_title('ROI Q-Q Plot')
            axs[1, 1].axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            st.info("ROI is right-skewed with a high mean (approx. 849), but with considerable dispersion and several high-value outliers. "
    "The distribution is not normal (confirmed by the Q-Q plot), and most campaigns achieve moderate ROI, with a few achieving exceptionally high returns. "
    "This suggests that while most campaigns perform within a predictable range, there are rare cases of outstanding performance that can significantly impact overall results. "
    "Segmenting campaigns by ROI reveals clear groups of high, medium, and low performers, highlighting the importance of identifying and replicating the factors behind top-performing campaigns.")

            # ROI by main categories (boxplots)
            st.subheader("ROI by Main Categories")
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            sns.boxplot(x='channel', y='calculated_roi', data=filtered_df, palette='viridis', showfliers=False, ax=axs[0, 0])
            axs[0, 0].set_title('ROI by Channel')
            sns.boxplot(x='type', y='calculated_roi', data=filtered_df, palette='Set1', showfliers=False, ax=axs[0, 1])
            axs[0, 1].set_title('ROI by Campaign Type')
            sns.boxplot(x='target_audience', y='calculated_roi', data=filtered_df, palette='Set2', showfliers=False, ax=axs[1, 0])
            axs[1, 0].set_title('ROI by Target Audience')
            sns.boxplot(x='budget', y='calculated_roi', data=filtered_df, palette='Set3', showfliers=False, ax=axs[1, 1])
            axs[1, 1].set_title('ROI by Budget')
            plt.tight_layout()
            st.pyplot(fig)
            st.info("ROI is distributed homogeneously across channels, with 'paid' showing the highest mean and lowest variability. "
    "'Social media', 'email', and 'podcast' campaign types are the most consistent and profitable, while 'event' is excluded due to poor performance. "
    "No substantial differences are observed between B2B and B2C audiences, but B2C shows slightly higher variability. "
    "There is a slight inverse relationship between budget and ROI: lower and mid-range budgets tend to achieve higher ROI, while very high budgets show lower efficiency. "
    "These patterns suggest that focusing on efficient channels and campaign types, and optimizing budget allocation, can maximize ROI.")

            # Correlation heatmap (ROI vs numeric variables)
            st.subheader("Correlation between ROI and Numeric Variables")
            numeric_cols = ['calculated_roi', 'budget', 'conversion_rate', 'revenue', 'net_profit', 'campaign_duration', 'cost_per_conversion', 'profit_margin']
            correlation = filtered_df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, ax=ax)
            ax.set_title('Correlation between ROI and Numeric Variables')
            st.pyplot(fig)
            st.info("ROI is highly positively correlated with net profit (r ≈ 0.71) and profit margin, and negatively correlated with cost per conversion (-0.24) and budget (-0.47). "
    "This means that maximizing profit margin and reducing cost per conversion are key to improving ROI, while simply increasing budget does not guarantee better results. "
    "Conversion rate has a weak but positive correlation with ROI, indicating that improving conversion can have a tangible, though not linear, impact on returns. "
    "These insights support a strategy focused on operational efficiency and targeted investment rather than increasing spend indiscriminately.")

            # Conclusions
            st.markdown("""
            **Conclusions:**  
            - ROI is positively correlated with net profit and profit margin, and negatively with cost per conversion and budget.  
            - Social media campaigns and paid channels are the most efficient.  
            - Higher budgets do not guarantee higher ROI.
            """)

        # --- 4. B2B vs B2C Comparison ---
        with tabs[3]:
            st.header("Comparative Analysis: B2B vs B2C")
            st.markdown("**This section compares key metrics between B2B and B2C audiences.**")

            # Conversion rate boxplot
            st.subheader("Conversion Rate: B2B vs B2C")
            filtered_audience = filtered_df[filtered_df['target_audience'].isin(['B2B', 'B2C'])]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x='target_audience', y='conversion_rate', data=filtered_audience, showfliers=False, ax=ax)
            ax.set_title('Conversion Rate: B2B vs B2C')
            st.pyplot(fig)
            st.info("Conversion rates are nearly identical for B2B and B2C, with both segments showing a mean around 0.54 and virtually overlapping distributions. "
    "Statistical tests (t-test, p-value ≈ 0.94) confirm there is no significant difference between the two audiences. "
    "This suggests that audience type alone does not determine conversion success, and any observed differences are likely due to random variation.")

            # Key metrics by audience (boxplots)
            st.subheader("Key Metrics by Audience")
            metrics = ['calculated_roi', 'conversion_rate', 'profit_margin', 'revenue', 'budget', 'cost_per_conversion']
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            for i, metric in enumerate(metrics):
                sns.boxplot(x='target_audience', y=metric, data=filtered_audience, ax=axes[i], showfliers=False, palette='viridis')
                axes[i].set_title(f'{metric} Comparison')
            plt.tight_layout()
            st.pyplot(fig)
            st.info("All key metrics are highly similar between B2B and B2C, except for profit margin, which is about 27% higher for B2C. "
    "Cost per conversion is slightly higher for B2C, possibly explaining the higher profit margin. "
    "Overall, both segments have comparable ROI, conversion rate, revenue, and budget, with overlapping interquartile ranges and similar variability.")

            # Conversion rate by channel and audience (bar)
            st.subheader("Conversion Rate by Channel and Audience")
            channel_conversion = filtered_df.groupby(['target_audience', 'channel'])['conversion_rate'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='channel', y='conversion_rate', hue='target_audience', data=channel_conversion, ax=ax)
            ax.set_title('Conversion Rate by Channel and Audience')
            st.pyplot(fig)
            st.info("B2C outperforms B2B in the paid channel (0.61 vs 0.53), while B2B leads in organic (0.57 vs 0.52) and email. "
    "Promotion and referral channels show minimal differences between audiences. "
    "These results highlight that channel-audience combinations can influence conversion, even if overall averages are similar.")

            # Conversion rate by type and audience (bar)
            st.subheader("Conversion Rate by Campaign Type and Audience")
            type_conversion = filtered_df.groupby(['target_audience', 'type'])['conversion_rate'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x='type', y='conversion_rate', hue='target_audience', data=type_conversion, ax=ax)
            ax.set_title('Conversion Rate by Campaign Type and Audience')
            st.pyplot(fig)
            st.info(" Social media campaigns have identical conversion rates for B2B and B2C (0.55). "
    "B2C surprisingly outperforms B2B in webinars (0.58 vs 0.52), while B2B leads slightly in email. "
    "Event campaigns are ineffective for both audiences, especially B2B. "
    "These patterns suggest that campaign type can interact with audience to produce nuanced results.")

            # Heatmap: conversion by channel and audience
            st.subheader("Conversion Rate Heatmap: Channel vs Audience")
            heatmap_data = filtered_df.pivot_table(values='conversion_rate', index='channel', columns='target_audience', aggfunc='mean')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f', linewidths=.5, ax=ax)
            ax.set_title('Conversion Rate by Channel and Audience')
            st.pyplot(fig)
            st.info("he heatmap confirms natural specialization: B2B excels in organic channels, while B2C dominates paid. "
    "Promotion and referral channels are less dependent on audience type. "
    "These findings reinforce the importance of aligning channel strategy with audience characteristics for optimal conversion.")

            # Conclusions
            st.markdown("""
            **Conclusions:**  
            - No significant differences in conversion or ROI between B2B and B2C.  
            - B2C performs better in paid and webinar channels; B2B leads in organic and email.  
            - Profit margin is higher for B2C.
            """)

        # --- 5. Most Profitable Campaigns ---
        with tabs[4]:
            st.header("Most Profitable Campaigns")
            st.markdown("**This section highlights the top campaigns by net profit and analyzes their characteristics.**")

            # Top 10 by net profit
            st.subheader("Top 10 Campaigns by Net Profit")
            filtered_df['net_profit'] = filtered_df['revenue'] - filtered_df['budget']
            top_campaigns = filtered_df.sort_values('net_profit', ascending=False).head(10)
            st.dataframe(top_campaigns[['campaign_name', 'net_profit', 'calculated_roi', 'revenue', 'budget', 'channel', 'type', 'conversion_rate', 'campaign_duration', 'target_audience']], use_container_width=True)
            st.info("The most profitable campaigns achieve net profits well above the dataset average, with the top campaign exceeding $900,000. "
    "These campaigns combine high conversion rates, efficient budget allocation, and optimal duration. "
    "Most are concentrated in the B2B segment and leverage channels such as email and organic, confirming their strategic value for profitability. "
    "Compared to the least profitable campaigns, top performers operate with lower budgets but generate much higher revenues, demonstrating exceptional ROI efficiency.")

            # ROI vs Net Profit bar
            st.subheader("ROI vs Net Profit")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='campaign_name', y='calculated_roi', data=top_campaigns, palette='plasma', ax=ax)
            ax.set_title('ROI of Top 10 Campaigns by Net Profit')
            ax.set_ylabel('ROI')
            ax.set_xlabel('Campaign Name')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            st.info("The top 10 campaigns by net profit also show high ROI, but there is not always a direct relationship: "
    "some campaigns achieve outstanding ROI with moderate net profit, while others balance both metrics. "
    "This highlights the importance of not only maximizing ROI but also ensuring substantial absolute returns.")

            # Distribution of campaign types (pie)
            st.subheader("Distribution of Campaign Types")
            campaign_types = top_campaigns['type'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(campaign_types, labels=campaign_types.index, autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribution of Successful Campaign Types')
            st.pyplot(fig)
            st.info("The most profitable campaigns are dominated by email and organic types, with social media and podcast also present. "
    "This distribution suggests that while social media campaigns can be highly effective, email and organic strategies are more consistently represented among top performers. "
    "Event campaigns are notably absent, confirming their lower profitability.")

            # Correlation with net profit (heatmap)
            st.subheader("Correlation with Net Profit")
            correlation_columns = ['net_profit', 'budget', 'conversion_rate', 'calculated_roi', 'campaign_duration', 'revenue']
            correlation_matrix = filtered_df[correlation_columns].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            ax.set_title('Correlation: Factors Influencing Net Profit')
            st.pyplot(fig)
            st.info("Net profit is highly correlated with revenue and conversion rate, and moderately with ROI. "
    "Budget is not a strong predictor of net profit, reinforcing that efficient allocation and high conversion are more critical than simply increasing spend. "
    "Campaign duration shows a weak relationship, suggesting that optimal timing is less important than targeting and execution quality.")

            # Conclusions
            st.markdown("""
            **Conclusions:**  
            - High conversion rates and efficient budgets drive profitability.  
            - Email and organic channels dominate among top campaigns.  
            - B2B segment is more profitable on average.
            """)

        # --- 6. Budget vs Revenue Correlation ---
        with tabs[5]:
            st.header("Budget vs Revenue Correlation")
            st.markdown("**This section analyzes the relationship between budget and revenue, including polynomial regression and ROI efficiency.**")

            # Pearson/Spearman correlation
            st.subheader("Correlation Calculation")
            pearson_corr, p_value = stats.pearsonr(filtered_df['budget'], filtered_df['revenue'])
            spearman_corr, sp_p_value = stats.spearmanr(filtered_df['budget'], filtered_df['revenue'])
            st.write(f"Pearson correlation: {pearson_corr:.4f} (p={p_value:.4f})")
            st.write(f"Spearman correlation: {spearman_corr:.4f} (p={sp_p_value:.4f})")
            st.info(
    "There is a moderate positive correlation between budget and revenue (Pearson ≈ 0.12), statistically significant but not strong. "
    "This means that increasing the budget generally leads to higher revenue, but the relationship is not linear and shows diminishing returns at higher investment levels. "
    "Paid campaigns show the most consistent trend, while organic campaigns are more dispersed, indicating less predictable results. "
    "Doubling the budget does not necessarily double the revenue, especially at higher investment levels."
)

            # Polynomial regression
            st.subheader("Polynomial Regression: Budget vs Revenue")
            X = filtered_df[['budget']]
            y = filtered_df['revenue']
            degree = 2
            poly = PolynomialFeatures(degree=degree)
            X_poly = poly.fit_transform(X)
            poly_model = LinearRegression()
            poly_model.fit(X_poly, y)
            y_poly_pred = poly_model.predict(X_poly)
            sorted_idx = np.argsort(filtered_df['budget'])
            sorted_budget = filtered_df['budget'].iloc[sorted_idx]
            sorted_poly_pred = y_poly_pred[sorted_idx]
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(x='budget', y='revenue', data=filtered_df, alpha=0.6, ax=ax)
            ax.plot(sorted_budget, sorted_poly_pred, color='green', linewidth=2, label=f'Polynomial Regression (degree {degree})')
            ax.set_title('Budget vs Revenue: Polynomial Model')
            ax.set_xlabel('Budget')
            ax.set_ylabel('Revenue')
            ax.legend()
            st.pyplot(fig)
            st.info(
    "The polynomial regression curve clearly shows diminishing marginal returns: "
    "as budget increases, revenue grows at a decreasing rate, especially at high investment levels. "
    "The optimal investment range, where each additional dollar has the greatest impact, is between $30,000 and $70,000. "
    "Both very low and very high budgets are less efficient, confirming the existence of a saturation point."
)

            # By campaign type
            st.subheader("Budget vs Revenue by Campaign Type")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(x='budget', y='revenue', data=filtered_df, alpha=0.6, hue='type', ax=ax)
            ax.plot(sorted_budget, sorted_poly_pred, color='green', linewidth=2, label=f'Polynomial Regression (degree {degree})')
            ax.set_title('Budget vs Revenue by Campaign Type')
            ax.set_xlabel('Budget')
            ax.set_ylabel('Revenue')
            ax.legend()
            st.pyplot(fig)
            st.info(
    "The relationship between budget and revenue varies by campaign type. "
    "Podcast, social media, and webinar campaigns show the strongest positive correlations, "
    "while email campaigns show almost no correlation, indicating that budget is not a key driver for email performance. "
    "This highlights the need for differentiated budget strategies by campaign type."
)

            # Correlation matrix
            st.subheader("Correlation Matrix")
            correlation_vars = ['budget', 'revenue', 'calculated_roi', 'conversion_rate', 'campaign_duration']
            corr_matrix = filtered_df[correlation_vars].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            st.info(
    "The correlation matrix confirms that budget and revenue are moderately correlated, "
    "but budget is negatively correlated with ROI. "
    "Campaign duration and conversion rate show weak relationships with both budget and revenue, "
    "suggesting that other factors are more important for maximizing efficiency."
)

            # ROI vs Budget scatter
            st.subheader("ROI vs Budget by Channel")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.scatterplot(x='budget', y='calculated_roi', data=filtered_df, alpha=0.6, hue='channel', s=80, ax=ax)
            lowess = sm.nonparametric.lowess(filtered_df['calculated_roi'], filtered_df['budget'], frac=0.3)
            ax.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=3, label='LOWESS Trend')
            ax.set_title('Budget vs ROI by Channel')
            ax.set_xlabel('Budget')
            ax.set_ylabel('ROI')
            ax.legend()
            st.pyplot(fig)
            st.info(
    "There is a moderate inverse relationship between budget and ROI (correlation ≈ -0.57): "
    "higher budgets tend to have lower ROI, especially above 70,000. "
    "The most efficient campaigns are concentrated in the low and mid budget segments, "
    "while high-budget campaigns show lower and less variable ROI. "
    "This supports the strategy of focusing investments in the 30,000 - 70,000 range for optimal efficiency."
)

            # Revenue by budget segment (boxplot)
            st.subheader("Revenue by Budget Segment")
            filtered_df['budget_segment'] = pd.qcut(filtered_df['budget'], 4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='budget_segment', y='revenue', data=filtered_df, palette='GnBu', ax=ax)
            ax.set_title('Revenue by Budget Segment')
            st.pyplot(fig)
            st.info(
    "Average and median revenue increase with higher budget segments, "
    "but the variability is also greater in the high-budget group. "
    "The relationship is not linear: while higher budgets often yield higher absolute revenue, "
    "the efficiency per dollar invested decreases at the top end. "
    "The optimal revenue segments are mid-high and high, but with careful attention to efficiency."
)

            # ROI by budget segment (boxplot)
            st.subheader("ROI by Budget Segment")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='budget_segment', y='calculated_roi', data=filtered_df, palette='Accent', ax=ax)
            ax.set_title('ROI by Budget Segment')
            st.pyplot(fig)
            st.info(
    "ROI is highest in the low and mid-low budget segments, "
    "with both mean and median above the global average. "
    "High-budget campaigns show lower and less dispersed ROI, confirming that efficiency is maximized in moderate investment ranges. "
    "This reinforces the recommendation to avoid over-investment and focus on optimizing mid-range budgets."
)

            # Conclusions
            st.markdown("""
            **Conclusions:**  
            - Budget and revenue are moderately correlated, but with diminishing returns at higher budgets.  
            - ROI decreases as budget increases; efficiency is highest in low and mid segments.
            """)

        # --- 7. High ROI & Revenue Campaigns ---
        with tabs[6]:
            st.header("Campaigns with ROI > 0.5 and Revenue > 500,000")
            st.markdown("**This section identifies and analyzes high-performing campaigns.**")

            # Filter and table
            high_performance = filtered_df[(filtered_df['calculated_roi'] > 0.5) & (filtered_df['revenue'] > 500000)].copy()
            st.subheader("High-Performance Campaigns Table")
            st.dataframe(high_performance[['campaign_name', 'calculated_roi', 'revenue', 'budget', 'type', 'channel', 'conversion_rate']], use_container_width=True)
            st.info(
    "These high-performance campaigns represent the top quartile in both ROI and revenue. "
    "They achieve a unique balance of high returns and substantial income, with conversion rates and efficiency well above the dataset average. "
    "The average conversion rate and ROI in this group are significantly higher than the overall mean, confirming their standout status. "
    "These campaigns serve as benchmarks for best practices in marketing strategy."
)

            # Distribution by type and channel
            st.subheader("Distribution by Campaign Type")
            st.dataframe(high_performance['type'].value_counts().reset_index().rename(columns={'index': 'Type', 'type': 'Count'}))
            st.subheader("Distribution by Channel")
            st.dataframe(high_performance['channel'].value_counts().reset_index().rename(columns={'index': 'Channel', 'channel': 'Count'}))
            st.info(
    "Social media campaigns dominate the high ROI-high revenue segment, followed by email and podcast. "
    "Paid and organic channels are the most frequent among top performers, while event campaigns are almost absent. "
    "This distribution highlights the effectiveness of digital-first strategies and the limited impact of event-based marketing in achieving both high ROI and revenue."
)

            # ROI vs Revenue scatter
            st.subheader("ROI vs Revenue Quadrant")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=high_performance, x='revenue', y='calculated_roi', size='budget', hue='type', sizes=(50, 400), alpha=0.7, ax=ax)
            ax.set_title('High-Performance Campaigns: ROI vs Revenue')
            ax.set_xlabel('Revenue')
            ax.set_ylabel('ROI')
            st.pyplot(fig)
            st.info(
    "The scatter plot shows that high ROI and high revenue are not always achieved simultaneously, but social media campaigns are most likely to reach both. "
    "Moderate budgets (not the highest) are often associated with optimal performance, indicating that efficiency, not just investment, is key. "
    "There are also outliers with exceptional ROI but moderate revenue, suggesting that niche targeting and operational excellence can yield outstanding results even without massive budgets."
)

            # Conclusions
            st.markdown("""
            **Conclusions:**  
            - Social media campaigns dominate the high ROI-high revenue quadrant.  
            - Moderate budgets can achieve optimal performance.  
            - High efficiency is not exclusive to high budgets.
            """)

        # --- 8. Seasonality & Temporal Patterns ---
        with tabs[7]:
            st.header("Seasonality and Temporal Patterns")
            st.markdown("**This section explores monthly, quarterly, and annual trends in campaign performance.**")

            # Monthly performance
            filtered_df['start_month'] = filtered_df['start_date'].dt.month
            filtered_df['start_month_name'] = filtered_df['start_date'].dt.month.map({
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            })
            monthly_performance = filtered_df.groupby('start_month_name')[['revenue', 'calculated_roi', 'conversion_rate']].mean()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            monthly_performance = monthly_performance.reindex(month_order)
            st.subheader("Monthly Performance Trends")
            fig, ax = plt.subplots(figsize=(14, 6))
            monthly_performance.plot(ax=ax)
            ax.set_title('Monthly Performance Trends')
            ax.set_xlabel('Month')
            st.pyplot(fig)
            st.info(
    "There are two clear performance peaks: spring (April-May) and autumn (September-October). "
    "These months show the highest average ROI and conversion rates, aligning with key business and consumer cycles. "
    "Social media campaigns maintain stable performance year-round, while email and webinar types are more sensitive to seasonality. "
    "February and November are also strong for B2C, while B2B peaks in March, June, and September."
)

            # Quarterly performance
            filtered_df['start_quarter'] = filtered_df['start_date'].dt.to_period('Q')
            quarterly_performance = filtered_df.groupby('start_quarter')[['revenue', 'calculated_roi', 'conversion_rate', 'net_profit', 'campaign_duration']].mean()
            st.subheader("Quarterly Performance Trends")
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            quarterly_performance['revenue'].plot(kind='bar', ax=axs[0, 0], color='skyblue', title='Revenue by Quarter')
            quarterly_performance['calculated_roi'].plot(kind='bar', ax=axs[0, 1], color='orange', title='ROI by Quarter')
            quarterly_performance['conversion_rate'].plot(kind='bar', ax=axs[1, 0], color='green', title='Conversion Rate by Quarter')
            quarterly_performance['net_profit'].plot(kind='bar', ax=axs[1, 1], color='red', title='Net Profit by Quarter')
            plt.tight_layout()
            st.pyplot(fig)
            st.info(
    "Q2 and Q4 are the most profitable quarters, with Q2 leading in ROI and Q4 in conversion rate. "
    "Longer campaigns perform best in Q1, while short, intensive campaigns excel in Q4. "
    "This suggests that campaign duration and timing should be adapted to the seasonal business cycle for maximum impact."
)

            # Annual trends
            filtered_df['start_year'] = filtered_df['start_date'].dt.year
            yearly_performance = filtered_df.groupby('start_year')[['revenue', 'calculated_roi', 'conversion_rate']].mean()
            st.subheader("Annual Performance Trends")
            fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            yearly_performance['revenue'].plot(ax=axs[0], marker='o', linewidth=2, title='Revenue by Year')
            yearly_performance['calculated_roi'].plot(ax=axs[1], marker='o', linewidth=2, title='ROI by Year')
            yearly_performance['conversion_rate'].plot(ax=axs[2], marker='o', linewidth=2, title='Conversion Rate by Year')
            plt.tight_layout()
            st.pyplot(fig)
            st.info(
    "Yearly trends show stable or slightly increasing performance, with no evidence of long-term decline. "
    "This indicates that the marketing strategy is resilient to annual fluctuations and external shocks, and that best practices are being maintained over time."
)

            # Conclusions
            st.markdown("""
            **Conclusions:**  
            - Two clear performance peaks: spring (April-May) and autumn (September-October).  
            - Wednesday and Thursday are optimal days to launch campaigns.  
            - Long campaigns perform better in Q1; short, intensive campaigns excel in Q4.
            """)

        # --- 9. General Conclusion ---
        with tabs[8]:
            st.header("🧠 General Conclusion")
            st.markdown("""
            This section summarizes the main findings and strategic recommendations from the entire analysis. Each subsection below highlights a key area of insight, followed by a final executive summary.
            """)

            # 1. Global Campaign Performance
            st.subheader("📊 1. Global Campaign Performance")
            st.markdown("""
            - **ROI and Conversion:** The average ROI is high (849.4), but with considerable dispersion. Conversion rates are homogeneous across channels and campaign types, averaging around 0.54.
            - **Efficiency:** The most profitable campaigns are not necessarily those with the highest budgets; efficiency peaks in low-to-mid investment ranges.
            """)

            # 2. Channels and Campaign Types
            st.subheader("🚦 2. Channels and Campaign Types")
            st.markdown("""
            - **Channels:** 'Paid' slightly leads in ROI and proportion of successful campaigns, but there are no statistically significant differences in conversion between channels.
            - **Campaign Types:** 'Social media', 'email', and 'podcast' are the most consistent and profitable. 'Event' is the least efficient.
            - **Winning Combinations:** The combination of 'paid' + 'social media' delivers the highest ROI.
            """)

            # 3. B2B vs B2C Audiences
            st.subheader("👥 3. B2B vs B2C Audiences")
            st.markdown("""
            - **Similarity:** No significant differences in conversion rate or ROI between B2B and B2C.
            - **Nuances:** B2C achieves better conversion in 'paid' and 'webinar', while B2B leads in 'organic' and 'email'. Profit margin is higher for B2C.
            """)

            # 4. Budget–Results Relationship
            st.subheader("📈 4. Budget–Results Relationship")
            st.markdown("""
            - **Revenue:** There is a moderate positive correlation between budget and revenue, but with diminishing marginal returns.
            - **ROI:** There is a moderate inverse relationship; higher investment does not guarantee better ROI. The optimal investment range is between $30,000 and $70,000.
            """)

            # 5. Statistical Insights
            st.subheader("🧪 5. Statistical Insights")
            st.markdown("""
            - **ANOVA and Post-hoc:** No significant differences in conversion by channel within each audience.
            - **Key Factors:** Profit margin, cost per conversion, and operational efficiency are the main ROI drivers.
            """)

            # 6. Successful Campaigns
            st.subheader("🏆 6. Successful Campaigns")
            st.markdown("""
            - **Characteristics:** High conversion, budget efficiency, optimal duration, and precise targeting.
            - **Channels and Types:** Email and organic campaigns dominate in B2B; social media excels in both segments.
            - **Recommendation:** Replicate models that combine high ROI, revenue, and conversion, prioritizing quality over quantity.
            """)

            # 7. Temporal and Seasonal Patterns
            st.subheader("📅 7. Temporal and Seasonal Patterns")
            st.markdown("""
            - **Seasonality:** Two clear performance peaks in spring (April-May) and autumn (September-October).
            - **Day of the Week:** Wednesday and Thursday are optimal for campaign launches.
            - **Duration:** Long campaigns work best in Q1; short, intensive campaigns are most effective in Q4.
            """)

            # 8. Strategic Recommendations
            st.subheader("💡 8. Strategic Recommendations")
            st.markdown("""
            - Optimize profit margin and reduce cost per conversion.
            - Invest in 'paid' channels and 'social media' types to maximize ROI.
            - Adjust budgets to mid-range and avoid over-investment.
            - Leverage seasonal cycles and launch campaigns on high-impact days and months.
            - Continuously monitor and replicate best practices identified in the analysis.
            """)

            # Executive Summary
            st.markdown("""
            ---
            **Executive Summary:**  
            The analysis demonstrates that marketing campaign efficiency is driven by optimal channel-type combinations, moderate budgets, and strategic timing. Social media and paid channels consistently deliver high ROI, while B2B and B2C audiences show similar performance. Budget increases do not guarantee higher ROI, and efficiency peaks in mid-range investments. Seasonality plays a crucial role, with spring and autumn being the most effective periods. Replicating high-performing models and focusing on operational efficiency are key to maximizing marketing returns.
            """)
except Exception as e:
    st.error(f"Error loading or processing data: {e}")
    st.info("Verify that the 'all_month.csv' file is available and has the correct format.")

# --- Dashboard information ---
st.sidebar.markdown("---")
st.sidebar.info("""
**About this Dashboard**

This dashboard displays data from a marketing campaigns for an audience channel.
\nDeveloped with Streamlit and Plotly Express.
""")