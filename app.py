# --- Libraries ---
from math import inf
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
from dash import Dash, dcc, html, Input, Output
import plotly.figure_factory as ff
import plotly.graph_objects as go

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
            col1, col2 = st.columns(2)

            st.header("Channel-Based Analysis")
            st.markdown("**This section explores the distribution, ROI, and comparative metrics of marketing channels.**")

            with col1:
                # Frequency by channel
                st.subheader("Conversion Rate by Channel and Audience")
                channel_conversion = filtered_df.groupby(['target_audience', 'channel'])['conversion_rate'].mean().reset_index()
                fig = px.bar(
                    channel_conversion,
                    x='channel',
                    y='conversion_rate',
                    color='target_audience',
                    barmode='group',
                    text_auto='.2f',
                    title='Conversion Rate by Channel and Audience',
                    color_discrete_sequence=px.colors.qualitative.Set3,  # Paleta moderna y diferente a las demás
                    hover_data={
                        'channel': True,
                        'target_audience': True,
                        'conversion_rate': ':.2f'
                    }
                )
                fig.update_layout(
                    template='plotly_dark',  # Fondo negro
                    xaxis_title='Channel',
                    yaxis_title='Conversion Rate',
                    legend_title='Target Audience',
                    height=500,
                    bargap=0.18
                )
                fig.update_traces(
                    marker_line_width=1.5,
                    marker_line_color='black'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("Promotion is the most frequent channel, followed by referral, paid, and organic. "
    "Promotion leads in campaign count (221), but paid achieves the highest average ROI. "
    "All channels show a balanced distribution, indicating a well-diversified marketing strategy. "
    "Referral and organic channels, while less frequent, still contribute significantly to overall performance.")

            with col2:
                # ROI by channel (boxplot)
                #st.subheader("ROI Distribution by Channel")
                #fig, ax = plt.subplots(figsize=(20, 10))
                #sns.boxplot(x='channel', y='calculated_roi', data=channel_metrics, palette='Set1', ax=ax)
                #for i, v in enumerate(channel_metrics['calculated_roi']):
                #    ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
                #ax.set_title('ROI Distribution by Channel')
                #ax.set_xlabel('Channel')
                #ax.set_ylabel('ROI')
                #st.pyplot(fig)
                st.subheader("ROI Distribution by Channel")
                fig = px.box(
                    filtered_df,
                    x="channel",
                    y="calculated_roi",
                    color="channel",
                    points="all",  # Muestra todos los puntos para mayor detalle
                    title="ROI Distribution by Channel"
                )
                fig.update_traces(quartilemethod="exclusive")  # Consistente con tu plantilla
                fig.update_layout(
                    xaxis_title="Channel",
                    yaxis_title="ROI",
                    boxmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)
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

            col1, col2 = st.columns(2)
            with col1:
                # ROI bar chart by channel (Plotly, fondo negro, colores modernos y diferentes)
                st.subheader("Average ROI by Channel")
                fig = px.bar(
                    channel_metrics,
                    x='channel',
                    y='calculated_roi',
                    color='channel',
                    text_auto='.2f',
                    title='Average ROI by Channel',
                    color_discrete_sequence=px.colors.qualitative.Prism,  # Paleta moderna y diferente a las demás
                    hover_data={
                        'channel': True,
                        'calculated_roi': ':.2f',
                        'conversion_rate': ':.2f',
                        'profit_margin': ':.2f',
                        'revenue': ':.2f',
                        'budget': ':.2f',
                        'num_campaigns': True
                    }
                )
                fig.update_layout(
                    template='plotly_dark',  # Fondo negro
                    xaxis_title='Channel',
                    yaxis_title='Average ROI',
                    showlegend=False,
                    height=500,
                    bargap=0.18,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                fig.update_traces(
                    marker_line_width=1.5,
                    marker_line_color='black'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("The bar chart confirms paid as the channel with the highest average ROI, closely followed by promotion and organic. "
    "All four channels fluctuate between 806 and 894 in average ROI, indicating a balanced performance across the marketing mix.")

            with col2:
                st.subheader("ROI by Channel and Campaign Type")
                fig = px.box(
                    filtered_df,
                    x="channel",
                    y="calculated_roi",
                    color="type",
                    points="all",  # Muestra todos los puntos para mayor detalle
                    title="ROI by Channel and Campaign Type"
                )
                fig.update_traces(quartilemethod="exclusive")  # Consistente con tu plantilla
                fig.update_layout(
                    xaxis_title="Channel",
                    yaxis_title="ROI",
                    boxmode="group"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("Paid channel shows the most stable ROI across types, with 50% of values between 450 and 1400. "
    "Other channels display greater variability, but the central 50% of all categories are between ROI 300 and 1500. "
    "This suggests that paid campaigns are more consistent, while other channels may offer higher upside but also more risk.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ROI by Channel and Target Audience")
                x_axis = st.radio(
                    "Eje X:",
                    options=[
                        "channel",
                        "target_audience",
                        "channel_audience"
                    ],
                    format_func=lambda x: {
                        "channel": "Channel",
                        "target_audience": "Target Audience",
                        "channel_audience": "Channel + Audience"
                    }[x],
                    horizontal=True,
                    key="roi_x_axis"
                )
                y_axis = st.radio(
                    "Eje Y:",
                    options=["calculated_roi"],
                    format_func=lambda x: {"calculated_roi": "ROI"}[x],
                    horizontal=True,
                    key="roi_y_axis"
                )

                plot_df = filtered_df.copy()
                if x_axis == "channel_audience":
                    plot_df["channel_audience"] = plot_df["channel"] + " - " + plot_df["target_audience"]
                    fig = px.box(plot_df, x="channel_audience", y=y_axis, color="channel")
                else:
                    fig = px.box(
                        plot_df,
                        x=x_axis,
                        y=y_axis,
                        color="channel" if x_axis == "channel" else "target_audience"
                    )
                fig.update_layout(title="ROI by Channel and Target Audience", xaxis_title=x_axis, yaxis_title=y_axis)
                st.plotly_chart(fig, use_container_width=True)
                st.info(
                    "The top 25% ROI values are highly dispersed across audiences for all channels. "
                    "This means that more than half of the audience-channel combinations fall below the overall mean ROI (849.4). "
                    "There is no clear audience-channel combination that consistently outperforms others, highlighting the need for tailored strategies."
    )

            with col2:
                # ROI vs Budget scatter
                st.subheader("ROI vs Budget by Channel")
                # Ordenar para que referral se dibuje primero y paid al final
                channel_order = ["referral", "organic", "promotion", "paid"]
                filtered_df['channel'] = pd.Categorical(filtered_df['channel'], categories=channel_order, ordered=True)
                filtered_df = filtered_df.sort_values('channel')

                size_values = filtered_df['calculated_roi'].abs()
                if size_values.max() > 0:
                    size_values = 20 + 80 * (size_values - size_values.min()) / (size_values.max() - size_values.min())
                else:
                    size_values = 40

                fig_scatter = px.scatter(
                    filtered_df,
                    x="budget",
                    y="calculated_roi",
                    color="channel",
                    size=size_values,
                    size_max=40,  # Reduce el tamaño máximo para menos solapamiento
                    opacity=0.7,  # Más opaco para distinguir colores
                    hover_name="calculated_roi_performance",
                    color_discrete_map={
                        "paid": "blue",
                        "organic": "green",
                        "promotion": "orange",
                        "referral": "purple"
                    },
                    labels={"budget": "budget", "calculated_roi": "calculated_roi"}
                )

                # Añade borde negro a los círculos
                fig_scatter.update_traces(marker=dict(line=dict(width=1, color='black')))

                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)
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

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Average Revenue by Campaign Type")
                fig = px.bar(
                    campaign_metrics_by_revenue,
                    x='type',
                    y='revenue',
                    color='type',
                    hover_data={
                        'revenue': ':.2f',
                        'conversion_rate': ':.2f',
                        'calculated_roi': ':.2f',
                        'net_profit': ':.2f',
                        'count': True
                    },
                    title='Average Revenue by Campaign Type',
                    labels={'type': 'Campaign Type', 'revenue': 'Average Revenue'}
                )
                fig.update_layout(showlegend=False, xaxis_title='Campaign Type', yaxis_title='Average Revenue', height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.info("Revenue is homogeneous across most campaign types, with the exception of 'event', which is significantly lower. "
        "Most types achieve average revenues between $450,000 and $480,000. "
        "This suggests that, except for events, all campaign types are capable of generating substantial revenue, providing flexibility in campaign planning.")
                
            with col2:
                # Conversion rate by type (bar)
                campaign_metrics_by_conv = campaign_metrics.sort_values('conversion_rate', ascending=False)
                st.subheader("Average Conversion Rate by Campaign Type")
                fig = px.bar(
                    campaign_metrics_by_conv,
                    x='type',
                    y='conversion_rate',
                    color='type',
                    color_discrete_sequence=px.colors.qualitative.Vivid,  # Paleta moderna y vibrante
                    hover_data={
                        'conversion_rate': ':.2f',
                        'revenue': ':.2f',
                        'calculated_roi': ':.2f',
                        'net_profit': ':.2f',
                        'count': True
                    },
                    title='Average Conversion Rate by Campaign Type',
                    labels={'type': 'Campaign Type', 'conversion_rate': 'Average Conversion Rate'}
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_title='Campaign Type',
                    yaxis_title='Average Conversion Rate',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("Conversion rates are also homogeneous, ranging from 0.52 to 0.55 across most types. "
        "This indicates that the probability of converting leads is stable regardless of campaign type, with no type showing a clear advantage in conversion efficiency.")

            # Revenue vs Conversion scatter
            #st.subheader("Revenue vs Conversion Rate by Campaign Type")
            #fig, ax = plt.subplots(figsize=(10, 6))
            #sns.scatterplot(x='conversion_rate', y='revenue', size='count', sizes=(100, 500), hue='calculated_roi', data=campaign_metrics, palette='viridis', ax=ax)
            #for i, row in campaign_metrics.iterrows():
            #    ax.text(row['conversion_rate']+0.02, row['revenue'], row['type'], fontsize=11)
            #ax.set_title('Revenue vs Conversion Rate by Campaign Type')
            #ax.set_xlabel('Conversion Rate')
            #ax.set_ylabel('Average Revenue')
            #st.pyplot(fig)
            #st.info("'Podcast' campaigns achieve the highest average revenue, despite having a similar conversion rate to other types. "
    #"'Social media' stands out with the highest average ROI (most intense color), while 'event' campaigns are the lowest in both revenue and conversion. "
    #"The size of the circles shows that 'social media' and 'email' are the most common types. "
    #"Overall, the distribution is homogeneous, but small differences may indicate specific optimization opportunities.")

            # Key metrics table
            st.subheader("Key Metrics by Campaign Type")
            st.dataframe(campaign_metrics, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                # Net profit by type (interactive bar chart con filtros y media general)
                st.subheader("Average Net Profit by Campaign Type")

                if 'campaign_metrics_by_profit' not in locals():
                    campaign_metrics_by_profit = campaign_metrics.sort_values('net_profit', ascending=False)

                # Filtro dinámico de tipos de campaña
                campaign_types = campaign_metrics_by_profit['type'].unique().tolist()
                selected_types = st.multiselect(
                    "Filter by Campaign Type",
                    options=campaign_types,
                    default=campaign_types,
                    key="net_profit_type_filter"
                )
                filtered_profit = campaign_metrics_by_profit[campaign_metrics_by_profit['type'].isin(selected_types)]

                # Cálculo de la media general
                overall_mean = campaign_metrics_by_profit['net_profit'].mean()

                fig = px.bar(
                    filtered_profit,
                    x='type',
                    y='net_profit',
                    color='type',
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    hover_data={
                        'net_profit': ':.2f',
                        'revenue': ':.2f',
                        'conversion_rate': ':.2f',
                        'calculated_roi': ':.2f',
                        'count': True
                    },
                    text='net_profit',
                    title='Average Net Profit by Campaign Type',
                    labels={'type': 'Campaign Type', 'net_profit': 'Average Net Profit'}
                )
                fig.update_traces(
                    texttemplate='%{text:,.0f}',
                    textposition='outside',
                    marker_line_width=1.5,
                    marker_line_color='black'
                )
                # Línea de media general
                fig.add_hline(
                    y=overall_mean,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Overall Mean: {overall_mean:,.0f}",
                    annotation_position="top left"
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis_title='Campaign Type',
                    yaxis_title='Average Net Profit',
                    height=500,
                    uniformtext_minsize=8,
                    uniformtext_mode='hide'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("'Social media', 'email', and 'podcast' types are the most homogeneous and profitable in terms of net profit. "
        "'Webinar' is slightly below, while 'event' is much lower, making it the least profitable option.")

            
                # ROI vs Net Profit scatter
                #campaign_metrics_filtered = campaign_metrics[campaign_metrics['type'] != 'event']
                #st.subheader("ROI vs Net Profit by Campaign Type")
                #fig, ax = plt.subplots(figsize=(10, 6))
                #sns.scatterplot(x='calculated_roi', y='net_profit', size='count', sizes=(100, 500), hue='type', data=campaign_metrics_filtered, palette='viridis', ax=ax)
                #ax.set_title('ROI vs Net Profit by Campaign Type')
                #ax.set_xlabel('Average ROI')
                #ax.set_ylabel('Average Net Profit')
                #st.pyplot(fig)
                #st.info("The top three campaign types show a homogeneous distribution in both ROI and net profit, confirming their consistency and reliability. "
        #"This highlights the strategic value of focusing on these types for sustained profitability.")

            with col2:
                # Efficiency: profit per dollar
                campaign_metrics['budget'] = filtered_df.groupby('type')['budget'].mean().reset_index()['budget']
                campaign_metrics['profit_per_dollar'] = campaign_metrics['net_profit'] / campaign_metrics['budget']
                campaign_metrics_by_efficiency = campaign_metrics.sort_values('profit_per_dollar', ascending=False)
                st.subheader("Net Profit per Dollar Invested by Campaign Type")
                fig, ax = plt.subplots(figsize=(15,7))
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

            col1, col2 = st.columns(2)
            with col1:
                # Correlation heatmap (interactive Plotly)
                st.subheader("Correlation between Key Metrics")
                metrics = ['budget', 'revenue', 'net_profit', 'calculated_roi', 'conversion_rate', 'profit_per_dollar']
                correlation = campaign_metrics[metrics].corr()

                

                fig = ff.create_annotated_heatmap(
                    z=correlation.values.round(2),
                    x=list(correlation.columns),
                    y=list(correlation.index),
                    colorscale='Viridis',
                    showscale=True,
                    annotation_text=correlation.round(2).astype(str).values
                )
                fig.update_layout(
                    title='Correlation between Financial and Performance Metrics',
                    xaxis_title="Metric",
                    yaxis_title="Metric",
                    width=800,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
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

            # ROI distribution (histogram, boxplot, QQ plot) - versión interactiva y moderna
            st.subheader("ROI Distribution")

            # Selección de métrica para explorar
            roi_metric = st.selectbox(
                "Select metric to analyze",
                options=[
                    ("ROI", "calculated_roi"),
                    ("Net Profit", "net_profit"),
                    ("Revenue", "revenue"),
                    ("Conversion Rate", "conversion_rate")
                ],
                format_func=lambda x: x[0],
                index=0,
                key="roi_dist_metric"
            )[1]

            col1, col2 = st.columns(2)

            with col1:
                # Histograma interactivo
                fig_hist = px.histogram(
                    filtered_df,
                    x=roi_metric,
                    nbins=30,
                    color_discrete_sequence=["#636EFA"],
                    marginal="box",
                    opacity=0.85,
                    title=f"Distribution of {roi_metric.replace('_', ' ').title()}",
                    hover_data=filtered_df.columns
                )
                fig_hist.update_layout(
                    bargap=0.05,
                    xaxis_title=roi_metric.replace('_', ' ').title(),
                    yaxis_title="Count",
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            with col2:
                # Boxplot interactivo
                fig_box = px.box(
                    filtered_df,
                    y=roi_metric,
                    color_discrete_sequence=["#00CC96"],
                    points="all",
                    title=f"{roi_metric.replace('_', ' ').title()} Boxplot",
                    hover_data=filtered_df.columns
                )
                fig_box.update_traces(quartilemethod="exclusive")
                fig_box.update_layout(
                    yaxis_title=roi_metric.replace('_', ' ').title(),
                    template="plotly_white",
                    height=400
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            col1, col2 = st.columns(2)

            with col1:
                # Q-Q Plot estilizado con fondo negro
                st.subheader(f"Interactive Q-Q Plot: {roi_metric.replace('_', ' ').title()}")

                # Calcular Q-Q plot con línea ajustada
                data = filtered_df[roi_metric].dropna()
                (osm, osr), (slope, intercept, _) = stats.probplot(data, dist="norm", fit=True)

                # Calcular puntos de la línea real de ajuste
                line_y = slope * osm + intercept

                fig = go.Figure()

                # Puntos de datos
                fig.add_trace(go.Scatter(
                    x=osm,
                    y=osr,
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='orange', size=7),
                    hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
                ))

                # Línea de ajuste verdadera
                fig.add_trace(go.Scatter(
                    x=osm,
                    y=line_y,
                    mode='lines',
                    name='Fit Line',
                    line=dict(color='cyan', dash='dash'),
                    hoverinfo='skip'
                ))

                fig.update_layout(
                    template='plotly_dark',
                    title=f"Q-Q Plot of {roi_metric.replace('_', ' ').title()}",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    height=500,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )

                st.plotly_chart(fig, use_container_width=True)
                st.info("ROI is right-skewed with a high mean (approx. 849), but with considerable dispersion and several high-value outliers. "
        "The distribution is not normal (confirmed by the Q-Q plot), and most campaigns achieve moderate ROI, with a few achieving exceptionally high returns. "
        "This suggests that while most campaigns perform within a predictable range, there are rare cases of outstanding performance that can significantly impact overall results. "
        "Segmenting campaigns by ROI reveals clear groups of high, medium, and low performers, highlighting the importance of identifying and replicating the factors behind top-performing campaigns.")

            # ROI by main categories (boxplots)
            st.subheader("ROI by Main Categories")

            # ROI by Channel (Plotly)
            fig_channel = px.box(
                filtered_df,
                x='channel',
                y='calculated_roi',
                color='channel',
                points="all",
                color_discrete_sequence=px.colors.qualitative.Vivid,
                title="ROI by Channel",
                hover_data=filtered_df.columns
            )
            fig_channel.update_traces(quartilemethod="exclusive")
            fig_channel.update_layout(
                xaxis_title="Channel",
                yaxis_title="ROI",
                boxmode="group",
                height=400,
                showlegend=False
            )

            # ROI by Campaign Type (Plotly)
            fig_type = px.box(
                filtered_df,
                x='type',
                y='calculated_roi',
                color='type',
                points="all",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="ROI by Campaign Type",
                hover_data=filtered_df.columns
            )
            fig_type.update_traces(quartilemethod="exclusive")
            fig_type.update_layout(
                xaxis_title="Campaign Type",
                yaxis_title="ROI",
                boxmode="group",
                height=400,
                showlegend=False
            )

            # ROI by Target Audience (Plotly)
            fig_audience = px.box(
                filtered_df,
                x='target_audience',
                y='calculated_roi',
                color='target_audience',
                points="all",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="ROI by Target Audience",
                hover_data=filtered_df.columns
            )
            fig_audience.update_traces(quartilemethod="exclusive")
            fig_audience.update_layout(
                xaxis_title="Target Audience",
                yaxis_title="ROI",
                boxmode="group",
                height=400,
                showlegend=False
            )

            # ROI by Budget Quartile (Plotly)
            filtered_df['budget_quartile'] = pd.qcut(filtered_df['budget'], 4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
            fig_budget = px.box(
                filtered_df,
                x='budget_quartile',
                y='calculated_roi',
                color='budget_quartile',
                points="all",
                color_discrete_sequence=px.colors.qualitative.Prism,
                title="ROI by Budget Quartile",
                hover_data=filtered_df.columns
            )
            fig_budget.update_traces(quartilemethod="exclusive")
            fig_budget.update_layout(
                xaxis_title="Budget Quartile",
                yaxis_title="ROI",
                boxmode="group",
                height=400,
                showlegend=False
            )

            # Mostrar los 4 gráficos en una cuadrícula dinámica
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_channel, use_container_width=True)
                st.plotly_chart(fig_audience, use_container_width=True)
            with col2:
                st.plotly_chart(fig_type, use_container_width=True)
                st.plotly_chart(fig_budget, use_container_width=True)
            st.info("ROI is distributed homogeneously across channels, with 'paid' showing the highest mean and lowest variability. "
    "'Social media', 'email', and 'podcast' campaign types are the most consistent and profitable, while 'event' is excluded due to poor performance. "
    "No substantial differences are observed between B2B and B2C audiences, but B2C shows slightly higher variability. "
    "There is a slight inverse relationship between budget and ROI: lower and mid-range budgets tend to achieve higher ROI, while very high budgets show lower efficiency. "
    "These patterns suggest that focusing on efficient channels and campaign types, and optimizing budget allocation, can maximize ROI."
    "calculated_roi mediana por Canal:"
    "channel"
    "paid        856.77"
    "organic     807.50"
    "referral    772.62"
    "promotion   689.32"

    "calculated_roi mediana por Tipo de Campaña:"
    "type"
    "email          826.79"
    "social media   794.50"
    "podcast        771.54"
    "webinar        739.42"

    "calculated_roi mediana por Audiencia Objetivo:"
    "target_audience"
    "B2B   780.18"
    "B2C   771.68"

    "calculated_roi mediana por Categoría de Presupuesto:"
    "budget_category"
    "low budget      1,958.49"
    "medium budget   1,248.13"
    "high budget       562.75")

            col1, col2 = st.columns(2)
            with col1:
                # Correlation heatmap (ROI vs numeric variables)
                st.subheader("Correlation between ROI and Numeric Variables")
                numeric_cols = [
                    'calculated_roi', 'budget', 'conversion_rate', 'revenue',
                    'net_profit', 'campaign_duration', 'cost_per_conversion', 'profit_margin'
                ]
                # Asegura que todas las columnas existen y son numéricas
                available_cols = [col for col in numeric_cols if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col])]
                correlation = filtered_df[available_cols].corr()

                fig = ff.create_annotated_heatmap(
                    z=correlation.values.round(2),
                    x=list(correlation.columns),
                    y=list(correlation.index),
                    colorscale='Viridis',
                    showscale=True,
                    annotation_text=correlation.round(2).astype(str).values
                )
                fig.update_layout(
                    title='Correlation between ROI and Numeric Variables',
                    xaxis_title="Metric",
                    yaxis_title="Metric",
                    width=800,
                    height=500,
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)

                # Selecciona las 3 variables más correlacionadas con ROI (excluyendo la propia)
                corr_roi = correlation['calculated_roi'].drop('calculated_roi', errors='ignore')
                most_correlated = corr_roi.abs().sort_values(ascending=False).head(3).index.tolist()

                # Scatterplots modernos con fondo negro y hover interactivo
                st.subheader("ROI vs Most Correlated Variables")
                color_palettes = [px.colors.qualitative.Vivid, px.colors.qualitative.Pastel, px.colors.qualitative.Dark24]
                for i, column in enumerate(most_correlated):
                    # Asigna un color diferente a cada scatter usando una columna categórica si existe, si no, usa un color único
                    color_col = None
                    if 'type' in filtered_df.columns:
                        color_col = 'type'
                        color_seq = color_palettes[i % len(color_palettes)]
                    elif 'channel' in filtered_df.columns:
                        color_col = 'channel'
                        color_seq = color_palettes[i % len(color_palettes)]
                    else:
                        color_seq = [px.colors.qualitative.Vivid[i % len(px.colors.qualitative.Vivid)]]

                    fig_scatter = px.scatter(
                        filtered_df,
                        x=column,
                        y='calculated_roi',
                        color=color_col,
                        color_discrete_sequence=color_seq,
                        trendline='ols',
                        template='plotly_dark',
                        title=f'calculated_roi vs {column}',
                        labels={column: column, 'calculated_roi': 'ROI'},
                        hover_data=filtered_df.columns
                    )
                    fig_scatter.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
                    st.plotly_chart(fig_scatter, use_container_width=True)
                st.info("ROI is highly positively correlated with net profit (r ≈ 0.71) and profit margin, and negatively correlated with cost per conversion (-0.24) and budget (-0.47). "
        "This means that maximizing profit margin and reducing cost per conversion are key to improving ROI, while simply increasing budget does not guarantee better results. "
        "Conversion rate has a weak but positive correlation with ROI, indicating that improving conversion can have a tangible, though not linear, impact on returns. "
        "These insights support a strategy focused on operational efficiency and targeted investment rather than increasing spend indiscriminately.")

            # Analizamos las mejores combinaciones de canal y tipo
            st.subheader("Top 10 Channel & Campaign Type Combinations by ROI")

            # Filtros interactivos para canal y tipo
            channels = df['channel'].unique().tolist()
            types = df['type'].unique().tolist()
            selected_channels = st.multiselect("Filter Channels", options=channels, default=channels, key="top_combos_channels")
            selected_types = st.multiselect("Filter Campaign Types", options=types, default=types, key="top_combos_types")

            # Agrupa y filtra según selección
            channel_type_calculated_roi = df.groupby(['channel', 'type'])['calculated_roi'].mean().reset_index()
            channel_type_calculated_roi = channel_type_calculated_roi[
                channel_type_calculated_roi['channel'].isin(selected_channels) &
                channel_type_calculated_roi['type'].isin(selected_types)
            ]
            channel_type_calculated_roi = channel_type_calculated_roi.sort_values('calculated_roi', ascending=False)
            top_combinations = channel_type_calculated_roi.head(10)

            # Gráfico interactivo con Plotly
            fig = px.bar(
                top_combinations,
                x='calculated_roi',
                y='channel',
                color='type',
                orientation='h',
                color_discrete_sequence=px.colors.qualitative.Pastel2,
                hover_data={
                    'channel': True,
                    'type': True,
                    'calculated_roi': ':.2f'
                },
                title='Top 10 Channel & Campaign Type Combinations by Average ROI'
            )
            fig.update_layout(
                template='plotly_dark',
                xaxis_title='Average ROI',
                yaxis_title='Channel',
                legend_title='Campaign Type',
                height=480,
                margin=dict(l=30, r=30, t=60, b=40)
            )
            fig.update_traces(marker_line_width=1.5, marker_line_color='black')
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(top_combinations, use_container_width=True)

            st.info(
                "This chart displays the top 10 channel and campaign type combinations by average ROI. "
                "Social media and paid channels are consistently among the most efficient, while organic and email also appear in the top combinations. "
                "Use the filters above to explore how different channels and campaign types perform. "
                "Focusing on these high-performing combinations can help maximize marketing efficiency and returns."
            )

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

            # Key metrics by audience (boxplots)
            st.subheader("Key Metrics by Audience")

            filtered_audience = filtered_df[filtered_df['target_audience'].isin(['B2B', 'B2C'])]
            metrics = [
                ('ROI', 'calculated_roi'),
                ('Conversion Rate', 'conversion_rate'),
                ('Profit Margin', 'profit_margin'),
                ('Revenue', 'revenue'),
                ('Budget', 'budget'),
                ('Cost per Conversion', 'cost_per_conversion')
            ]

            # Mostrar los boxplots en una cuadrícula moderna con Plotly y fondo negro
            col1, col2, col3 = st.columns(3)
            for i, (label, metric) in enumerate(metrics):
                fig = px.box(
                    filtered_audience,
                    x='target_audience',
                    y=metric,
                    color='target_audience',
                    color_discrete_sequence=px.colors.qualitative.Dark24,
                    points="all",
                    title=f"{label}: B2B vs B2C",
                    hover_data=filtered_audience.columns
                )
                fig.update_traces(quartilemethod="exclusive")
                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Audience",
                    yaxis_title=label,
                    showlegend=False,
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                if i < 3:
                    with col1:
                        st.plotly_chart(fig, use_container_width=True)
                elif i < 5:
                    with col2:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    with col3:
                        st.plotly_chart(fig, use_container_width=True)
            st.info("All key metrics are highly similar between B2B and B2C, except for profit margin, which is about 27% higher for B2C. "
    "Cost per conversion is slightly higher for B2C, possibly explaining the higher profit margin. "
    "Overall, both segments have comparable ROI, conversion rate, revenue, and budget, with overlapping interquartile ranges and similar variability.")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Conversion Rate by Channel and Audience")
                channel_conversion = filtered_df.groupby(['target_audience', 'channel'])['conversion_rate'].mean().reset_index()
                fig = px.bar(
                    channel_conversion,
                    x='channel',
                    y='conversion_rate',
                    color='target_audience',
                    barmode='group',
                    text_auto='.2f',
                    title='Conversion Rate by Channel and Audience',
                    hover_data={
                        'channel': True,
                        'target_audience': True,
                        'conversion_rate': ':.2f'
                    }
                )
                fig.update_layout(
                    template='plotly_dark',  # Fondo negro
                    xaxis_title='Channel',
                    yaxis_title='Conversion Rate',
                    legend_title='Target Audience',
                    height=500
                )
                fig.update_traces(
                    marker_line_width=1.5,
                    marker_line_color='black'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Conversion rate by type and audience (bar)
                st.subheader("Conversion Rate by Campaign Type and Audience")
                type_conversion = filtered_df.groupby(['target_audience', 'type'])['conversion_rate'].mean().reset_index()
                fig = px.bar(
                    type_conversion,
                    x='type',
                    y='conversion_rate',
                    color='target_audience',
                    barmode='group',
                    text_auto='.2f',
                    title='Conversion Rate by Campaign Type and Audience',
                    color_discrete_sequence=px.colors.qualitative.Pastel2,  # Paleta moderna y diferente
                    hover_data={
                        'type': True,
                        'target_audience': True,
                        'conversion_rate': ':.2f'
                    }
                )
                fig.update_layout(
                    template='plotly_dark',  # Fondo negro
                    xaxis_title='Campaign Type',
                    yaxis_title='Conversion Rate',
                    legend_title='Target Audience',
                    height=500,
                    bargap=0.25
                )
                fig.update_traces(
                    marker_line_width=1.5,
                    marker_line_color='black'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info(" Social media campaigns have identical conversion rates for B2B and B2C (0.55). "
        "B2C surprisingly outperforms B2B in webinars (0.58 vs 0.52), while B2B leads slightly in email. "
        "Event campaigns are ineffective for both audiences, especially B2B. "
        "These patterns suggest that campaign type can interact with audience to produce nuanced results.")

            col1, col2 = st.columns(2)
            with col1:
                # Conversion rate boxplot
                st.subheader("Conversion Rate: B2B vs B2C")
                # Adaptación: Boxplot interactivo con Plotly, fondo negro y colores modernos/diferentes
                fig = px.box(
                    filtered_audience,
                    x='target_audience',
                    y='conversion_rate',
                    color='target_audience',
                    color_discrete_sequence=px.colors.qualitative.Prism,  # Paleta moderna y diferente
                    points="all",
                    title="Conversion Rate: B2B vs B2C",
                    hover_data=filtered_audience.columns
                )
                fig.update_traces(quartilemethod="exclusive")
                fig.update_layout(
                    template="plotly_dark",  # Fondo negro
                    xaxis_title="Audience",
                    yaxis_title="Conversion Rate",
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("Conversion rates are nearly identical for B2B and B2C, with both segments showing a mean around 0.54 and virtually overlapping distributions. "
        "Statistical tests (t-test, p-value ≈ 0.94) confirm there is no significant difference between the two audiences. "
        "This suggests that audience type alone does not determine conversion success, and any observed differences are likely due to random variation.")

            with col2:
                # Heatmap: conversion by channel and audience
                # Heatmap: correlación entre ROI y variables numéricas, fondo negro, colores modernos
                st.subheader("Correlation between ROI and Numeric Variables")
                numeric_cols = [
                    'calculated_roi', 'budget', 'conversion_rate', 'revenue',
                    'net_profit', 'campaign_duration', 'cost_per_conversion', 'profit_margin'
                ]
                correlation = filtered_df[numeric_cols].corr()

                fig = ff.create_annotated_heatmap(
                    z=correlation.values.round(2),
                    x=list(correlation.columns),
                    y=list(correlation.index),
                    colorscale='Cividis',  # Paleta moderna y diferente
                    showscale=True,
                    annotation_text=correlation.round(2).astype(str).values
                )
                fig.update_layout(
                    title='Correlation between ROI and Numeric Variables',
                    xaxis_title="Metric",
                    yaxis_title="Metric",
                    width=800,
                    height=500,
                    title_x=0.5,
                    template='plotly_dark'  # Fondo negro
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("The heatmap confirms natural specialization: B2B excels in organic channels, while B2C dominates paid. "
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

            col1, col2 = st.columns(2)
            with col1:
                # ROI vs Net Profit bar
                st.subheader("Correlation between ROI and Numeric Variables")
                numeric_cols = [
                    'calculated_roi', 'budget', 'conversion_rate', 'revenue',
                    'net_profit', 'campaign_duration', 'cost_per_conversion', 'profit_margin'
                ]
                correlation = filtered_df[numeric_cols].corr().reset_index().melt(id_vars='index')
                correlation.columns = ['Metric_X', 'Metric_Y', 'Correlation']

                fig = px.bar(
                    correlation,
                    x='Metric_X',
                    y='Correlation',
                    color='Metric_Y',
                    barmode='group',
                    text_auto='.2f',
                    title='Correlation between ROI and Numeric Variables',
                    color_discrete_sequence=px.colors.qualitative.Bold,  # Paleta moderna y diferente
                    hover_data={
                        'Metric_X': True,
                        'Metric_Y': True,
                        'Correlation': ':.2f'
                    }
                )
                fig.update_layout(
                    template='plotly_dark',  # Fondo negro
                    xaxis_title="Metric",
                    yaxis_title="Correlation",
                    height=500,
                    bargap=0.18,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                fig.update_traces(
                    marker_line_width=1.5,
                    marker_line_color='black'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info(
    "This bar chart visualizes the correlation between ROI and the main numeric variables in the dataset. "
    "ROI shows the strongest positive correlation with profit margin and net profit, while budget and cost per conversion are negatively correlated. "
    "These insights help identify the key drivers of performance and support data-driven decisions to maximize campaign efficiency.")
                
            with col2:
                # Distribution of campaign types (pie)
                st.subheader("Distribution of Campaign Types")
                campaign_types = top_campaigns['type'].value_counts()
                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=campaign_types.index,
                            values=campaign_types.values,
                            pull=[0.08 if i == 0 else 0 for i in range(len(campaign_types))],  # Destaca el más frecuente
                            marker=dict(
                                colors=px.colors.qualitative.Pastel2  # Paleta moderna y dinámica
                            ),
                            textinfo='percent+label',
                            hoverinfo='label+value+percent'
                        )
                    ]
                )
                fig.update_layout(
                    title='Distribution of Successful Campaign Types',
                    template='plotly_dark',  # Fondo negro
                    legend_title='Campaign Type',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info("The most profitable campaigns are dominated by email and organic types, with social media and podcast also present. "
        "This distribution suggests that while social media campaigns can be highly effective, email and organic strategies are more consistently represented among top performers. "
        "Event campaigns are notably absent, confirming their lower profitability.")

            col1, col2 = st.columns(2)
            with col1:
                # Correlation with net profit (heatmap)
                st.subheader("Correlation with Net Profit")
                correlation_columns = ['net_profit', 'budget', 'conversion_rate', 'calculated_roi', 'campaign_duration', 'revenue']
                correlation_matrix = filtered_df[correlation_columns].corr()
                fig, ax = plt.subplots(figsize=(20, 10))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                ax.set_title('Correlation: Factors Influencing Net Profit')
                # Cambia el fondo a negro
                fig.patch.set_facecolor('black')
                ax.set_facecolor('black')
                # Cambia el color de los ejes y etiquetas a blanco para contraste
                ax.title.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
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

            col1, col2 = st.columns(2)
            with col1:
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

                # Usar Plotly para fondo negro e interactividad
                import plotly.graph_objects as go

                fig = go.Figure()

                # Scatter de los puntos
                fig.add_trace(go.Scatter(
                    x=filtered_df['budget'],
                    y=filtered_df['revenue'],
                    mode='markers',
                    name='Campaigns',
                    marker=dict(color='orange', size=8, line=dict(width=1, color='black')),
                    hovertemplate='Budget: %{x:,.0f}<br>Revenue: %{y:,.0f}<extra></extra>'
                ))

                # Línea de regresión polinómica
                fig.add_trace(go.Scatter(
                    x=sorted_budget,
                    y=sorted_poly_pred,
                    mode='lines',
                    name=f'Polynomial Regression (degree {degree})',
                    line=dict(color='lime', width=3),
                    hovertemplate='Budget: %{x:,.0f}<br>Predicted Revenue: %{y:,.0f}<extra></extra>'
                ))

                fig.update_layout(
                    template='plotly_dark',
                    title='Budget vs Revenue: Polynomial Model',
                    xaxis_title='Budget',
                    yaxis_title='Revenue',
                    height=600,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )

                st.plotly_chart(fig, use_container_width=True)
                st.info(
        "The polynomial regression curve clearly shows diminishing marginal returns: "
        "as budget increases, revenue grows at a decreasing rate, especially at high investment levels. "
        "The optimal investment range, where each additional dollar has the greatest impact, is between $30,000 and $70,000. "
        "Both very low and very high budgets are less efficient, confirming the existence of a saturation point."
    )

            with col2:
                # By campaign type
                st.subheader("Budget vs Revenue by Campaign Type")
                fig = go.Figure()

                palette = px.colors.qualitative.Pastel2
                types = filtered_df['type'].unique()
                color_map = {t: palette[i % len(palette)] for i, t in enumerate(types)}

                for t in types:
                    df_type = filtered_df[filtered_df['type'] == t]
                    fig.add_trace(go.Scatter(
                        x=df_type['budget'],
                        y=df_type['revenue'],
                        mode='markers',
                        name=t,
                        marker=dict(color=color_map[t], size=9, line=dict(width=1, color='black')),
                        hovertemplate=f"Type: {t}<br>Budget: %{{x:,.0f}}<br>Revenue: %{{y:,.0f}}<extra></extra>"
                    ))

                # Línea de regresión polinómica
                fig.add_trace(go.Scatter(
                    x=sorted_budget,
                    y=sorted_poly_pred,
                    mode='lines',
                    name=f'Polynomial Regression (degree {degree})',
                    line=dict(color='lime', width=3),
                    hovertemplate='Budget: %{x:,.0f}<br>Predicted Revenue: %{y:,.0f}<extra></extra>'
                ))

                fig.update_layout(
                    template='plotly_dark',
                    title='Budget vs Revenue by Campaign Type',
                    xaxis_title='Budget',
                    yaxis_title='Revenue',
                    height=600,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )

                st.plotly_chart(fig, use_container_width=True)
                st.info(
        "The relationship between budget and revenue varies by campaign type. "
        "Podcast, social media, and webinar campaigns show the strongest positive correlations, "
        "while email campaigns show almost no correlation, indicating that budget is not a key driver for email performance. "
        "This highlights the need for differentiated budget strategies by campaign type."
    )

            col1, col2 = st.columns(2)
            with col1:
                # Correlation matrix
                st.subheader("Correlation Matrix")
                correlation_vars = ['budget', 'revenue', 'calculated_roi', 'conversion_rate', 'campaign_duration']
                corr_matrix = filtered_df[correlation_vars].corr()

                import plotly.figure_factory as ff

                fig = ff.create_annotated_heatmap(
                    z=corr_matrix.values.round(2),
                    x=list(corr_matrix.columns),
                    y=list(corr_matrix.index),
                    colorscale='Turbo',  # Paleta moderna y dinámica
                    showscale=True,
                    annotation_text=corr_matrix.round(2).astype(str).values
                )
                fig.update_layout(
                    title='Correlation Matrix',
                    xaxis_title="Metric",
                    yaxis_title="Metric",
                    width=700,
                    height=500,
                    title_x=0.5,
                    template='plotly_dark',  # Fondo negro moderno
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info(
        "The correlation matrix confirms that budget and revenue are moderately correlated, "
        "but budget is negatively correlated with ROI. "
        "Campaign duration and conversion rate show weak relationships with both budget and revenue, "
        "suggesting that other factors are more important for maximizing efficiency."
)

            with col2:
                # ROI vs Budget scatter
                st.subheader("ROI vs Budget by Channel")
                # Usar Plotly para fondo negro e interactividad
                fig = go.Figure()

                # Paleta moderna y vibrante para los canales
                channel_palette = {
                    "paid": "#636EFA",      # Azul
                    "organic": "#00CC96",   # Verde
                    "promotion": "#FFA15A", # Naranja
                    "referral": "#AB63FA"   # Violeta
                }

                for channel in filtered_df['channel'].unique():
                    df_channel = filtered_df[filtered_df['channel'] == channel]
                    fig.add_trace(go.Scatter(
                        x=df_channel['budget'],
                        y=df_channel['calculated_roi'],
                        mode='markers',
                        name=channel.capitalize(),
                        marker=dict(
                            color=channel_palette.get(channel, "#CCCCCC"),
                            size=10,
                            line=dict(width=1, color='black')
                        ),
                        hovertemplate=f"Channel: {channel.capitalize()}<br>Budget: %{{x:,.0f}}<br>ROI: %{{y:,.2f}}<extra></extra>"
                    ))

                # LOWESS trend line (global, not by channel)
                import statsmodels.api as sm
                lowess = sm.nonparametric.lowess(filtered_df['calculated_roi'], filtered_df['budget'], frac=0.3)
                fig.add_trace(go.Scatter(
                    x=lowess[:, 0],
                    y=lowess[:, 1],
                    mode='lines',
                    name='LOWESS Trend',
                    line=dict(color='red', width=3, dash='dash'),
                    hovertemplate='Budget: %{x:,.0f}<br>LOWESS ROI: %{y:,.2f}<extra></extra>'
                ))

                fig.update_layout(
                    template='plotly_dark',
                    title='Budget vs ROI by Channel',
                    xaxis_title='Budget',
                    yaxis_title='ROI',
                    height=600,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )

                st.plotly_chart(fig, use_container_width=True)
                st.info(
        "There is a moderate inverse relationship between budget and ROI (correlation ≈ -0.57): "
        "higher budgets tend to have lower ROI, especially above 70,000. "
        "The most efficient campaigns are concentrated in the low and mid budget segments, "
        "while high-budget campaigns show lower and less variable ROI. "
        "This supports the strategy of focusing investments in the 30,000 - 70,000 range for optimal efficiency."
    )

            col1, col2 = st.columns(2)
            with col1:
                # Revenue by budget segment (boxplot)
                st.subheader("Revenue by Budget Segment")
                filtered_df['budget_segment'] = pd.qcut(filtered_df['budget'], 4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
                fig = px.box(
                    filtered_df,
                    x='budget_segment',
                    y='revenue',
                    color='budget_segment',
                    color_discrete_sequence=px.colors.qualitative.Vivid,
                    points="all",
                    title='Revenue by Budget Segment',
                    hover_data=filtered_df.columns
                )
                fig.update_traces(quartilemethod="exclusive")
                fig.update_layout(
                    template='plotly_dark',
                    xaxis_title='Budget Segment',
                    yaxis_title='Revenue',
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.info(
        "Average and median revenue increase with higher budget segments, "
        "but the variability is also greater in the high-budget group. "
        "The relationship is not linear: while higher budgets often yield higher absolute revenue, "
        "the efficiency per dollar invested decreases at the top end. "
        "The optimal revenue segments are mid-high and high, but with careful attention to efficiency."
    )

            with col2:
                # ROI by budget segment (boxplot)
                st.subheader("ROI by Budget Segment")
                fig = px.box(
                    filtered_df,
                    x='budget_segment',
                    y='calculated_roi',
                    color='budget_segment',
                    color_discrete_sequence=px.colors.qualitative.Pastel2,
                    points="all",
                    title='ROI by Budget Segment',
                    hover_data=filtered_df.columns
                )
                fig.update_traces(quartilemethod="exclusive")
                fig.update_layout(
                    template='plotly_dark',
                    xaxis_title='Budget Segment',
                    yaxis_title='ROI',
                    showlegend=False,
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
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

            # Scatterplot moderno y dinámico con Plotly
            fig = go.Figure()

            # Paleta moderna para los tipos de campaña
            type_palette = px.colors.qualitative.Vivid
            type_list = high_performance['type'].unique()
            color_map = {t: type_palette[i % len(type_palette)] for i, t in enumerate(type_list)}

            for t in type_list:
                df_type = high_performance[high_performance['type'] == t]
                fig.add_trace(go.Scatter(
                    x=df_type['revenue'],
                    y=df_type['calculated_roi'],
                    mode='markers',
                    name=t,
                    marker=dict(
                        color=color_map[t],
                        size=12 + 18 * (df_type['budget'] - high_performance['budget'].min()) / (high_performance['budget'].max() - high_performance['budget'].min() + 1e-6),
                        line=dict(width=1, color='black'),
                        opacity=0.85
                    ),
                    hovertemplate=(
                        f"Type: {t}<br>"
                        "Revenue: %{x:,.0f}<br>"
                        "ROI: %{y:,.2f}<br>"
                        "Budget: %{marker.size:.0f}"
                        "<extra></extra>"
                    )
                ))

            fig.update_layout(
                template='plotly_dark',
                title='High-Performance Campaigns: ROI vs Revenue',
                xaxis_title='Revenue',
                yaxis_title='ROI',
                height=480,
                width=820,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=30, r=30, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.info(
    "The scatter plot shows that high ROI and high revenue are not always achieved simultaneously, but social media campaigns are most likely to reach both. "
    "Moderate budgets (not the highest) are often associated with optimal performance, indicating that efficiency, not just investment, is key. "
    "There are also outliers with exceptional ROI but moderate revenue, suggesting that niche targeting and operational excellence can yield outstanding results even without massive budgets."
)
            
            # --- Gráfico 3D interactivo de campañas de alto rendimiento ---
            st.subheader("3D Analysis of High-Performance Campaigns")

            # Filtros interactivos para tipo y canal
            types_3d = high_performance['type'].unique().tolist()
            channels_3d = high_performance['channel'].unique().tolist()
            selected_types_3d = st.multiselect("Filter Campaign Types (3D)", options=types_3d, default=types_3d, key="3d_types")
            selected_channels_3d = st.multiselect("Filter Channels (3D)", options=channels_3d, default=channels_3d, key="3d_channels")

            filtered_3d = high_performance[
                high_performance['type'].isin(selected_types_3d) &
                high_performance['channel'].isin(selected_channels_3d)
            ]

            # Paleta moderna para el color (usa calculated_roi como ejemplo)
            color_scale = px.colors.sequential.Viridis

            fig_3d = go.Figure(data=[go.Scatter3d(
                x=filtered_3d['calculated_roi'],
                y=filtered_3d['revenue'],
                z=filtered_3d['conversion_rate'],
                mode='markers',
                marker=dict(
                    size=filtered_3d['budget'] / 10000 + 6,
                    color=filtered_3d['calculated_roi'],  # Cambiado aquí
                    colorscale=color_scale,
                    colorbar=dict(title='ROI'),
                    opacity=0.8,
                    line=dict(width=1, color='black')
                ),
                text=filtered_3d['campaign_name'],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "ROI: %{x:.2f}<br>"
                    "Revenue: %{y:,.0f}<br>"
                    "Conversion Rate: %{z:.2f}<br>"
                    "Budget: %{marker.size:.0f}k<br>"
                    "ROI: %{marker.color:.2f}<extra></extra>"
                )
            )])

            fig_3d.update_layout(
                template='plotly_dark',
                title='3D Analysis of High-Performance Campaigns',
                scene=dict(
                    xaxis_title='ROI',
                    yaxis_title='Revenue',
                    zaxis_title='Conversion Rate',
                    bgcolor='black'
                ),
                height=540,
                width=820,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(fig_3d, use_container_width=True)

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

            # Usar Plotly para un gráfico moderno, dinámico y con fondo negro
            monthly_performance = monthly_performance.reset_index()
            fig = go.Figure()

            # Paleta moderna y vibrante
            color_map = {
                'revenue': '#00CC96',
                'calculated_roi': '#636EFA',
                'conversion_rate': '#FFA15A'
            }

            for metric in ['revenue', 'calculated_roi', 'conversion_rate']:
                fig.add_trace(go.Scatter(
                    x=monthly_performance['start_month_name'],
                    y=monthly_performance[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(width=3, color=color_map[metric]),
                    marker=dict(size=9, color=color_map[metric], line=dict(width=1, color='black')),
                    hovertemplate=f"Month: %{{x}}<br>{metric.replace('_', ' ').title()}: %{{y:,.2f}}<extra></extra>"
                ))

            fig.update_layout(
                template='plotly_dark',
                title='Monthly Performance Trends',
                xaxis_title='Month',
                yaxis_title='Value',
                height=380,
                width=820,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=30, r=30, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
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

            quarters = quarterly_performance.index.astype(str)
            color_map = {
                'revenue': '#00CC96',
                'calculated_roi': '#636EFA',
                'conversion_rate': '#FFA15A',
                'net_profit': '#EF553B'
            }

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=quarters,
                y=quarterly_performance['revenue'],
                name='Revenue',
                marker_color=color_map['revenue'],
                hovertemplate='Quarter: %{x}<br>Revenue: %{y:,.0f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=quarters,
                y=quarterly_performance['calculated_roi'],
                name='ROI',
                marker_color=color_map['calculated_roi'],
                hovertemplate='Quarter: %{x}<br>ROI: %{y:,.2f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=quarters,
                y=quarterly_performance['conversion_rate'],
                name='Conversion Rate',
                marker_color=color_map['conversion_rate'],
                hovertemplate='Quarter: %{x}<br>Conversion Rate: %{y:,.2f}<extra></extra>'
            ))
            fig.add_trace(go.Bar(
                x=quarters,
                y=quarterly_performance['net_profit'],
                name='Net Profit',
                marker_color=color_map['net_profit'],
                hovertemplate='Quarter: %{x}<br>Net Profit: %{y:,.0f}<extra></extra>'
            ))

            fig.update_layout(
                template='plotly_dark',
                barmode='group',
                title='Quarterly Performance Trends',
                xaxis_title='Quarter',
                yaxis_title='Value',
                height=420,
                width=820,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=30, r=30, t=60, b=40)
            )

            st.plotly_chart(fig, use_container_width=True)
            st.info(
    "Q2 and Q4 are the most profitable quarters, with Q2 leading in ROI and Q4 in conversion rate. "
    "Longer campaigns perform best in Q1, while short, intensive campaigns excel in Q4. "
    "This suggests that campaign duration and timing should be adapted to the seasonal business cycle for maximum impact."
)

            # Annual trends
            filtered_df['start_year'] = filtered_df['start_date'].dt.year
            yearly_performance = filtered_df.groupby('start_year')[['revenue', 'calculated_roi', 'conversion_rate']].mean()
            st.subheader("Annual Performance Trends")
            color_map = {
            'revenue': '#00CC96',
            'calculated_roi': '#636EFA',
            'conversion_rate': '#FFA15A'
        }

        years = yearly_performance.index.astype(str)

        col1, col2, col3 = st.columns(3)
        with col1:
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Scatter(
                x=years,
                y=yearly_performance['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(width=3, color=color_map['revenue']),
                marker=dict(size=8, color=color_map['revenue'], line=dict(width=1, color='black')),
                hovertemplate="Year: %{x}<br>Revenue: %{y:,.0f}<extra></extra>"
            ))
            fig_rev.update_layout(
                template='plotly_dark',
                title='Revenue by Year',
                xaxis_title='Year',
                yaxis_title='Revenue',
                height=300,
                width=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        with col2:
            fig_roi = go.Figure()
            fig_roi.add_trace(go.Scatter(
                x=years,
                y=yearly_performance['calculated_roi'],
                mode='lines+markers',
                name='ROI',
                line=dict(width=3, color=color_map['calculated_roi']),
                marker=dict(size=8, color=color_map['calculated_roi'], line=dict(width=1, color='black')),
                hovertemplate="Year: %{x}<br>ROI: %{y:,.2f}<extra></extra>"
            ))
            fig_roi.update_layout(
                template='plotly_dark',
                title='ROI by Year',
                xaxis_title='Year',
                yaxis_title='ROI',
                height=300,
                width=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_roi, use_container_width=True)

        with col3:
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=years,
                y=yearly_performance['conversion_rate'],
                mode='lines+markers',
                name='Conversion Rate',
                line=dict(width=3, color=color_map['conversion_rate']),
                marker=dict(size=8, color=color_map['conversion_rate'], line=dict(width=1, color='black')),
                hovertemplate="Year: %{x}<br>Conversion Rate: %{y:,.2f}<extra></extra>"
            ))
            fig_conv.update_layout(
                template='plotly_dark',
                title='Conversion Rate by Year',
                xaxis_title='Year',
                yaxis_title='Conversion Rate',
                height=300,
                width=320,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_conv, use_container_width=True)
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