import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Matplotlib style for a cleaner look
plt.style.use('ggplot')

# Define a consistent color palette for Churn/Exited (Exited=Red, Retained=Blue)
CHURN_PALETTE = ["#4c72b0", "#c44e52"] # Blue for Retained, Red for Exited

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== HELPER FUNCTIONS ==========
@st.cache_data
def load_data():
    # Placeholder for the data loading. Assuming Churn_Modelling.csv is available.
    try:
        df = pd.read_csv("Churn_Modelling.csv")
    except FileNotFoundError:
        st.error("Error: Churn_Modelling.csv not found. Please ensure the file is in the correct location.")
        st.stop()
        
    df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)
    
    # Create a mapped column for visualization clarity
    df['Exited_Label'] = df['Exited'].map({0: 'Retained', 1: 'Exited'}) 
    return df

df = load_data()
# Get a numeric version for correlation (using the original 'Exited' column)
df_numeric = df.drop(columns=['Exited_Label']).copy()


# ========== PAGE TITLE & INTRODUCTION ==========
st.title("🏦 Customer Churn Analytics Dashboard")
st.markdown("""
This interactive dashboard provides **key metrics** and **visual analysis** of customer churn data. 
Use the tabs below to explore data distributions, behavioral patterns, and core correlation insights.
""")

st.markdown("---")

# ========== KEY PERFORMANCE INDICATORS (KPIs) - METRICS ==========
st.subheader("Key Performance Indicators (KPIs)")

total_customers = len(df)
exited_customers = df[df['Exited'] == 1].shape[0]
retained_customers = total_customers - exited_customers
churn_rate = (exited_customers / total_customers) * 100

col1, col2, col3 = st.columns(3)

col1.metric(
    label="Total Customers Analyzed",
    value=f"{total_customers:,}",
)

col2.metric(
    label="Exited Customers",
    value=f"{exited_customers:,}",
    delta=f"{retained_customers:,} Retained",
    delta_color="normal" # Green if high, Red if low, but normal for count
)

col3.metric(
    label="Overall Churn Rate",
    value=f"{churn_rate:.2f}%",
    delta=f"{(100 - churn_rate):.2f}% Retained",
    delta_color="inverse" # Red if churn is high, Green if low
)

st.markdown("---")

# ========== TABBED ANALYSIS SECTIONS ==========
tab1, tab2, tab3, tab4 = st.tabs([
    "📘 Data Overview",
    "📊 Univariate Distributions",
    "📈 Categorical Churn Analysis",
    "🔥 Correlation & Deep Dive"
])

# --- TAB 1: DATA OVERVIEW ---
with tab1:
    st.header("Dataset Structure and Sample")
    st.write("The dataset consists of 10,000 banking customers with key features used to predict churn.")

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

    col_shape, col_cols = st.columns(2)
    with col_shape:
        st.subheader("Data Shape")
        st.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    with col_cols:
        st.subheader("Features List")
        st.code(df.columns.tolist())

# --- TAB 2: UNIVARIATE DISTRIBUTIONS ---
with tab2:
    st.header("Distribution of Key Numerical Features")
    st.markdown("Understanding the distribution of features provides insight into customer segments and data spread.")

    col_dist1, col_dist2 = st.columns(2)

    # 1. Credit Score Distribution
    with col_dist1:
        st.subheader("Credit Score Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["CreditScore"], bins=30, kde=True, color=CHURN_PALETTE[0], ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title("Credit Score Distribution", fontsize=14)
        st.pyplot(fig)
        st.markdown("""
        * Distribution is slightly **left-skewed**.
        * Most customers have a **credit score between 600–750**.
        """)

    # 2. Age Distribution
    with col_dist2:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["Age"], bins=30, kde=True, color=CHURN_PALETTE[1], ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title("Age Distribution of Customers", fontsize=14)
        st.pyplot(fig)
        st.markdown("""
        * The distribution is **positively skewed** towards younger customers.
        * The core demographic is in the **30s to 40s** range.
        """)

    col_dist3, col_dist4 = st.columns(2)
    
    # 3. Balance Distribution
    with col_dist3:
        st.subheader("Balance Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["Balance"], bins=40, kde=True, color="#55a868", ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title("Account Balance Distribution", fontsize=14)
        st.pyplot(fig)
        st.markdown("""
        * A significant peak at **zero balance** suggests many inactive accounts.
        * The rest of the balances are spread out.
        """)

    # 4. Estimated Salary Distribution
    with col_dist4:
        st.subheader("Estimated Salary Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["EstimatedSalary"], bins=30, kde=True, color="#ccb974", ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title("Estimated Salary Distribution", fontsize=14)
        st.pyplot(fig)
        st.markdown("""
        * Salaries appear to be **uniformly distributed** across the range.
        * No major skew or clear bias exists.
        """)

# --- TAB 3: CATEGORICAL CHURN ANALYSIS ---
with tab3:
    st.header("Churn Patterns Across Categorical Groups")
    st.markdown("Analyzing churn rates based on demographic and account characteristics, with 'Exited' highlighted in red.")

    col_cat1, col_cat2 = st.columns(2)

    # 1. Gender vs Churn
    with col_cat1:
        st.subheader("Gender vs. Churn Status")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x="Gender", hue="Exited_Label", palette=CHURN_PALETTE, ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title("Customer Churn by Gender", fontsize=14)
        st.pyplot(fig)
        st.markdown("""
        * Female customers have a **higher proportional churn rate** compared to male customers.
        """)

    # 2. Geography vs Churn
    with col_cat2:
        st.subheader("Geography vs. Churn Status")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x="Geography", hue="Exited_Label", palette=CHURN_PALETTE, ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title("Customer Churn by Country", fontsize=14)
        st.pyplot(fig)
        st.markdown("""
        * Customers from **Germany** have a significantly **higher churn rate** than those in France or Spain.
        * Germany is a critical focus area for intervention.
        """)

    st.markdown("---")
    st.subheader("Churn Rate by Account Characteristics")

    col_acc1, col_acc2 = st.columns(2)
    
    # 3. Products vs Churn
    with col_acc1:
        st.subheader("Products Held vs. Churn Rate")
        fig, ax = plt.subplots(figsize=(8, 5))
        product_churn = df.groupby('NumOfProducts')['Exited'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        product_churn_exited = product_churn[product_churn['Exited'] == 1]
        
        sns.barplot(data=product_churn_exited, x='NumOfProducts', y='percent', color=CHURN_PALETTE[1], ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_title("Churn Rate by Number of Products", fontsize=14)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Number of Products")
        st.pyplot(fig)
        st.markdown("""
        * Customers holding **3 or 4 products** have an extremely high churn rate (approaching 100%).
        * This suggests a product design or dissatisfaction issue with multi-product customers.
        """)
    
    # 4. IsActiveMember vs Churn
    with col_acc2:
        st.subheader("Active Member Status vs. Churn")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x="IsActiveMember", hue="Exited_Label", palette=CHURN_PALETTE, ax=ax, edgecolor='black', linewidth=0.5)
        ax.set_xticklabels(['Inactive (0)', 'Active (1)'])
        ax.set_title("Churn by Active Status", fontsize=14)
        st.pyplot(fig)
        st.markdown("""
        * **Inactive members** are more likely to churn, but **active members** still contribute a large portion of the overall churn volume.
        """)


# --- TAB 4: CORRELATION & DEEP DIVE ---
with tab4:
    st.header("Feature Interrelationships")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap of Numerical Features")
    st.write("The heatmap shows linear relationships between the numerical variables, including the target variable (Exited).")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        df_numeric.corr(numeric_only=True),
        annot=True,
        cmap="YlOrRd", 
        fmt=".2f",
        ax=ax,
        linewidths=0.5,
        linecolor='black'
    )
    ax.set_title("Feature Correlation Matrix", fontsize=16)
    st.pyplot(fig)

    st.markdown("""
    **Key Correlation Findings (with Churn/Exited):**
    * **Age (0.29):** Shows the strongest positive correlation with churn. **Older customers are more likely to churn.**
    * **NumOfProducts (-0.05):** A weak negative correlation.
    * **IsActiveMember (-0.16):** A moderate negative correlation, meaning active members are less likely to churn.
    """)

    st.markdown("---")

    # Pairplot (optional — kept in expander)
    st.subheader("Deep Dive: Pairwise Relationships")
    # st.warning("Generating the pairplot is computationally intensive and may take a moment. Use sparingly.")
    with st.expander("Click to generate and view Pairplot"):
        try:
            pair_df = df_numeric[["CreditScore", "Age", "Balance", "EstimatedSalary", "Exited"]]
            fig = sns.pairplot(pair_df, hue="Exited", diag_kind="kde", palette=CHURN_PALETTE)
            st.pyplot(fig)
            st.success("Pairplot generated successfully.")
        except Exception as e:
            st.error(f"Could not generate pairplot: {e}")

st.markdown("---")

st.info("🎯 **Dashboard Summary:** The strongest predictors of churn are **Age**, **Geography (Germany)**, and having **3 or more products**. These are the critical areas for the business to focus on.")
