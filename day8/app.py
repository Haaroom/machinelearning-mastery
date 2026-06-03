import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Advanced EDA Dashboard", layout="wide")
st.title("Advanced EDA Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
        
    st.subheader("Column Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Datatype": df.dtypes.astype(str)
    })
    st.dataframe(info_df)
    
    st.subheader("Missing Values")
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_df.sort_values(by="Missing %", ascending=False))
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.subheader("Feature Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Numeric Features: {len(numeric_cols)}")
    with col2:
        st.write(f"Categorical Features: {len(categorical_cols)}")
        
    if numeric_cols:
        st.subheader("Numeric Feature Distributions")
        selected_num = st.selectbox("Select Numerical Column", numeric_cols)
        
        # Display selected numeric distribution and box plot
        fig_hist = px.histogram(df, x=selected_num, title=f"Distribution of {selected_num}")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Using px.box instead of the non-existent st.box_chart
        fig_box = px.box(df, y=selected_num, title=f"Box Plot of {selected_num}")
        st.plotly_chart(fig_box, use_container_width=True)
        
        st.subheader(f"Statistics for {selected_num}")
        stats = pd.DataFrame({
            "Metric": ["Mean", "Median", "Min", "Max", "Std"],
            "Value": [
                df[selected_num].mean(),
                df[selected_num].median(),
                df[selected_num].min(),
                df[selected_num].max(),
                df[selected_num].std()
            ]
        })
        st.dataframe(stats)
        
    if categorical_cols:
        st.subheader("Categorical Analysis")
        selected_cat = st.selectbox("Select Categorical Column", categorical_cols)
        
        # Display summary of all categorical columns
        cat_stats = pd.DataFrame({
            "Column": categorical_cols,
            "Unique Values": [df[col].nunique() for col in categorical_cols],
            "Top Value": [df[col].mode()[0] for col in categorical_cols],
            "Top Value Count": [df[col].value_counts().iloc[0] for col in categorical_cols]
        })
        st.dataframe(cat_stats)
        
        # Display bar chart for the selected categorical column
        counts = df[selected_cat].value_counts().reset_index()
        counts.columns = [selected_cat, 'Count']
        
        fig_cat = px.bar(counts, x=selected_cat, y='Count', title=f"{selected_cat} Distribution")
        st.plotly_chart(fig_cat, use_container_width=True)
