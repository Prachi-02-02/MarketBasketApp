import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split

# Convert transaction data into basket format
def create_basket(data):
    basket = data.groupby(['TransactionID', 'Item'])['Item'].count().unstack().fillna(0)
    return basket.astype(bool)

# Evaluate rules on the test dataset
def evaluate_rules(rules, test_basket):
    correct = 0
    total = 0
    for _, row in rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        for _, transaction in test_basket.iterrows():
            if all(transaction.get(item, False) for item in antecedents):
                total += 1
                if all(transaction.get(item, False) for item in consequents):
                    correct += 1
    return correct / total if total > 0 else 0

# Streamlit UI Setup
st.set_page_config(page_title="🛒 Market Basket Dashboard", layout="wide", page_icon="🛍️")
st.markdown("""
    <style>
        .main {
            background-color: #f7f7f7;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🛒 Market Basket Analysis Dashboard")
st.markdown("Upload your **transaction dataset** and explore item associations using the Apriori algorithm with rule evaluation and interactive visuals.")

file = st.file_uploader("📁 Upload CSV File", type="csv")

if file:
    df = pd.read_csv(file)

    if "TransactionID" not in df.columns or "Item" not in df.columns:
        st.error("⚠️ The CSV file must contain 'TransactionID' and 'Item' columns.")
    else:
        st.subheader("📄 Preview of Uploaded Data")
        st.dataframe(df.head())

        # Show item frequency bar chart
        st.subheader("📊 Item Frequency Distribution")
        item_counts = df['Item'].value_counts().head(10)
        fig, ax = plt.subplots()
        item_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Item")
        ax.set_title("Top 10 Most Frequent Items")
        st.pyplot(fig)

        # Train/Test split
        train_ids, test_ids = train_test_split(df['TransactionID'].unique(), test_size=0.2, random_state=42)
        train_df = df[df['TransactionID'].isin(train_ids)]
        test_df = df[df['TransactionID'].isin(test_ids)]

        train_basket = create_basket(train_df)
        test_basket = create_basket(test_df)

        # Sliders for support and confidence
        st.sidebar.header("⚙️ Rule Settings")
        min_support = st.sidebar.slider("Minimum Support", min_value=0.01, max_value=0.5, value=0.01, step=0.01)
        min_confidence = st.sidebar.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.1)

        # Apriori algorithm
        frequent_itemsets = apriori(train_basket, min_support=min_support, use_colnames=True)

        st.subheader("🏆 Top 10 Frequent Itemsets")
        st.dataframe(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        st.subheader("📈 Top Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

        # Heatmap of lift vs confidence
        st.subheader("🔍 Lift vs. Confidence Heatmap")
        if not rules.empty:
            heatmap_data = rules[['lift', 'confidence']].dropna()
            fig2, ax2 = plt.subplots()
            sns.heatmap(heatmap_data.corr(), annot=True, cmap="YlGnBu", ax=ax2)
            ax2.set_title("Correlation between Lift and Confidence")
            st.pyplot(fig2)
        else:
            st.info("No rules found with current thresholds. Try lowering support or confidence.")

        # Accuracy evaluation
        accuracy = evaluate_rules(rules, test_basket)
        st.success(f"✅ Accuracy on unseen transactions: **{accuracy:.2%}**")

        st.caption("Built with ❤️ using Python, Pandas, MLxtend, and Streamlit.")
