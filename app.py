import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# Streamlit UI
st.set_page_config(page_title="Market Basket Analysis Dashboard", layout="wide")
st.title("🛒 Market Basket Analysis (Apriori + Evaluation)")
st.markdown("Upload your **transaction dataset** and discover item associations using Apriori algorithm with ML-style evaluation!")

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

        # Apriori algorithm
        frequent_itemsets = apriori(train_basket, min_support=0.01, use_colnames=True)

        st.subheader("🏆 Top 10 Frequent Itemsets")
        st.dataframe(frequent_itemsets.sort_values(by='support', ascending=False).head(10))

        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

        st.subheader("📈 Top Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

        # Accuracy evaluation
        accuracy = evaluate_rules(rules, test_basket)
        st.success(f"✅ Accuracy on unseen transactions: **{accuracy:.2%}**")

        st.caption("Built with ❤️ using Python, Pandas, MLxtend, and Streamlit.")
