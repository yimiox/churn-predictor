# test_explainer.py
from src.data_loader import get_train_test
from src.model import load_model
from src.explainer import (shap_global_importance, shap_waterfall_fig,
                            lime_explanation, generate_recommendations)

X_train, X_test, y_train, y_test, features = get_train_test()
model = load_model()

# Global SHAP
importance_df = shap_global_importance(model, X_train.sample(200, random_state=42), features)
print("Top 5 features:\n", importance_df.head())

# Local SHAP + LIME for one customer
customer = X_test.iloc[[0]]
fig1 = shap_waterfall_fig(model, X_train, customer, features)
fig2, exp_df = lime_explanation(model, X_train, customer, features)
recs, level, color = generate_recommendations(exp_df, churn_probability=0.75)

print(f"\nRisk level: {level}")
for r in recs:
    print(f"  → {r['advice']}")

fig1.savefig("shap_waterfall.png")
fig2.savefig("lime_chart.png")
print("\nSaved shap_waterfall.png and lime_chart.png — open them to verify!")