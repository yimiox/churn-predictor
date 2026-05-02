# train.py  — run this once: python train.py
from src.data_loader import get_train_test
from src.model import train_model, evaluate_model

X_train, X_test, y_train, y_test, features = get_train_test()
model = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)

print(f"\nTest AUC: {metrics['auc']}")
print(metrics["report"])