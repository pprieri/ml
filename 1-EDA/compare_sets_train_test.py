integer_cols = X.select_dtypes(include=['int']).columns

print("Values in test but not in train")
for col in integer_cols:
    mismatched_codes = len(np.setdiff1d(X[col].unique(), X_test[col].unique()))
    print(f"{col:20} {mismatched_codes:4}")
