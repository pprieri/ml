#Cross numericals and/or OH-encoded categoricals via Polynomial Features
# of sklearn:

features = 
target = 

X = df[features]
y = df[target]

poly_transformer = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
X_poly= pd.DataFrame(poly_transformer.fit_transform(X_3),columns=poly_transformer.get_feature_names(features),index=X.index)