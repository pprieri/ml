#First divide variables between categorical and numerical

    for col in num_cols:

        X_train[col], X_test[col] = num_input_median(X_train[col], X_test[col])


    for col in cat_cols:

        if col!='ROW ID':

            X_train[col], X_test[col] = cat_input_missing(X_train[col], X_test[col])

            #Mean encoding: not really useful here as we have only a few points
#             mean_target = pd.concat([X_train[col],y_train],axis=1).groupby(col)[TARGET].mean().to_dict()
#             X_train[col] = X_train[col].map(mean_target)
#             X_test[col] = X_test[col].map(mean_target)


            #Ordinal Encoding
            train_ordered_labels = pd.concat([X_train[col],y_train],axis=1).groupby(col)[TARGET].mean().sort_values().index.values
            ordinal_mapping = {k: i for i, k in enumerate(train_ordered_labels, 0)}
            ordinal_mapping['Other']=-1

            test_labels = X_test[col].unique()

            for label in test_labels:

                if label not in train_ordered_labels:

                    X_test[col] = X_test[col].replace(label,'Other')

            X_train[col] = X_train[col].map(ordinal_mapping)
            X_test[col] = X_test[col].map(ordinal_mapping)
