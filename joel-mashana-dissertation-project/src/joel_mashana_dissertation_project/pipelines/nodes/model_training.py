# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split

# def train_decision_tree(data, target_column):
    
#     X = data.drop(target_column, axis=1)
#     # X = data.drop(columns_to_exclude, axis=1)
#     y = data[target_column]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     decision_tree = DecisionTreeClassifier(random_state=42)

#     decision_tree.fit(X_train, y_train)

#     predictions = decision_tree.predict(X_test)

#     # calculate_accuracy(y_test, predictions)
#     # store_and_print_classification_report(y_test, predictions)

#     # print_auc(decision_tree, X_test, y_test)
#     return decision_tree