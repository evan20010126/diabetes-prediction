from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def evalutate(best_model, X_test, y_test):

    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    return acc, auc, cm
