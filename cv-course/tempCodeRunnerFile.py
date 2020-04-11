pred = model.predict_classes(x_test)
print(classification_report(y_test, pred))