
# CNN_evaluate function: This function evaluates the model's performance on the test set.
def CNN_evaluate(model,test):
    loss, accuracy = model.evaluate(test)
    return loss,accuracy        