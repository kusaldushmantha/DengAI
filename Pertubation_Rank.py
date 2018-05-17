from sklearn import metrics
import numpy as np
import pandas as pd

def pertubation_rank(model, x, y, names, regression):
    errors = []
    
    for i in range(x.shape[1]):
        hold = np.array(x[:,i])
        np.random.shuffle(x[:,i])
        
        if(regression):
            pred = model.predict(x)
            error = metrics.mean_squared_error(y,pred)
        else:
            pred = model.predict_proba(x)
            error = metrics.log_loss(y,pred)
        
        errors.append(error)
        x[:, i] = hold
    
    max_error = np.max(errors)
    importance = [e/max_error for e in errors]
    
    data = {'name':names, 'error':errors, 'importance':importance}
    result = pd.DataFrame(data, columns=['name','error','importance'])
    result.sort_values(by=['importance'], ascending=[0], inplace=True)
    result.reset_index(inplace=True, drop=True)
    return result