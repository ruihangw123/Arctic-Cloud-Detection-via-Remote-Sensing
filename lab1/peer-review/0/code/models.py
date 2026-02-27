# models will be placed here.

#################################
## Train-Validation-Test Split ##
#################################
from sklearn.model_selection import train_test_split

def train_validation_test(df, train = 0.5,
                            valid = 0.3,
                            test = 0.2,
                            shuffle = True
                            ):
    dtrain, dtest = train_test_split(df, test_size=test , shuffle = shuffle) 
    dtrain, dvalid = train_test_split(dtrain, test_size= (
        valid/(valid + train)
    ))
    return dtrain, dvalid, dtest 

########################
## CDR Implementation ##
########################

def cdr():

    return

#########################
## Logistic Regression ##
#########################


##########################
## Personal Model
##########################