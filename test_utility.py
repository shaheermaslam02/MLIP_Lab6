import pytest
import pandas as pd
import numpy as np
from prediction_demo import data_preparation,data_split,train_model,eval_model

@pytest.fixture
def housing_data_sample():
    return pd.DataFrame(
      data ={
      'price':[13300000,12250000],
      'area':[7420,8960],
    	'bedrooms':[4,4],	
      'bathrooms':[2,4],	
      'stories':[3,4],	
      'mainroad':["yes","yes"],	
      'guestroom':["no","no"],	
      'basement':["no","no"],	
      'hotwaterheating':["no","no"],	
      'airconditioning':["yes","yes"],	
      'parking':[2,3],
      'prefarea':["yes","no"],	
      'furnishingstatus':["furnished","unfurnished"]}
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints has same length
    assert feature_df.shape[0]==len(target_series)

    #Feature only has numerical values
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number,np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split(feature_target_sample):
    # Call the data_split function
    return_tuple = data_split(*feature_target_sample)
    
    # Check if the returned tuple contains four elements (train and test sets for both features and target)
    assert len(return_tuple) == 4, "The data_split function should return 4 elements."

    # Unpack the returned tuple
    X_train, X_test, y_train, y_test = return_tuple
    
    # Check if the train and test splits have the correct number of rows
    assert X_train.shape[0] == y_train.shape[0], "Training feature set and target should have the same number of rows."
    assert X_test.shape[0] == y_test.shape[0], "Test feature set and target should have the same number of rows."

    # Check if the total number of rows is the same as the original dataset
    assert (X_train.shape[0] + X_test.shape[0]) == feature_target_sample[0].shape[0], "The total number of rows should match the original feature set."

    # Optionally, check if the split sizes are approximately correct
    train_size = X_train.shape[0] / feature_target_sample[0].shape[0]
    test_size = X_test.shape[0] / feature_target_sample[0].shape[0]
    
    assert np.isclose(train_size + test_size, 1.0, atol=0.01), "Train and test sizes should add up to 1."
