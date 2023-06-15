import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %%
def create_model(data):
    X = data.drop(['diagnosis'], axis=1)'])
    y = data['diagnosis']
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, scaler

# %%
def test_model(model, scaler):
    # Test the model
    X_test = pd.read_csv("Source/test.csv")
    X_test = X_test.drop(['Unnamed: 32', 'id'], axis=1)
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    return y_pred

# %%
def get_clean_data():
    data = pd.read_csv("Source/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0}) # M = Malignant, B = Benign
    return data

get_clean_data()

# %% 
def main():
    data = get_clean_data()
    model = create_model(data)

if __name__ == "__main__":
    main()
    
# %%
