{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "19c15ed2-6f6b-4795-864b-89bf4b7571b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Load Libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split  \n",
    "from sklearn.neural_network import MLPClassifier \n",
    "from sklearn.preprocessing import StandardScaler  \n",
    "from sklearn.tree import DecisionTreeClassifier  \n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "7e2eab97-176e-4b58-b35e-9bd6aa638dc4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>BP</th>\n",
       "      <th>Cholesterol</th>\n",
       "      <th>Na_to_K</th>\n",
       "      <th>Drug</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>23</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>25.355</td>\n",
       "      <td>drugY</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>47</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>13.093</td>\n",
       "      <td>drugC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>47</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>10.114</td>\n",
       "      <td>drugC</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>28</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7.798</td>\n",
       "      <td>drugX</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>61</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>18.043</td>\n",
       "      <td>drugY</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age  Sex  BP  Cholesterol  Na_to_K   Drug\n",
       "0   23    1   2            1   25.355  drugY\n",
       "1   47    0   1            1   13.093  drugC\n",
       "2   47    0   1            1   10.114  drugC\n",
       "3   28    1   0            1    7.798  drugX\n",
       "4   61    1   1            1   18.043  drugY"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Load Data\n",
    "data = pd.read_csv('./drugdataset.csv')\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "789534b2-f520-415b-9bf8-82e2dd4f75d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['drugY', 'drugC', 'drugX', 'drugA', 'drugB'], dtype=object)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.Drug.unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e34bba9f-6ef1-4367-b2cc-39cbcbaca02c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create x and y variables\n",
    "X = data.drop('Drug',axis=1).to_numpy()\n",
    "y = data['Drug'].to_numpy()\n",
    "\n",
    "#Create Train and Test datasets\n",
    "from sklearn.model_selection import train_test_split  \n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size = 0.20,random_state=100)\n",
    "\n",
    "#Scale the data\n",
    "from sklearn.preprocessing import StandardScaler  \n",
    "sc = StandardScaler()  \n",
    "x_train2 = sc.fit_transform(X_train)\n",
    "x_test2 = sc.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "99723d58-1d4a-43f9-9df6-1a2514a2c534",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 5  0  0  0  0]\n",
      " [ 0  2  0  0  1]\n",
      " [ 0  0  3  0  0]\n",
      " [ 0  0  0 11  0]\n",
      " [ 0  0  0  1 17]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "       drugA       1.00      1.00      1.00         5\n",
      "       drugB       1.00      0.67      0.80         3\n",
      "       drugC       1.00      1.00      1.00         3\n",
      "       drugX       0.92      1.00      0.96        11\n",
      "       drugY       0.94      0.94      0.94        18\n",
      "\n",
      "    accuracy                           0.95        40\n",
      "   macro avg       0.97      0.92      0.94        40\n",
      "weighted avg       0.95      0.95      0.95        40\n",
      "\n"
     ]
    }
   ],
   "source": [
    "#Script for Neural Network\n",
    "from sklearn.neural_network import MLPClassifier  \n",
    "mlp = MLPClassifier(hidden_layer_sizes=(5, 4, 5),\n",
    "                    activation='relu',solver='adam',\n",
    "                    max_iter=10000,random_state=100)  \n",
    "mlp.fit(x_train2, y_train) \n",
    "predictions = mlp.predict(x_test2) \n",
    "\n",
    "#Evaluation Report and Matrix\n",
    "from sklearn.metrics import classification_report, confusion_matrix  \n",
    "target_names=['drugA', 'drugB', 'drugC', 'drugX', 'drugY']\n",
    "print(confusion_matrix(y_test,predictions))  \n",
    "print(classification_report(y_test,predictions,target_names=target_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "fc46bb98-219e-4292-8a60-6564e7b9c8da",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 5  0  0  0  0]\n",
      " [ 0  3  0  0  0]\n",
      " [ 0  0  2  0  1]\n",
      " [ 0  0  0 10  1]\n",
      " [ 0  0  0  1 17]]\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "       drugA       1.00      1.00      1.00         5\n",
      "       drugB       1.00      1.00      1.00         3\n",
      "       drugC       1.00      0.67      0.80         3\n",
      "       drugX       0.91      0.91      0.91        11\n",
      "       drugY       0.89      0.94      0.92        18\n",
      "\n",
      "    accuracy                           0.93        40\n",
      "   macro avg       0.96      0.90      0.93        40\n",
      "weighted avg       0.93      0.93      0.92        40\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "\n",
    "# Initialize Logistic Regression Model\n",
    "lr_classifier = LogisticRegression(max_iter=10000, random_state=100)\n",
    "\n",
    "# Train the model\n",
    "lr_classifier.fit(x_train2, y_train)\n",
    "\n",
    "# Make predictions\n",
    "predictions = lr_classifier.predict(x_test2)\n",
    "\n",
    "# Evaluate the model using classification metrics\n",
    "target_names = ['drugA', 'drugB', 'drugC', 'drugX', 'drugY']\n",
    "print(confusion_matrix(y_test, predictions))\n",
    "print(classification_report(y_test, predictions, target_names=target_names))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "7a11f3fc-918c-4899-a730-126f8fb00b13",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>BP</th>\n",
       "      <th>Cholesterol</th>\n",
       "      <th>Na_to_K</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>200.000000</td>\n",
       "      <td>200.000000</td>\n",
       "      <td>200.000000</td>\n",
       "      <td>200.000000</td>\n",
       "      <td>200.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>44.315000</td>\n",
       "      <td>0.480000</td>\n",
       "      <td>1.090000</td>\n",
       "      <td>0.515000</td>\n",
       "      <td>16.084485</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>16.544315</td>\n",
       "      <td>0.500854</td>\n",
       "      <td>0.821752</td>\n",
       "      <td>0.501029</td>\n",
       "      <td>7.223956</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>15.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>6.269000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>31.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>10.445500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>45.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>13.936500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>58.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>19.380000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>74.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>38.247000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Age         Sex          BP  Cholesterol     Na_to_K\n",
       "count  200.000000  200.000000  200.000000   200.000000  200.000000\n",
       "mean    44.315000    0.480000    1.090000     0.515000   16.084485\n",
       "std     16.544315    0.500854    0.821752     0.501029    7.223956\n",
       "min     15.000000    0.000000    0.000000     0.000000    6.269000\n",
       "25%     31.000000    0.000000    0.000000     0.000000   10.445500\n",
       "50%     45.000000    0.000000    1.000000     1.000000   13.936500\n",
       "75%     58.000000    1.000000    2.000000     1.000000   19.380000\n",
       "max     74.000000    1.000000    2.000000     1.000000   38.247000"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.describe()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
### Breakdown of Tests
1. **Data Preparation**:
   - Categorical variables are encoded.
   - Features are scaled for optimal performance.
2. **Model Training**:
   - An ANN with `(5, 4, 5)` hidden layers is trained on the data.
   - Logistic Regression is trained as a baseline.
3. **Model Evaluation**:
   - Confusion matrices and classification reports are generated for both models.

### Deployment
This project is designed for educational purposes. Deployment in production requires additional steps like deploying the model to a cloud environment or integrating it into a web application.

## Author
Komal Rajwani

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

