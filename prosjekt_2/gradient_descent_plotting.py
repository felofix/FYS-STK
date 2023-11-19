import numpy as np
import matplotlib.pyplot as plt 
from gradient_descent import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import seaborn as sns

# Franke Function

np.random.seed(19)

def FrankeFunction(x, y):
	term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
	term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
	term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))	
	term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
	return term1 + term2 + term3 + term4

def create_design_matrix(x, y, order):
    if (len(x) != len(y)):
        AssertionError("x and y must have the same length!")
    
    number_of_combinations = int((order+1)*(order+2)/2)
    X = np.zeros((len(x), number_of_combinations))
    
    col_nr = 0
    for i in range(order+1):
        for j in range(i+1):
            X[:,col_nr] = x**(i-j)*y**(j)
            col_nr += 1
            
    return X

def scale(X_train, X_test):
    # Scaling function.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def find_optimal_betas_OLS(X, z):
    return np.linalg.pinv(X.T@X)@X.T@z

def calculate_OLS_loss(y_true, y_pred):

    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")

    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def CostRidge(y_true, y_pred, lamb, beta):
    n = len(y_true)
    if n != len(y_pred):
        raise ValueError("The length of y_true and y_pred must be the same.")
    
    return np.sum((y_true - y_pred) ** 2) / len(y_true) + lamb* beta.T @ beta


# Generate data
n = 20
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
x,y = np.meshgrid(x,y)
z = FrankeFunction(x, y).ravel()
z+= 0.2*np.random.randn(z.size)

# Make life easier, flat x and y 
x_flat = x.flatten()
y_flat = y.flatten()
z_flat = z.flatten()


X = create_design_matrix(x_flat, y_flat, 6)
X_train, X_test, z_train, z_test = train_test_split(X, z_flat, test_size=0.2)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

z_train = z_train.reshape(-1,1)
z_test = z_test.reshape(-1,1)

def find_best_learning_rate_OLS(model, X, z, learning_rates, max_epochs):
    best_model_error = float('inf')
    optimal_learning_rate = None

    for learn in learning_rates:
        model.change_learning_rate(learn)
        model.reset_model()
        model.fit()
        z_pred = model.predict(X_test)
        model_error = calculate_OLS_loss(z_test, z_pred + z_test.mean())

        if model_error < best_model_error:
            best_model_error = model_error
            optimal_learning_rate = learn

    return best_model_error, optimal_learning_rate


def collect_errors_ridge(model,learning_rates, lamb_values):

    errors = np.zeros((len(learning_rates), len(lamb_values)))

    for i, learn in enumerate(learning_rates):
        for j, lambda_val in enumerate(lamb_values):
            model.change_learning_rate(learn)
            model.change_lamb(lambda_val)
            model.reset_model()
            model.fit()
            betas = model.betas()
            z_pred = model.predict(X_test)
            model_error = CostRidge(z_test, z_pred + z_test.mean(), lambda_val, betas)
            errors[i, j] = model_error

    return errors

def plot_heatmap_ridge(errors, learning_rates, lamb_values, name):

    plt.figure(figsize=(10, 8))

    ax = sns.heatmap(errors, annot=True, fmt=".5g", cmap='hot', 
                    xticklabels=np.round(lamb_values, 5), 
                    yticklabels=np.round(learning_rates, 5), 
                    cbar_kws={'label': 'Error'})

    ax.set_xlabel('Lambda Values')
    ax.set_ylabel('Learning Rates')
    ax.set_title(f'Heatmap of Ridge Regression Errors {name}')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig("plots/" + "heatmapRidge" + name + ".pdf")


def compare_momentum(optimal_lr, optimal_lr_momentum, max_epochs):
    error_standard = []
    error_momentum = []

    model = GradientDescent(X_train, z_train, optimal_lr, 0) 

    for epoch in range(max_epochs):
        model.reset_model()
        model = GradientDescent(X_train, z_train, optimal_lr, epoch) 
        model.fit()
        z_pred = model.predict(X_test)
        error = calculate_OLS_loss(z_test, z_pred + z_test.mean())
     
     
    model_momentum = GDWithMomentum(X_train, z_train, optimal_lr_momentum, 0) 

    for epoch in range(max_epochs):
        model_momentum.reset_model()
        model_momentum = GDWithMomentum(X_train, z_train, optimal_lr, epoch) 
        model_momentum.fit()
        z_pred = model_momentum.predict(X_test)
        error = calculate_OLS_loss(z_test, z_pred + z_test.mean())  
        error_momentum.append(error)
        
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(error_momentum), 1), error_momentum, label='Gradient Descent with Momentum')
    plt.plot(np.arange(0, len(error_standard), 1), error_standard, label='Standard Gradient Descent')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE Error')
    plt.title('Error Comparison over Epochs')c
    plt.savefig("plots/momentum_comparison.pdf")


learning_rates = np.linspace(1e-3, 1e-2, 6) 
lamb_values = np.linspace(1e-5, 1e-2, 6)  


def ols_errors():
    model = GradientDescent(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error Gradient Descent {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")

    model = GDWithMomentum(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error Gradient Descent with momentum {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")

    model = SGD(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error SGD {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")    

    model = SGDWithMomentum(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error SGD with momentum {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")


    model = Adagrad(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error Adagrad {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")

    model = AdagradWithMomentum(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error Adagrad with momentum {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")

    model = SGDAdagrad(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error SGD Adagrad {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")

    model = SGDAdagradWithMomentum(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error SGD Adagrad with momentum {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")

    model = RMSprop(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error RMSProp {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")

    model = SGD_RMSprop(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error SGD RMSProp {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")

    model = Adam(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error Adam {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")

    model = SGD_Adam(X_train, z_train, None, 1000)
    best_model_error, optimal_learning_rate = find_best_learning_rate_OLS(model, X_train, z_test, learning_rates, 1000)
    print(f"Best error SGD Adam {best_model_error:.3f} optimal learning rate {optimal_learning_rate:.3f}")
    
def ridge_plotting(learning_rates, lamb_values):
    model = GradientDescent(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "Gradient Descent")

    model = GDWithMomentum(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "Gradient Descent with momentum")
    

    model = SGDWithMomentum(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "SGD with momentum")


    model = Adagrad(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "Adagrad")


    model = AdagradWithMomentum(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "Adagrad with momentum")


    model = SGDAdagrad(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "SGD Adagrad")

    model = SGDAdagradWithMomentum(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "SGD Adagrad with momentum")


    model = RMSprop(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "RMSProp")


    model = SGD_RMSprop(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "SGD RMSProp")


    model = Adam(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "Adam")


    model = SGD_Adam(X_train, z_train, None, 500)
    errors = collect_errors_ridge(model, learning_rates, lamb_values)
    plot_heatmap_ridge(errors, learning_rates, lamb_values, "SGD Adam")


ridge_plotting(learning_rates, lamb_values)
compare_momentum(0.001, 0.005 , 1000)
ols_errors()