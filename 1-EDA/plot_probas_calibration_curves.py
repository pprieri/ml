##TPS-Nov22

import warnings
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,cross_val_score, GroupKFold, LeaveOneOut, StratifiedKFold, KFold
import matplotlib 

##A)

n_submission = 2
n_bins=10

disp = CalibrationDisplay.from_predictions(y,probas.iloc[:,n_submission],n_bins=n_bins)
plt.show()

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 5}

matplotlib.rc('font', **font)
warnings.filterwarnings("ignore")
plt.style.use('seaborn-whitegrid')

##B)
def plot_oof_histogram(name, oof, title=None):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.hist(oof, range=(0, 1), bins=100, density=True, color='#ffd700')
    ax1.set_title(f"{name} prediction histogram")
    ax1.set_facecolor('#0057b8') # blue
    
    CalibrationDisplay.from_predictions(y, oof, n_bins=20,
                                        strategy='quantile', ax=ax2, color='#ffd700')
    ax2.set_title('Probability calibration')
    ax2.set_facecolor('#0057b8') # blue
    ax2.legend('', frameon=False)
    if title is not None:
        plt.suptitle(title, y=1.0, fontsize=20)
    plt.show()
    
plot_oof_histogram('Original', probas.iloc[:,n_submission], title='Before calibration')
plot_oof_histogram('Calibrated', 
                   IsotonicRegression(out_of_bounds='clip')
                   .fit_transform(probas.iloc[:,n_submission], y)
                   .clip(0.001, 0.999),
                   title='After calibration')

##
def plot_sample_proba(df,y_true,col_probas,metric='log_loss',n_bins=10,type_mean='arithmetic',n_cols_figure=3,width=20,height=15, hspace=0.2, wspace=0.5,alpha=0.5,cv=2):
    
    plt.style.use('seaborn-whitegrid')
    n_rows_figure = len(col_probas)

    fig, axs = plt.subplots(n_rows_figure,n_cols_figure,figsize=(width,height))

    for i, col in enumerate(col_probas):

        if len(col)==1:
            proba=df[col]

        else:
            if type_mean=='arithmetic':
                proba=df[col].mean(axis=1)

        if metric=='log_loss':
            score=log_loss(y_true,proba)


        ##Histogram of original probas
        axs[i,0].hist(proba,bins=n_bins,alpha=0.3)                
        axs[i,0].set_title(f"{col}, Score: {np.round(score,2)}")

        ##Calibration
        colors = plt.cm.get_cmap("Dark2")

        lr = LogisticRegression(C=1)
        model = GaussianNB()
        model_isotonic = CalibratedClassifierCV(model, cv=cv, method="isotonic")
        model_sigmoid  = CalibratedClassifierCV(model, cv=cv, method="sigmoid")

        clf_list = [
        (lr, "Logistic"),
        #(model, f"{model.__class__.__name__}"),
        (model_isotonic, f"{model.__class__.__name__}" + " Isotonic"),
        (model_sigmoid, f"{model.__class__.__name__}" + " Sigmoid"),
        ]

        X = df[col]
        y = y_true

        X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=0)
        calibration_displays = {}

        for j,(clf,name) in enumerate(clf_list):

            clf.fit(X_train, y_train)
            display = CalibrationDisplay.from_estimator(
                clf,
                X_valid,
                y_valid,
                n_bins=n_bins,
                name=name,
                ax=axs[i,1],
                color=colors(j),
                alpha=alpha
            )
            calibration_displays[name] = display

        axs[i,1].set_title(f"Calibration plots {model.__class__.__name__}")

        disp = CalibrationDisplay.from_predictions(y_true,proba,n_bins=n_bins,ax=axs[i,1],label='original')

        title=f"Calibration curve {col}"
        axs[i,1].set(title=title, xlabel="Mean predicted probability", ylabel="Count")

        for j,(clf,name) in enumerate(clf_list):

            axs[i,2+j].hist(
                calibration_displays[name].y_prob,
                range=(0, 1),
                bins=n_bins,
                label=name,
                color=colors(j),
                alpha=alpha
            )
            score_cali = log_loss(y_valid,calibration_displays[name].y_prob)
            axs[i,2+j].set(title=f"{name} / {np.round(score_cali,2)}", xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()