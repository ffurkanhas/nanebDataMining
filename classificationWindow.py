from tkinter import *
from tkinter import ttk
from bnnGui import columns_names,df
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score
from sklearn.metrics import roc_auc_score
import sklearn
from tkinter.scrolledtext import ScrolledText

class classificationWindow(Frame):

    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.create()

    def create(self):
        window = Toplevel(self)
        self.logisticRegression_frame = ttk.LabelFrame(window, text='LogisticRegression',
                                                       height=100)
        self.logisticRegression_frame.grid(column=5, row=1, columnspan=4, sticky='nesw')

        self.logisticRegression_frame.grid_configure(padx=5, pady=5)

        self.comboboxLogisticRegression = ttk.Combobox(self.logisticRegression_frame)
        self.comboboxLogisticRegression.grid(column=0, row=0)
        self.comboboxLogisticRegression['values'] = columns_names
        self.comboboxLogisticRegression.set('satisfaction_level')

        self.logisticRegression_addButton = ttk.Button(self.logisticRegression_frame, text='Add',
                                                       command=lambda: self.logisticRegressionAdd())
        self.logisticRegression_addButton.grid(column=1, row=0, sticky='nesw')
        self.LogisticRegressionList = set()
        ttk.Label(self.logisticRegression_frame, text='target_name').grid(column=0, row=1, sticky=W)
        self.comboboxLogisticRegression2 = ttk.Combobox(self.logisticRegression_frame)
        self.comboboxLogisticRegression2.grid(column=1, row=1)
        self.comboboxLogisticRegression2['values'] = columns_names
        self.comboboxLogisticRegression2.set('satisfaction_level')

        self.logisticRegression_addButton = ttk.Button(self.logisticRegression_frame, text='Clear',
                                                       command=lambda: self.logisticRegressionClear())
        self.logisticRegression_addButton.grid(column=2, row=0, sticky='nesw')

        ttk.Label(self.logisticRegression_frame, text='test_size').grid(column=0, row=2, sticky=W)
        self.logisticRegressionTestSize = Text(self.logisticRegression_frame, height=1, width=10)
        self.logisticRegressionTestSize.grid(column=1, row=2, sticky=W)

        ttk.Label(self.logisticRegression_frame, text='random_state').grid(column=0, row=3, sticky=W)
        self.logisticRegressionRandomState = Text(self.logisticRegression_frame, height=1, width=10)
        self.logisticRegressionRandomState.grid(column=1, row=3, sticky=W)

        self.logisticRegression_button = ttk.Button(self.logisticRegression_frame, text='LogisticRegression',
                                                    command=lambda: self.calculateLogisticRegression())
        self.logisticRegression_button.grid(column=1, row=4, columnspan=2, sticky='nesw')

        ttk.Label(self.logisticRegression_frame, text='List').grid(column=4, row=0, sticky=W)
        self.logisticRegressionResultList = Text(self.logisticRegression_frame, height=2, width=30)
        self.logisticRegressionResultList.grid(column=4, row=1, sticky=W, columnspan=3, rowspan=1)
        self.logisticRegressionResultList.config(state=DISABLED)

        ttk.Label(self.logisticRegression_frame, text='Result').grid(column=4, row=2, sticky=W)
        self.logisticRegressionResult = ScrolledText(self.logisticRegression_frame, height=7, width=30)
        self.logisticRegressionResult.grid(column=4, row=3, sticky=W, columnspan=3, rowspan=5)
        self.logisticRegressionResult.config(state=DISABLED)

        #############

        self.svm_frame = ttk.LabelFrame(window, text='SVM',
                                        height=100)
        self.svm_frame.grid(column=5, row=2, columnspan=4, sticky='nesw')

        self.svm_frame.grid_configure(padx=5, pady=5)

        self.comboboxSvm = ttk.Combobox(self.svm_frame)
        self.comboboxSvm.grid(column=0, row=0)
        self.comboboxSvm['values'] = columns_names
        self.comboboxSvm.set('satisfaction_level')

        self.svm_addButton = ttk.Button(self.svm_frame, text='Add',
                                        command=lambda: self.svmAdd())
        self.svm_addButton.grid(column=1, row=0, sticky='nesw')
        self.SvmList = set()
        ttk.Label(self.svm_frame, text='target_name').grid(column=0, row=1, sticky=W)
        self.comboboxSvm2 = ttk.Combobox(self.svm_frame)
        self.comboboxSvm2.grid(column=1, row=1)
        self.comboboxSvm2['values'] = columns_names
        self.comboboxSvm2.set('satisfaction_level')

        self.svm_addButton = ttk.Button(self.svm_frame, text='Clear',
                                        command=lambda: self.svmClear())
        self.svm_addButton.grid(column=2, row=0, sticky='nesw')

        ttk.Label(self.svm_frame, text='test_size').grid(column=0, row=2, sticky=W)
        self.svmTestSize = Text(self.svm_frame, height=1, width=10)
        self.svmTestSize.grid(column=1, row=2, sticky=W)

        ttk.Label(self.svm_frame, text='random_state').grid(column=0, row=3, sticky=W)
        self.svmRandomState = Text(self.svm_frame, height=1, width=10)
        self.svmRandomState.grid(column=1, row=3, sticky=W)

        self.svm_button = ttk.Button(self.svm_frame, text='SVM',
                                     command=lambda: self.calculateSvm())
        self.svm_button.grid(column=1, row=4, columnspan=2, sticky='nesw')

        ttk.Label(self.svm_frame, text='List').grid(column=4, row=0, sticky=W)
        self.svmResultList = Text(self.svm_frame, height=2, width=30)
        self.svmResultList.grid(column=4, row=1, sticky=W, columnspan=3, rowspan=1)
        self.svmResultList.config(state=DISABLED)

        ttk.Label(self.svm_frame, text='Result').grid(column=4, row=2, sticky=W)
        self.svmResult = ScrolledText(self.svm_frame, height=7, width=30)
        self.svmResult.grid(column=4, row=3, sticky=W, columnspan=3, rowspan=5)
        self.svmResult.config(state=DISABLED)

        ############

        for child in window.winfo_children():
            for child2 in child.winfo_children():
                child2.grid_configure(padx=5, pady=5)


    def logisticRegressionAdd(self):
        self.selection = self.comboboxLogisticRegression.get()
        self.LogisticRegressionList.add(self.selection )
        self.logisticRegressionResultList.config(state=NORMAL)
        self.logisticRegressionResultList.delete(1.0, END)
        self.logisticRegressionResultList.insert(END, self.LogisticRegressionList)
        self.logisticRegressionResultList.config(state=DISABLED)

    def logisticRegressionClear(self):
        self.LogisticRegressionList.clear()
        self.logisticRegressionResultList.config(state=NORMAL)
        self.logisticRegressionResultList.delete(1.0, END)
        self.logisticRegressionResultList.insert(END, self.LogisticRegressionList)
        self.logisticRegressionResultList.config(state=DISABLED)

    def calculateLogisticRegression(self):
        self.targetName = self.comboboxLogisticRegression2.get()
        target_name = self.targetName
        X = df.drop(target_name, axis=1)
        y = df[target_name]
        X_selected = df[list(self.LogisticRegressionList)]
        X_selected_train, X_selected_test, y_train, y_test = train_test_split(X_selected, y, test_size=float(self.logisticRegressionTestSize.get("1.0",END)),
                                                                              random_state=int(self.logisticRegressionRandomState.get("1.0",END)), stratify=y)
        LogModel = LogisticRegression().fit(X_selected_train, y_train)
        predictionOfLogModel = LogModel.predict(X_selected_test)
        text = "accuracy score: ", accuracy_score(y_test, predictionOfLogModel), "\n" + "recall_score: ", recall_score(y_test, predictionOfLogModel), "\n" + "precision_score: " , precision_score(y_test, predictionOfLogModel), "\n" + "roc_auc_score: " , roc_auc_score(y_test, predictionOfLogModel)
        self.logisticRegressionResult.config(state=NORMAL)
        self.logisticRegressionResult.delete(1.0, END)
        self.logisticRegressionResult.insert(END, text)
        self.logisticRegressionResult.config(state=DISABLED)

    def svmAdd(self):
        self.selection = self.comboboxSvm.get()
        self.SvmList.add(self.selection)
        self.svmResultList.config(state=NORMAL)
        self.svmResultList.delete(1.0, END)
        self.svmResultList.insert(END, self.SvmList)
        self.svmResultList.config(state=DISABLED)

    def svmClear(self):
        self.SvmList.clear()
        self.svmResultList.config(state=NORMAL)
        self.svmResultList.delete(1.0, END)
        self.svmResultList.insert(END, self.SvmList)
        self.svmResultList.config(state=DISABLED)

    def calculateSvm(self):
        self.targetName = self.comboboxSvm2.get()
        target_name = self.targetName
        X = df.drop(target_name, axis=1)
        y = df[target_name]
        X_selected = df[list(self.SvmList)]
        X_selected_train, X_selected_test, y_train, y_test = train_test_split(X_selected, y, test_size=float(self.svmTestSize.get("1.0",END)),
                                                                              random_state=int(self.svmRandomState.get("1.0",END)), stratify=y)
        svm_model = sklearn.svm.SVC().fit(X_selected_train, y_train)
        predictionOfLogModel = svm_model.predict(X_selected_test)
        text = "accuracy score: ", accuracy_score(y_test, predictionOfLogModel), "\n" + "recall_score: ", recall_score(y_test, predictionOfLogModel), "\n" + "precision_score: " , precision_score(y_test, predictionOfLogModel), "\n" + "roc_auc_score: " , roc_auc_score(y_test, predictionOfLogModel)
        self.svmResult.config(state=NORMAL)
        self.svmResult.delete(1.0, END)
        self.svmResult.insert(END, text)
        self.svmResult.config(state=DISABLED)