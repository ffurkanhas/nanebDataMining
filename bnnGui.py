from tkinter import *
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
import classificationWindow as classificationModule
from tkinter.scrolledtext import ScrolledText
import prince

df = pd.read_csv("HR_comma_sep.csv")
constantdf = df
columns_names = df.columns.tolist()
plotNames = ['plot', 'countplot', 'barplot', 'barplotPercentage', 'kdeplot', 'distplot', 'stripplot',
             'pointplot', 'lvplot', 'factorplot']

class App(Frame):

    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        self.root = parent
        self.init_gui()

    def init_gui(self):
        self.root.title('HR Analytics Tool')

        self.grid(column=0, row=0)
        ttk.Label(self, text='naneB Tool').grid(column=0, row=0,
                                          columnspan=10)

        ############

        self.plot_frame = ttk.LabelFrame(self, text='Plot',
                                         height=100)
        self.plot_frame.grid(column=0, row=1, columnspan=4, sticky='nesw')

        self.plot_frame.grid_configure(padx=5, pady=5)

        self.draw_button = ttk.Button(self.plot_frame, text='Draw', command=lambda: self.plotDrawOk())
        self.draw_button.grid(column=0, row=4, columnspan=4)

        ttk.Label(self.plot_frame, text='X').grid(column=0, row=1)

        ttk.Label(self.plot_frame, text='Y').grid(column=2, row=1)

        ttk.Label(self.plot_frame, text='Left').grid(column=3, row=1)

        self.combobox = ttk.Combobox(self.plot_frame)
        self.combobox.grid(column=0, row=2)
        self.combobox['values'] = columns_names
        self.combobox.set('satisfaction_level')

        self.combobox2 = ttk.Combobox(self.plot_frame)
        self.combobox2.grid(column=2, row=2)
        self.combobox2['values'] = columns_names
        self.combobox2.set('satisfaction_level')

        self.comboboxLeft = ttk.Combobox(self.plot_frame)
        self.comboboxLeft.grid(column=3, row=2)
        self.comboboxLeft['values'] = [0, 1]
        self.comboboxLeft.set(0)

        self.combobox3 = ttk.Combobox(self.plot_frame)
        self.combobox3.bind("<<ComboboxSelected>>", self.comboUpdate)
        self.combobox3.grid(row=3, column=0, columnspan=4)
        self.combobox3['values'] = plotNames
        self.combobox3.set('plot')

        self.checkCmd = IntVar()
        self.hueCheck = Checkbutton(self.plot_frame, variable=self.checkCmd, onvalue=1, offvalue=0, text="Hue=left")
        self.hueCheck.grid(column=3, row=4)
        self.hueCheck.configure(state="disabled")

        ############

        self.heatmap_frame = ttk.LabelFrame(self, text='HeatMap',
                                         height=100)
        self.heatmap_frame.grid(column=0, row=2, columnspan=4, sticky='nesw')

        self.heatmap_frame.grid_configure(padx=5, pady=5)

        self.heatmap_button = ttk.Button(self.heatmap_frame, text='HeatMap',command=lambda: self.heatMapOk())
        self.heatmap_button.place(relx=0.5, rely=0.5, anchor=CENTER)

        ###########

        self.cube_frame = ttk.LabelFrame(self, text='Cube',
                                         height=100)
        self.cube_frame.grid(column=0, row=3, columnspan=4, sticky='nesw')

        self.cube_frame.grid_configure(padx=5, pady=5)

        self.cube_button = ttk.Button(self.cube_frame, text='Cube',command=lambda: self.cubeOk())
        self.cube_button.grid(column=1, row=4, columnspan=2, sticky='nesw')

        self.comboboxCube = ttk.Combobox(self.cube_frame)
        self.comboboxCube.grid(column=0, row=2)
        self.comboboxCube['values'] = columns_names
        self.comboboxCube.set('satisfaction_level')

        self.comboboxCube2 = ttk.Combobox(self.cube_frame)
        self.comboboxCube2.grid(column=2, row=2)
        self.comboboxCube2['values'] = columns_names
        self.comboboxCube2.set('satisfaction_level')

        self.comboboxCube3 = ttk.Combobox(self.cube_frame)
        self.comboboxCube3.grid(row=2, column=3, columnspan=4)
        self.comboboxCube3['values'] = columns_names
        self.comboboxCube3.set('satisfaction_level')

        ###########

        self.histogram_frame = ttk.LabelFrame(self, text='Histogram',height=100)

        self.histogram_frame.grid(column=0, row=4, columnspan=4, sticky='nesw')

        self.histogram_frame.grid_configure(padx=5, pady=5)

        self.comboboxHistogram = ttk.Combobox(self.histogram_frame)
        self.comboboxHistogram.grid(column=0, row=0)
        self.comboboxHistogram['values'] = columns_names
        self.comboboxHistogram.set('satisfaction_level')

        ttk.Label(self.histogram_frame, text='Bin').grid(column=1, row=0)

        self.textBoxHistogram = Text(self.histogram_frame, height=1, width=20)
        self.textBoxHistogram.grid(column=2, row=0)

        self.histogram_button = ttk.Button(self.histogram_frame, text='Histogram', command=lambda: self.histogramOk())
        self.histogram_button.grid(column=1, row=1, columnspan=2, sticky='nesw')

        ############

        self.kmeans_frame = ttk.LabelFrame(self, text='KMeans',
                                         height=100)
        self.kmeans_frame.grid(column=0, row=5, columnspan=4, sticky='nesw')

        self.kmeans_frame.grid_configure(padx=5, pady=5)

        self.comboboxKmeans = ttk.Combobox(self.kmeans_frame)
        self.comboboxKmeans.grid(column=0, row=0)
        self.comboboxKmeans['values'] = columns_names
        self.comboboxKmeans.set('satisfaction_level')

        self.comboboxKmeans2 = ttk.Combobox(self.kmeans_frame)
        self.comboboxKmeans2.grid(column=1, row=0)
        self.comboboxKmeans2['values'] = columns_names
        self.comboboxKmeans2.set('satisfaction_level')

        ttk.Label(self.kmeans_frame, text='n').grid(column=3, row=0)

        self.textBox = Text(self.kmeans_frame, height=1, width=10)
        self.textBox.grid(column=4, row=0)

        self.kmeans_button = ttk.Button(self.kmeans_frame, text='KMeans', command=lambda: self.kmenasOk())
        self.kmeans_button.grid(column=1, row=4, columnspan=2, sticky='nesw')

        #############

        self.cluster_frame = ttk.LabelFrame(self, text='Classification',
                                        height=100)
        self.cluster_frame.grid(column=4, row=1, columnspan=4,sticky='nesw')

        self.cluster_button = ttk.Button(self.cluster_frame, text='Classification',
                                     command=lambda: self.classification())
        self.cluster_button.grid(column=0, row=0, sticky='nesw')

        #############

        self.fiter_frame = ttk.LabelFrame(self, text='Filter Data',
                                            height=100)
        self.fiter_frame.grid(column=4, row=2, columnspan=4, sticky='nesw')

        self.filter_button = ttk.Button(self.fiter_frame, text='Filter',
                                         command=lambda: self.filterData())
        self.filter_button.grid(column=0, row=0, sticky='nesw')

        #############

        self.orange_frame = ttk.LabelFrame(self, text='Run Orange Tool',
                                          height=100)
        self.orange_frame.grid(column=4, row=3, columnspan=4, sticky='nesw')

        self.orange_button = ttk.Button(self.orange_frame, text='Run',
                                        command=lambda: self.runOrange())
        self.orange_button.grid(column=0, row=0, sticky='nesw')

        #############

        self.cross_frame = ttk.LabelFrame(self.cluster_frame, text='Cross Validation',
                                           height=100)
        self.cross_frame.grid_configure(padx=5,pady=5)
        self.cross_frame.grid(column=0, row=6, columnspan=4, sticky='nesw')

        self.cross_button = ttk.Button(self.cross_frame, text='Run',
                                        command=lambda: self.crossRun())
        self.cross_button.grid_configure(padx=5,pady=5)
        self.cross_button.grid(column=0, row=0, sticky='nesw')

        #############

        self.factor_analysis = ttk.LabelFrame(self, text='Factor Analysis',
                                          height=100)
        self.factor_analysis.grid_configure(padx=5, pady=5)
        self.factor_analysis.grid(column=0, row=6, columnspan=4, sticky='nesw')

        ttk.Label(self.factor_analysis, text='Component').grid(column=0, row=0)

        self.textBoxComponent = Text(self.factor_analysis, height=1, width=10)
        self.textBoxComponent.grid(column=1, row=0)

        ttk.Label(self.factor_analysis, text='Test Size').grid(column=0, row=1)

        self.textBoxTestSize = Text(self.factor_analysis, height=1, width=10)
        self.textBoxTestSize.grid(column=1, row=1)

        self.factor_button = ttk.Button(self.factor_analysis, text='Run',
                                       command=lambda: self.factorAnalysis())
        self.factor_button.grid_configure(padx=5, pady=5)
        self.factor_button.grid(column=0, row=2, sticky='nesw')

        #############

        for child in self.winfo_children():
            child.grid_configure(padx=5, pady=5)
            for child2 in child.winfo_children():
                child2.grid_configure(padx=5, pady=5)

    def plotDrawOk(self):
        self.selection = self.combobox.get()
        self.selection2 = self.combobox2.get()
        self.selection3 = self.combobox3.get()
        self.selectionLeft = self.comboboxLeft.get()
        window = Toplevel(self.plot_frame)
        if self.selection3 == 'plot':
            x = df[self.selection]
            y = df[self.selection2]
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            plt.plot(x[df.left == int(self.selectionLeft)], y[df.left == int(self.selectionLeft)], 'o', alpha=0.1)
            plt.ylabel(self.selection2)
            plt.title('Employees who left')
            plt.xlabel(self.selection)
            self.canvas.draw()
        if self.selection3 == 'countplot':
            self.fig, self.ax = plt.subplots(figsize=(15, 5))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            if self.checkCmd.get() == 1:
                sns.countplot(y=self.selection, hue='left', data=df).set_title('Employee ' + self.selection + ' Turnover Distribution');
            if self.checkCmd.get() == 0:
                sns.countplot(y=self.selection, data=df).set_title('Employee ' + self.selection + ' Turnover Distribution');
        if self.selection3 == 'barplot':
            self.fig, self.ax = plt.subplots(figsize=(15, 5))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            if self.checkCmd.get() == 1:
                sns.barplot(df[self.selection], df[self.selection2],hue=df.left)
            if self.checkCmd.get() == 0:
                sns.barplot(df[self.selection], df[self.selection2])
        if self.selection3 == 'barplotPercentage':
            self.fig, self.ax = plt.subplots(figsize=(15, 5))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            if self.checkCmd.get() == 1:
                ax = sns.barplot(x=self.selection, y=self.selection, hue="left", data=df,estimator=lambda x: len(x) / len(df) * 100)
            if self.checkCmd.get() == 0:
                ax = sns.barplot(x=self.selection, y=self.selection, data=df,estimator=lambda x: len(x) / len(df) * 100)
            ax.set(ylabel="Percent")
        if self.selection3 == 'kdeplot':
            self.fig, self.ax = plt.subplots(figsize=(15, 5))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            ax = sns.kdeplot(df.loc[(df['left'] == 0), self.selection], color='b', shade=True, label='no turnover')
            ax = sns.kdeplot(df.loc[(df['left'] == 1), self.selection], color='r', shade=True, label='turnover')
            plt.title('Employee ' + self.selection + ' Distribution - Turnover V.S. No Turnover')
        if self.selection3 == 'distplot':
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            plt.xlabel(self.selection, fontsize=12)
            plt.ylabel('distribution', fontsize=12)
            sns.distplot(df[self.selection], kde=True)
        if self.selection3 == 'stripplot':
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            sns.stripplot(df[self.selection], df[self.selection2])
        if self.selection3 == 'pointplot':
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            sns.pointplot(df[self.selection], df[self.selection2])
        if self.selection3 == 'lvplot':
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            sns.lvplot(df[self.selection], df[self.selection2])
        if self.selection3 == 'factorplot':
            self.fig, self.ax = plt.subplots(figsize=(7, 7))
            self.canvas = FigureCanvasTkAgg(self.fig, window)
            self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
            sns.factorplot(y=self.selection2, x=self.selection, data=df, kind="box", ax=self.ax)
            plt.tight_layout()
            plt.gcf().clear()

    def comboUpdate(self, event=None):
        if event.widget.get() == 'plot':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="enabled")
            self.comboboxLeft.configure(state="enabled")
            self.hueCheck.configure(state="disabled")
        if event.widget.get() == 'barplot':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="enabled")
            self.comboboxLeft.configure(state="disabled")
            self.hueCheck.configure(state="active")
        if event.widget.get() == 'barplotPercentage':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="disabled")
            self.comboboxLeft.configure(state="disabled")
            self.hueCheck.configure(state="active")
        if event.widget.get() == 'countplot':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="disabled")
            self.comboboxLeft.configure(state="disabled")
            self.hueCheck.configure(state="active")
        if event.widget.get() == 'kdeplot':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="disabled")
            self.comboboxLeft.configure(state="disabled")
            self.hueCheck.configure(state="disabled")
        if event.widget.get() == 'distplot':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="disabled")
            self.comboboxLeft.configure(state="disabled")
            self.hueCheck.configure(state="disabled")
        if event.widget.get() == 'stripplot':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="enabled")
            self.comboboxLeft.configure(state="disabled")
            self.hueCheck.configure(state="disabled")
        if event.widget.get() == 'pointplot':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="enabled")
            self.comboboxLeft.configure(state="disabled")
            self.hueCheck.configure(state="disabled")
        if event.widget.get() == 'lvplot':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="enabled")
            self.comboboxLeft.configure(state="disabled")
            self.hueCheck.configure(state="disabled")
        if event.widget.get() == 'factorplot':
            self.combobox.configure(state="enabled")
            self.combobox2.configure(state="enabled")
            self.comboboxLeft.configure(state="disabled")
            self.hueCheck.configure(state="disabled")

    def cubeOk(self):
        self.selection = self.comboboxCube.get()
        self.selection2 = self.comboboxCube2.get()
        self.selection3 = self.comboboxCube3.get()
        window = Toplevel(self.plot_frame)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, window)
        self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
        x = df[self.selection]
        y = df[self.selection2]
        z = df[self.selection3]
        c = df['left']
        _ = self.ax.scatter(xs=x, ys=y, zs=z, c=c)
        _ = self.ax.set_xlabel(self.selection)
        _ = self.ax.set_ylabel(self.selection2)
        _ = self.ax.set_zlabel(self.selection3)
        _ = plt.title('Plot 1: Multivariate Visualization with Number of Projects by Color')

    def heatMapOk(self):
        window = Toplevel(self.plot_frame)
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, window)
        self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
        correlation = df.corr()
        sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
        plt.title('Correlation between different fearures')

    def histogramOk(self):
        self.selection = self.comboboxHistogram.get()
        window = Toplevel(self.plot_frame)
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, window)
        self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
        self.ax.hist(df[self.selection], bins=int(self.textBoxHistogram.get("1.0",END)))
        self.ax.set_xlabel(self.selection)
        self.ax.set_ylabel("Count")

    def kmenasOk(self):
        self.selection = self.comboboxKmeans.get()
        self.selection2 = self.comboboxKmeans2.get()
        window = Toplevel(self.plot_frame)
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, window)
        self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=1)
        data_left = df[df["left"] == 1]
        kmeans = KMeans(n_clusters=int(self.textBox.get("1.0",END)), random_state=2)
        kmeans.fit(data_left[[self.selection, self.selection2]])
        kmeans_colors = ['red' if c == 0 else 'orange' if c == 2 else 'blue' for c in kmeans.labels_]
        plt.scatter(x=self.selection, y=self.selection2, data=data_left,
                    alpha=0.25, color=kmeans_colors)
        plt.xlabel(self.selection)
        plt.ylabel(self.selection2)
        plt.scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], color="black", marker="X", s=100)
        plt.title("Clustering of the employed who left by Kmeans")

    def classification(self):
        classificationModule.classificationWindow(self.root)

    def filterData(self):
        window = Toplevel(self)
        self.filter_frame = ttk.LabelFrame(window, text='Filter Data',
                                           height=100)
        self.filter_frame.grid(column=0, row=0, columnspan=4, sticky='nesw')

        self.filter_frame.grid_configure(padx=5, pady=5)

        self.comboboxFilter = ttk.Combobox(self.filter_frame)
        self.comboboxFilter.bind("<<ComboboxSelected>>", self.comboUpdateFilter)
        self.comboboxFilter.grid(column=0, row=0)
        self.comboboxFilter['values'] = columns_names
        self.comboboxFilter.set('satisfaction_level')

        self.comboboxFilter2 = ttk.Combobox(self.filter_frame)
        self.comboboxFilter2.grid(column=0, row=1)
        self.comboboxFilter2['values'] = ['<', '>', '==']
        self.comboboxFilter2.set(self.comboboxFilter2['values'][1])

        self.comboboxFilter3 = ttk.Combobox(self.filter_frame)
        self.comboboxFilter3.grid(column=0, row=2)
        self.comboboxFilter3['values'] = df['satisfaction_level'].unique().tolist()
        self.comboboxFilter3.set('0.5')

        self.filterList = set()

        self.add_button = ttk.Button(self.filter_frame, text='Add Filter',
                                     command=lambda: self.addFilter())
        self.add_button.grid(column=1, row=1, columnspan=2, sticky='nesw')

        self.add_button = ttk.Button(self.filter_frame, text='Remove Filter',
                                     command=lambda: self.removeFilter())
        self.add_button.grid(column=1, row=2, columnspan=2, sticky='nesw')

        self.data = Listbox(self.filter_frame, width=22, height=10)
        self.data.grid(row=0, column=5, rowspan=3)

        self.add_button = ttk.Button(self.filter_frame, text='Apply Filter',
                                     command=lambda: self.applyFilter())
        self.add_button.grid(column=0, row=3, columnspan=7, sticky='nesw')

        ttk.Label(self.filter_frame, text='Head of Data').grid(column=0, row=4, sticky=W)
        self.filterResult = ScrolledText(self.filter_frame, height=15, width=70)
        self.filterResult.grid(column=0, row=4, sticky=W, columnspan=10, rowspan=5)
        self.filterResult.config(state=DISABLED)

        for child in window.winfo_children():
            for child2 in child.winfo_children():
                child2.grid_configure(padx=5, pady=5)

    def comboUpdateFilter(self, event=None):
        value = event.widget.get()
        self.comboboxFilter3.configure(values=df[value].unique().tolist())
        array = df[value].unique().tolist()
        self.comboboxFilter3.set(array[0])

    def addFilter(self):
        self.selection = self.comboboxFilter.get()
        self.selection2 = self.comboboxFilter2.get()
        self.selection3 = self.comboboxFilter3.get()
        self.data.insert(END, str(self.selection + ":" + self.selection2 + ":" + self.selection3))
        self.filterList.add(self.selection + ":" + self.selection2 + ":" + self.selection3)

    def removeFilter(self):
        self.selectionIndex = self.data.curselection()
        self.filterList.remove(self.data.get(self.selectionIndex[0]))
        self.data.delete(self.selectionIndex)

    def applyFilter(self):
        global df
        df = constantdf
        if self.filterList:
            for filter in self.filterList:
                tempList = filter.split(":")
                if tempList[0] != "salary" and tempList[0] != "sales":
                    if tempList[1] == "<":
                        df = df[df[tempList[0]] > float(tempList[2])]
                    if tempList[1] == ">":
                        df = df[df[tempList[0]] < float(tempList[2])]
                    if tempList[1] == "==":
                        df = df[df[tempList[0]] != float(tempList[2])]
                else:
                    if tempList[1] == "==":
                        df = df[df[tempList[0]] != tempList[2]]

        self.filterResult.config(state=NORMAL)
        self.filterResult.delete(1.0, END)
        self.filterResult.insert(END, df)
        self.filterResult.config(state=DISABLED)

    def showHead(self):
        tempDf = df.head()
        self.filterResult.config(state=NORMAL)
        self.filterResult.delete(1.0, END)
        self.filterResult.insert(END, tempDf)
        self.filterResult.config(state=DISABLED)

    def runOrange(self):
        import os
        os.system("orange-canvas")

    def crossRun(self):

        window = Toplevel(self)
        ttk.Label(window, text='Result').grid(column=0, row=0, sticky=W)
        self.crossResult = ScrolledText(window, height=15, width=70)
        self.crossResult.grid(column=0, row=1, sticky=W, columnspan=10, rowspan=5)
        self.crossResult.config(state=DISABLED)
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier as knn
        from sklearn.naive_bayes import GaussianNB as GB
        from sklearn.svm import SVC
        from sklearn.model_selection import cross_validate
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        df_drop = df.drop(labels=['sales', 'salary'], axis=1)
        left_col = df_drop['left']
        df_drop.drop(labels=['left'], axis=1, inplace=True)
        df_drop.insert(0, 'left', left_col)
        df_drop.head()
        X = df_drop.iloc[:, 1:8].values
        y = df_drop.iloc[:, 0].values
        X_std = StandardScaler().fit_transform(X)
        sklearn_pca = PCA(n_components=6)
        X_pca = sklearn_pca.fit_transform(X_std)
        models = ["RandomForestClassifier", "Gaussian Naive Bays", "KNN", "Logistic_Regression", "Support_Vector"]
        Classification_models = [RandomForestClassifier(n_estimators=100), GB(), knn(n_neighbors=7),
                                 LogisticRegression(), SVC()]
        Model_Accuracy = []
        scoring = {'acc': 'accuracy',
                   'f1': 'f1',
                   'precision': 'precision',
                   'recall': 'recall',
                   'roc_auc': 'roc_auc'
                   }
        for model, model_name in zip(Classification_models, models):
            print(model_name)
            scores = cross_validate(model, X_pca, y, scoring=scoring, cv=10, return_train_score=True)
            Model_Accuracy.append(scores)
        self.crossResult.config(state=NORMAL)
        self.crossResult.delete(1.0, END)
        for i, m in zip(Model_Accuracy, models):
            self.crossResult.insert(END, "\n" + m)
            self.crossResult.insert(END, "\n--------\n")
            for j in i:
                self.crossResult.insert(END, str(j) + ": " + str(i[j].mean()) + "\n")
        self.crossResult.config(state=DISABLED)

    def factorAnalysis(self):
        self.componentSize = int(self.textBoxComponent.get("1.0", END))
        self.testSize = float(self.textBoxTestSize.get("1.0", END))
        from sklearn.decomposition import FactorAnalysis
        from sklearn.metrics import accuracy_score
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        window = Toplevel(self)
        ttk.Label(window, text='Result').grid(column=0, row=0, sticky=W)
        self.factorResult = ScrolledText(window, height=15, width=70)
        self.factorResult.grid(column=0, row=1, sticky=W, columnspan=10, rowspan=5)
        self.factorResult.config(state=DISABLED)
        df1 = df
        y = df1['left'].values
        df1 = df1.drop(['left', 'sales', 'salary'], axis=1)
        X = df1.values
        factor = FactorAnalysis(n_components=self.componentSize).fit(X)
        X_ = factor.fit_transform(X)
        Xtrain, Xtest, ytrain, ytest = train_test_split(X_, y, test_size=self.testSize)
        log_reg = LogisticRegression()
        log_reg.fit(Xtrain, ytrain)
        y_val_l = log_reg.predict(Xtest)
        accuracy_score(ytest, y_val_l)
        m = factor.components_
        n = factor.noise_variance_
        m1 = m ** 2
        m2 = np.sum(m1, axis=1)
        self.factorResult.config(state=NORMAL)
        self.factorResult.delete(1.0, END)

        for i in range(0,self.componentSize):
            pvar = (100 * m2[i]) / np.sum(m2)
            self.factorResult.insert(END, "pvar" + str(i) + ": " + str(pvar))
            self.factorResult.insert(END, "\n")

if __name__ == '__main__':
    root = Tk()
    App(root)
    root.mainloop()