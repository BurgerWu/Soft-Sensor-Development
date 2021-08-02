from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import KFold
from bokeh.models import FactorRange
from bokeh.transform import factor_cmap
from bokeh.palettes import Category20
from bokeh.plotting import figure, output_file, show
from bokeh.io import show
from bokeh.models import ColumnDataSource


class pls_evaluator:
    """
    pls_evaluator object can be used to evaluate data using PLSRegression model. This class contains
    several method to train, predict and perform cross validation on input dataset. The fit method can 
    return all the result tables, you can get the results by calling keys to the result dictionary.
    """
    def __init__(self, n_component, cv_fold):
        """
        n_component and cv_fold are essential inputs to pls_evaluator object.
        
        n_component: number of components to take in PLSRegression
        cv_fold: How many folds to split in cross validation
        """
        self.n_component = n_component
        self.cv_fold = cv_fold
        self.result_dict = {}
        self.num_variables = None
        self.x_data = None
        self.y_data = None
        self.original_model = None
        self.cv_model = None
        
    def fit(self, X, Y):
        """
        Fit method perform training, prediction and cross validation on input X and Y dataset
        
        X: Input independent variables
        Y: Input dependent variables
        """
        #Write properties to class  attributes
        self.num_variables = X.shape[1]
        self.x_data = X
        self.y_data = Y
        
        #Create PLSRegression object and train it with X and Y
        model = PLSRegression(n_components = self.n_component)
        model.fit(X, Y)
        
        #Make prediction using trained model
        original_pred = model.predict(X)
        
        #Store model instance in evaluator obejct
        self.original_model = model
        
        #Perform analyzing and evaluation and store results in result dictionary
        self.result_dict['x_score_table'] = self.x_score(model, X.index)
        self.result_dict['y_score_table'] = self.y_score(model, Y.index)
        self.result_dict['train_prediction_table'] = self.train_pred(Y['TARGET'], original_pred, X.index)
        self.result_dict['x_loading_table'] = self.x_loading(model, X.columns)
        self.result_dict['y_loading_table'] = self.y_loading(model, Y.columns)
        self.result_dict['x_weight_table'] = self.x_weight(model, X.columns)
        self.result_dict['y_weight_table'] = self.y_weight(model, Y.columns)
        self.result_dict['coef_table'] = self.coef(model, X.columns)
        self.result_dict['explained_variance_table'] = self.explained_variance(model, X, Y)
        
        #Perform cross validation and store results in result dictionary
        cv_result = self.cv(model)
        self.result_dict['cv_coef_table'] = cv_result[0]
        self.result_dict['cv_explained_variance_table'] = cv_result[1]
        self.cv_models = cv_result[2]
        
        return self.result_dict
    
    def cv(self, model):
        """
        This function performs cross validation using input trained model and return coefficient and explained variance result.
        
        model: Input trained model
        """
        
        #Create splits using KFold instance
        cv_splits = KFold(n_splits = self.cv_fold, shuffle = True)

        #Initiate table and lists
        cv_ev_table1 = pd.DataFrame(index = ["Factor_" + str(x) for x in range(1, self.n_component + 1)])
        train_explained_variance = np.zeros(self.n_component)
        validate_explained_variance = np.zeros(self.n_component)
        
        #Initiate cross validation index
        cv_index = 1
        
        #Create coefficient table for CV
        cv_coef_table = pd.DataFrame(index = self.x_data.columns)
        cv_coef_table.loc[:, "Original"] = model.coef_
        cv_models = []
        
        #Iterate through all splits
        for train_ind, test_ind in cv_splits.split(self.x_data):
            #Create submodel to predict submodel defined by KFold
            sub_trainX = self.x_data.iloc[train_ind, :]
            sub_trainY = self.y_data.iloc[train_ind, :]
            sub_testX = self.x_data.iloc[test_ind, :]
            sub_testY = self.y_data.iloc[test_ind, :]
            sub_model = PLSRegression(n_components = self.n_component)
            sub_model.fit(sub_trainX, sub_trainY)
            cv_coef_table.loc[:, 'CV_' + str(cv_index)] = sub_model.coef_
            
            cv_models.append(sub_model)
            
            #Get training and validation explained variance result using explained_variance method
            train_explained_variance += self.explained_variance(sub_model, sub_trainX, sub_trainY)['Explained_Variance']
            validate_explained_variance += self.explained_variance(sub_model, sub_testX, sub_testY)['Explained_Variance']
            
            #Update cross valiation index
            cv_index += 1
        
        #Average the list to get average explained variance
        train_explained_variance /= self.cv_fold
        validate_explained_variance /= self.cv_fold
        
        return cv_coef_table, pd.DataFrame(index = ['Factor_' + str(x) for x in range(1, self.n_component + 1)], 
                                           data = {"CV_Explained_Variance_Train": train_explained_variance,
                                                   "CV_Explained_Variance_Validate": validate_explained_variance} ), cv_models
               
        
    
    def x_score(self, model, index):
        """
        This function creates x score table from input model. 
        The dimension of returned table is (n_samples X n_components).
        
        model: Input trained model
        index: Index for samples
        """
        return pd.DataFrame(index = index,
                            columns = ["Factor_" + str(x) for x in range(1, model.n_components + 1)],
                            data = model.x_scores_)
    
    def y_score(self, model, index):
        """
        This function creates y score table from input model
        The dimension of returned table is (n_samples X n_components).
        
        model: Input trained model
        index: Index for samples
        """
        return pd.DataFrame(index = index,
                             columns = ["Factor_" + str(x) for x in range(1, model.n_components + 1)],
                             data = model.y_scores_)
    def train_pred(self, actual, prediction, index):
        """
        This function returns actual and prediction results table 
        
        actual: Actual value of samples
        prediction: Prediction value of samples
        index: Index for sampels
        """
        return pd.DataFrame(index = index,
                            data = {"Actual": actual, "Prediction": prediction.reshape([prediction.shape[0],])})
    def x_loading(self, model, columns):
        """
        This function creates x loadings table from input model. 
        The dimension of returned table is (n_variables X n_components).
        
        model: Input trained model
        columns: Index for variables
        """
        return pd.DataFrame(data = model.x_loadings_, 
                            columns = ['Factor_' + str(x) for x in range(1, self.n_component + 1)],
                            index = columns)
    def y_loading(self, model, columns):
        """
        This function creates y loadings table from input model. 
        The dimension of returned table is (n_variables X n_components).
        
        model: Input trained model
        columns: Index for variables
        """
        return pd.DataFrame(data = model.y_loadings_, 
                            columns = ['Factor_' + str(x) for x in range(1, self.n_component + 1)],
                            index = columns)
    def x_weight(self, model, columns):
        """
        This function creates x weights table from input model. 
        The dimension of returned table is (n_variables X n_components).
        
        model: Input trained model
        columns: Index for variables
        """
        return pd.DataFrame(data = model.x_weights_, 
                            columns = ['Factor_' + str(x) for x in range(1, self.n_component + 1)],
                            index = columns)
    def y_weight(self, model, columns):
        """
        This function creates y weights table from input model. 
        The dimension of returned table is (n_variables X n_components).
        
        model: Input trained model
        columns: Index for variables
        """
        return pd.DataFrame(data = model.y_weights_, 
                            columns = ['Factor_' + str(x) for x in range(1, self.n_component + 1)],
                            index = columns)
    def coef(self, model, columns):
        """
        This function returns coefficient table based on input trained model/
        The dimension of returned table is (n_variables X 1)
        """
        return pd.DataFrame(index = columns, data = {'Coefficient': model.coef_.reshape([model.coef_.shape[0],])})
    
    def explained_variance(self, model, X, Y):
        """
        This function creates explained variance table from input model. This function will predict targets 
        using different number of components and evaluate explained variance.
        The dimension of returned table is (n_components X 1).
        
        model: Input trained model
        X: Input independent variables
        Y: Input dependent variables
        """
        #Initiate list for storing results
        ev_list = []
        
        #Iterate through different number of components
        for i in range(1, model.n_components + 1):
            
            #Predict using different number of components and scaled X dataset
            coef = np.dot(model.x_rotations_[:,:i], np.transpose(model.y_loadings_[:,:i])) * model.y_std_
            scale_x = (X - model.x_mean_) / model.x_std_
            pred_x = np.dot(scale_x, coef) + float(model.y_mean_)
            
            #Append explained variance result to result list
            ev_list.append(explained_variance_score(Y, pred_x))
            
        return pd.DataFrame(index = ['Factor_' + str(x) for x in range(1, model.n_components + 1)],
                            data = {'Explained_Variance': ev_list})
    
    
class result_plotter:
    """
    The result plotter class can help you perform different kinds of plot on the basis of input dataset and factors if necessary.    
    """
    def __init__(self):
        """
        You do not have to input anything when building result plotter object
        """
        self.kind = None
        self.cv_enabled = False
    def plot(self, kind, dataset1, *dataset2, **factors):
        """
        Plot function can perform different kinds of plotting depending on the kind input. Below is the choice of plotting and the keys words to kind. 
        
        Available plot choices are
        - explained_variance_plot
        - tu_score_plot
        - predict_reference_plot
        - prediction_line_plot
        - loadings_plot
        - weights_plot
        - coef_plot
        - cv_explained_variance_plot
        - cv_coef_plot

        kind: Specify type of plotting desired
        dataset1: First input dataset, usually for x variables
        dataset2: Second input dataset, usually for y variables
        factors: Specify desired component(using keyword factor1 and factor2)
                 for score plot please specicy one factor (ex: factor1 = 1)
                 for weights and loadings plot please specify two factors (ex: factor1 = 1, factor2 = 2)
        """
        #Check if input is valid
        try:
            kind = str(kind).lower().strip()
        except:
            raise ValueError('Please re-check your "kind" parameter')
        
        #Define available plot choices
        plot_pool = ['explained_variance_plot',
                     'tu_score_plot',
                     'predict_reference_plot',
                     'prediction_line_plot',
                     'loadings_plot',
                     'weights_plot',
                     'coef_plot',
                     'cv_explained_variance_plot',
                     'cv_coef_plot']
        
        #Raise error if input kind is notv valid
        if kind not in plot_pool:
            raise ValueError("Input not available, please check docstring for available plots")
        else:
            #Call plot method according to input kind
            if kind == plot_pool[0]:
                self.cv_enabled = False
                self.explained_variance(dataset1)
            elif kind == plot_pool[1]:
                self.tu_score(dataset1, *dataset2, **factors)
            elif kind == plot_pool[2]:
                self.predict_reference(dataset1)
            elif kind == plot_pool[3]:
                self.predict_line(dataset1)
            elif kind == plot_pool[4]:
                self.loadings_plot(dataset1, *dataset2, **factors)
            elif kind == plot_pool[5]:
                self.weights_plot(dataset1, *dataset2, **factors)            
            elif kind == plot_pool[6]:
                self.coef_plot(dataset1)
            elif kind == plot_pool[7]:
                self.cv_enabled = True
                self.explained_variance(dataset1)        
            elif kind == plot_pool[8]:
                self.cv_enabled = True
                self.cv_coef_plot(dataset1)            
            
    def explained_variance(self, dataset):
        """
        This function plots explained variance plot according to different numbers of components
        
        dataset: Input dataset for plotting
        """
        #Create source for plotting
        source = ColumnDataSource(dataset)
        
        #Check if the explained_variance plot is for cross validation and define tooltips for different scenario
        if self.cv_enabled == False:
            tooltips = [("index", "@index"),("Explained_Variance", "@Explained_Variance")]
        else:
            tooltips = [("index", "@index"),("Explained_Variance_Train", "@CV_Explained_Variance_Train"), ("Explained_Variance_Validate", "@CV_Explained_Variance_Validate")]
        
        #Create figure object
        explained_variance_fig = figure(plot_width = 600,
                                        plot_height = 400, 
                                        tooltips = tooltips, 
                                        x_range = (list(dataset.index)), 
                                        #y_range = ([0, 1]),
                                        x_axis_label = "Number of Components",
                                        y_axis_label = "Explained Variance",
                                        title = "Explained Variance V.S Number of Components")

        #Add plot to figure object 
        if self.cv_enabled == False:
            explained_variance_fig.line("index", 'Explained_Variance', source = source, legend_label = 'Training Data', color = 'blue')
        else:
            explained_variance_fig.line("index", 'CV_Explained_Variance_Train', source = source, legend_label = 'Training Data', color = 'blue')
            explained_variance_fig.line("index", 'CV_Explained_Variance_Validate', source = source, legend_label = 'Validation Data', color = 'red')

        
        
        explained_variance_fig.legend.location = "top_left"


        #Show the results
        show(explained_variance_fig)        
        
    def tu_score(self, datasetx, *datasety, **factors):
        """
        This function shows the T-U score plot along specific factor
        
        datasetx: X score table
        datasety: Y score table
        factors: Specify which component to plot (recognize factor1 only)
        """
        #Check if factor input and source generation is valid
        try:
            factor1 = "Factor_" + str(factors['factor1'])    
            score_combine = pd.concat([datasetx[factor1], datasety[0][factor1]], axis = 1)
            score_combine.columns = [factor1 + '_X', factor1 + '_Y']
            source = ColumnDataSource(score_combine)
        except:
            raise ValueError("To use this plot, please properly define datasety, factor1")
        
        #Create figure object
        tu_score_fig = figure(plot_width = 600, 
                              plot_height = 400, 
                              x_range = ([-5, 7]), 
                              y_range = ([-10, 12]), 
                              tooltips = [("Index", "@index")],
                              x_axis_label = "T-Score({factor1})",
                              y_axis_label = "U-Score({factor1})")

        #Add plot to figure object 
        tu_score_fig.circle(score_combine.columns[0], score_combine.columns[1], source = source)
    
        #Show T-U Score Plot
        show(tu_score_fig)
        pearson_r = pearsonr(score_combine.iloc[:,1], score_combine.iloc[:,0])
        
        #Print correlation coefficient result
        print("The pearson correlation coefficient of TU at {} is {}".format(factor1, pearson_r[0]))
        
    def predict_reference(self, dataset):
        """
        This function plots the prediction versus reference plot and also print out RMSE and r2 score
        
        dataset: Input predict actual table
        """
        source = ColumnDataSource(dataset)
        
        #Create figure object
        pred_fig = figure(plot_width = 600,
                   plot_height = 400, 
                   tooltips = [("index", "@index"), ("Actual", "@Actual"), ("Prediction", "@Prediction")], 
                   x_range = ([230, 270]), 
                   y_range = ([230, 270]),
                   x_axis_label = "Actual Value",
                   y_axis_label = "Prediction Value",
                   title = "Prediction V.S Reference Plot")

        #Add plot to figure object 
        pred_fig.circle("Actual", 'Prediction', source = source, color = 'blue')
        pred_fig.line([230, 270],[230, 270], color = 'red', legend_label = 'Ideal Fitting Line')
        pred_fig.legend.location = "top_left"


        #Show the results
        show(pred_fig)
        
        #Print out r2 and RMSE for the prediction
        R2 = r2_score(dataset.iloc[:, 0], dataset.iloc[:, 1])
        RMSE = np.sqrt(mean_squared_error(dataset.iloc[:, 0], dataset.iloc[:, 1]))
        print("The R2 score of this prediction is {}".format(R2))
        print("The RMSE of this prediction is {}".format(RMSE))
    
    def predict_line(self, dataset):
        """
        This function plots the prediction line plot.
        
        dataset: Input predict actual table
        """
        #Run this cell to get prediction line plot of our first model
        dataset = dataset.sort_index()
        source = ColumnDataSource(dataset)

        #Create figure object
        pred_line_fig = figure(plot_width = 600,
                   plot_height = 400, 
                   tooltips = [("Actual", "@Actual"), ("Prediction", "@Prediction")], 
                   x_range = ([0, dataset.shape[0]+1]), 
                   y_range = ([200, 300]),
                   x_axis_label = "Index",
                   y_axis_label = "Value",
                   title = "Prediction VS Actual Line Plot")

        #Add plot to figure object 
        pred_line_fig.line("index", 'Prediction', source = source, legend_label = 'Prediction', color = 'blue')
        pred_line_fig.line("index", 'Actual', source = source, legend_label = 'Actual Value', color = 'red')
        pred_line_fig.legend.location = "top_left"


        #Show the results
        show(pred_line_fig)
        
    def loadings_plot(self, datasetx, *datasety, **factors):    
        """
        This function plots loadings plot according to two factors specified in factors input
        
        datasetx: Input X loadings table
        datasety: Input Y loadings table
        factors: Specify two factors to plot (ex: factor1 = 1, factor2 = 2)
        """
        #Check if factors input and source generation is valid
        try:
            factor1 = "Factor_" + str(factors['factor1'])
            factor2 = "Factor_" + str(factors['factor2'])
            source2 = ColumnDataSource(datasety[0])
            source1 = ColumnDataSource(datasetx)
        except:
            raise ValueError("To use this plot, please properly define datasety, factor1 and factor2")
            
        #Create figure object
        loading_fig = figure(plot_width = 600, 
                             plot_height = 400, 
                             x_range = ([-1, 1]), 
                             y_range = ([-1, 1]), 
                             tooltips = [("Index", "@index"), (factor1, "@" + factor1), (factor2, "@" + factor2)],
                             x_axis_label = "Loadings Factor" + str(factor1),
                             y_axis_label = "Loadings Factor" + str(factor2))

        #Add plot to figure object 
        loading_fig.circle(factor1, factor2, source = source1, color = 'blue', legend_label = 'PV')
        loading_fig.circle(factor1, factor2, source = source2, color = 'red', legend_label = 'TARGET')
        loading_fig.line([2, -2], [0, 0])
        loading_fig.line([0, 0], [2, -2])

        #Show Loadings Plot
        show(loading_fig)
        
    def weights_plot(self, datasetx, *datasety, **factors):
        """
        This function plots weights plot according to two factors specified in factors input
        
        datasetx: Input X weights table
        datasety: Input Y weights table
        factors: Specify two factors to plot (ex: factor1 = 1, factor2 = 2)
        """
        #Check if factors input and source generation is valid
        try:
            factor1 = "Factor_" + str(factors['factor1'])
            factor2 = "Factor_" + str(factors['factor2'])
            source2 = ColumnDataSource(datasety[0])
            source1 = ColumnDataSource(datasetx)
        except:
            raise ValueError("To use this plot, please properly define datasety, factor1 and factor2")
        
        #Create figure object
        weight_fig = figure(plot_width = 600, 
                             plot_height = 400, 
                             x_range = ([-1, 1]), 
                             y_range = ([-1, 1]), 
                             tooltips = [("Index", "@index"), (factor1, "@" + factor1), (factor2, "@" + factor2)],
                             x_axis_label = "Weights Factor" + str(factor1),
                             y_axis_label = "Weights Factor" + str(factor2))

        #Add plot to figure object 
        weight_fig.circle(factor1, factor2, source = source1, color = 'blue', legend_label = 'PV')
        weight_fig.circle(factor1, factor2, source = source2, color = 'red', legend_label = 'TARGET')
        weight_fig.line([2, -2], [0, 0])
        weight_fig.line([0, 0], [2, -2])

        #Show Weights Plot
        show(weight_fig)
        
    def coef_plot(self, dataset):
        """
        This function plots coefficient plot according to input dataset
        
        dataset: Input coefficient table
        """
        #Create plotting source
        source = ColumnDataSource(dataset)
        
        #Create figure object
        coef_fig = figure(plot_width = 600,
                           plot_height = 400, 
                           tooltips = [("index", "@index"), ("Coefficient", "@Coefficient")], 
                           x_range = list(dataset.index),
                           x_axis_label = "Variable",
                           y_axis_label = "Coefficient",
                           title = "Coefficient Plot")

        #Add plot to figure object 
        coef_fig.vbar(x = 'index', top = 'Coefficient', source = source, color = 'blue', line_color = 'black')

        #Show the results
        show(coef_fig)

    def cv_coef_plot(self, dataset):
        """
        This function plots coefficient plot in every cross validation split according to input dataset
        
        dataset: Input coefficient table with cross validation results
        """
        #Create source for plotting advanced bar plot in Bokeh
        plot_tuple = [ (var, split) for var in dataset.index for split in dataset.columns]
        coefs = dataset.values
        coefs = coefs.reshape(dataset.shape[0]*dataset.shape[1])
        source = ColumnDataSource(data = dict(x = plot_tuple, counts = coefs))

        #Create figure object
        cv_coef_fig = figure(plot_width = 1200,
                             plot_height = 400, 
                             x_range = FactorRange(*plot_tuple),
                             tooltips = [("PV and CV", "@x"),("Coefficient", "@counts")],
                             x_axis_label = "Variable",
                             y_axis_label = "Coefficient",
                             title = "Coefficient Plot (Cross Validation)")

        #Add plot to figure object 
        cv_coef_fig.vbar(x = 'x', top = 'counts', source = source, line_color = 'black',
                          fill_color = factor_cmap('x', palette = Category20[dataset.shape[1]], factors = list(dataset.columns), start = 1, end = 2))
        cv_coef_fig.xaxis.major_label_text_color = None


        
        #Show coefficient plot with cross validation
        show(cv_coef_fig)