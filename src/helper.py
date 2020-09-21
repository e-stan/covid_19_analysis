from sklearn.model_selection import KFold
from DecoID.DecoID import flatten
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import scipy.stats as stats
import sklearn
import itertools
from multiprocessing import Process,Queue,Pool
import sklearn.cross_decomposition
import sklearn.ensemble
from sklearn.decomposition import PCA
import pandas as pd
import combat


def runSplit(X_train,X_test,y_train,y_test,X_train_blank,mol_names,fit_func,pred_func,params):
    obj = fit_func(X_train,y_train,X_train_blank,mol_names,params)
    y_pred = list(pred_func(obj,X_test))
    return list(y_test),list(y_pred)

#do cross fold validation
def crossVal(X,y,X_train_blank,mol_names,fit_func,pred_func,params,k=10,numCores=20):
    loo = KFold(n_splits=k)
    loo.get_n_splits(X)
    #p = Pool(numCores)
    argList = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        args = [X_train,X_test,y_train,y_test,X_train_blank,mol_names,fit_func,pred_func,params]
        argList.append(args)
    #print('starting')
    #output = p.starmap(runSplit,argList)
    #p.terminate()
    #p.join()
    output = [runSplit(*x) for x in argList]
    y_true = flatten([x[0] for x in output])
    y_preds = flatten([x[1] for x in output])
    #print('finished')
    return y_preds,y_true

def score(y_pred,y_true):
    return sklearn.metrics.roc_auc_score(y_true,y_pred)
    #return sklearn.metrics.f1_score(y_true, y_pred)


#imput matrix with half feature minimum
def imputeRowMin(arr,alt_min=2):
    #find the minimum non-zero value of each compound
    print(arr.size)
    numImp = 0
    max_vals = []
    for c in arr.transpose():
        tmp = [x for x in c if x > 1e-3]
        if len(tmp) > 0:
            val = np.min(tmp)
        else:
            val = alt_min
        max_vals.append(val)
    #impute values

    data_imp = np.zeros((len(arr),len(arr[0])))

    for c in range(len(arr[0])):
        for r in range(len(arr)):
            if arr[r,c] > 1e-3:
                data_imp[r,c] = arr[r,c]
            else:
                data_imp[r,c] = max_vals[c]/2
                numImp += 1
            if data_imp[r,c] < 1e-3:
                data_imp[r,c] = alt_min
                numImp += 1
    print(numImp)
    return data_imp


#perform feature selection from training data


def featureSelection(X_train,y_train,X_train_blank,mol_names,params,plot=True,out=True):
    #filter mets in blank
    method = params[0]
    if method == "stat":
        alpha = params[1]
        fc_cutoff = params[2]

        #compute stats
        if len(X_train_blank) > 0:
            pvals = []
            for met in range(len(X_train[0])):
                stat,p = stats.ttest_ind(X_train[:,met],X_train_blank[:,met],equal_var=False)
                pvals.append(p)

            reject,pvals_corr,_,_ = multipletests(pvals,alpha,method="fdr_bh")
            sigMet_blank = [x for x in range(len(reject)) if reject[x] and
                            np.mean(X_train[:,x])/np.mean(X_train_blank[:,x]) > fc_cutoff]
        else:
            sigMet_blank = [x for x in range(len(X_train[0]))]
        if out: print(len(sigMet_blank), "compounds sig. higher in samples than blank")

        #filter mets in vent+/-

        #separate negative and positive examples
        ventNeg = np.array([x for x,y in zip(X_train,y_train) if y < .5])
        ventPos = np.array([x for x,y in zip(X_train,y_train) if y > .5])

        #compute stats
        pvals = []
        for met in range(len(X_train[0])):
            stat,p = stats.ttest_ind(ventNeg[:,met],ventPos[:,met],equal_var=False)
            pvals.append(p)

        reject,pvals_corr,_,_ = multipletests(pvals,alpha,method="fdr_bh")
        sigMet_vent = [x for x in range(len(reject)) if reject[x]]

        reject = []
        for met in sigMet_vent:
            if abs(np.mean(ventNeg[:,met]) - np.mean(ventPos[:,met])) < np.log2(fc_cutoff):
                reject.append(met)

        sigMet_vent = [x for x in sigMet_vent if x not in reject]

        if out: print(len(sigMet_vent), "compounds sig. diff between vent+ and vent-")

        selectedFeatures = list(set(sigMet_vent).intersection(set(sigMet_blank)))
        if out: print(len(selectedFeatures), "compounds sig. diff between vent+ and vent- and sig. higher than blank")


        #plot feature intensity for sig mets
        if plot and len(selectedFeatures) > 0:
            xpos = 0
            poses = []
            labels = []
            for met in selectedFeatures:
                metName = mol_names[met]
                plt.bar([xpos,xpos+1],[np.mean(ventNeg[:,met]),np.mean(ventPos[:,met])],yerr=[np.std(ventNeg[:,met]),np.std(ventPos[:,met])],color=["red","black"],capsize=3)
                poses.append(xpos+.5)
                labels.append(metName)
                xpos += 3

            plt.plot([-10,-10],[0,1],color="red",label="ICU-")
            plt.plot([-10,-10],[0,1],color="black",label="ICU+")
            plt.xticks(poses,labels=labels,rotation = 90)
            plt.xlim((min(poses)-2,max(poses)+2))
            plt.legend()
            plt.ylabel("normalized intensity")
    if method == "corr":
        num_feats = int(params[1]*len(X_train[0]))
        corrs = []
        for m in range(len(X_train[0])):
            corrs.append(np.abs(stats.pearsonr(X_train[:,m],y_train)[0]))
        order = list(range(len(X_train[0])))
        order.sort(key=lambda x:corrs[x],reverse=True)
        selectedFeatures = order[:num_feats]

    return selectedFeatures

def optimizeHyperParams(params_to_it,X_train,y_train,X_train_blank,mol_names,trainModel,predictModel,k=None,n=2):

    if k == None or k > len(X_train) or k < 1:
        k = len(X_train)

    if n > k:
        n = k
    #iterate over hyperparams
    results = {}

    #get nested list of parameter names value pairs
    paramDicts = []

    for p,vals in params_to_it.items():
        paramDicts.append([{p:v} for v in vals])

    #get all parameter combinations
    paramsToIterate = list(itertools.product(*paramDicts))
    paramNames = list(params_to_it.keys())

    pool = Pool(n)
    paramList = []
    argList = []


    #cross validate all parameters
    for p in paramsToIterate:
        params = {}
        for d in p:
            params.update(d)
        #cross validate
        #y_pred,y_true = crossVal(X_train,y_train,X_train_blank,mol_names,trainModel,predictModel,params,k=k,numCores=n)
        argList.append([X_train,y_train,X_train_blank,mol_names,trainModel,predictModel,params,k,n])
        paramList.append(tuple([params[x] for x in paramNames]))

        #score
        #err_cv = sklearn.metrics.f1_score(y_true,y_pred)

        #output error
        #results[(params[x] for x in paramNames)] = err_cv

    output = pool.starmap(crossVal,argList)
    pool.terminate()
    pool.join()
    results = {params:score(y_pred,y_true) for params,(y_pred,y_true) in zip(paramList,output)}
    maxScore = np.max(list(results.values()))
    k = list(results.keys())
    k.sort(key=lambda x: abs(results[x]-maxScore))

    bestParams = k[0]
    bestParams = {p:v for p,v in zip(paramNames,bestParams)}
    return results,maxScore,bestParams,trainModel(X_train,y_train,X_train_blank,mol_names,bestParams)

def trainLogisticRegression(X_train,y_train,X_train_blank,mol_names,params):
    #get feature selection parameters
    feat_params = params["feat_selection"]

    #get estimator parameters
    otherParams = {key:val for key,val in params.items() if key != "feat_selection"}

    #make object
    obj = sklearn.linear_model.LogisticRegression(solver="saga",penalty="elasticnet",fit_intercept=True,class_weight="balanced",**otherParams)

    #selected features
    selected_feats = featureSelection(X_train, y_train, X_train_blank,mol_names,feat_params,False, False)

    #train model
    if len(selected_feats) > 0:
        X_train_sig = X_train[:, selected_feats]
        obj.fit(X_train_sig, y_train)
    obj._selectedvars = selected_feats

    return obj

def predictLogisticRegression(obj,X_test):
    if len(obj._selectedvars) > 0:
        return obj.predict_proba(X_test[:, obj._selectedvars])[:,1]
    else:
        return np.array([.5 for _ in X_test])

def trainElasticNet(X_train, y_train, X_train_blank, mol_names, params):
    # get feature selection parameters
    feat_params = params["feat_selection"]


    # get estimator parameters
    otherParams = {key:val for key,val in params.items() if key != "feat_selection"}

    # make object
    obj = sklearn.linear_model.ElasticNet(fit_intercept=True,**otherParams)

    # selected features
    selected_feats = featureSelection(X_train, y_train, X_train_blank,mol_names,feat_params,False, False)


    sampleWeights = sklearn.utils.class_weight.compute_sample_weight("balanced", y_train)


    # train model
    if len(selected_feats) > 0:
        X_train_sig = X_train[:, selected_feats]
        obj.fit(X_train_sig, y_train, sample_weight=sampleWeights)
        #classi = sklearn.linear_model.LogisticRegression(fit_intercept=True, solver="saga", class_weight="balanced")
        #classi.fit([[x] for x in obj.predict(X_train_sig)], y_train)
        #obj._classi = classi
    obj._selectedvars = selected_feats

    return obj

def predictElasticNet(obj, X_test):
    if len(obj._selectedvars) > 0:
        #return obj._classi.predict([[x] for x in obj.predict(X_test[:, obj._selectedvars])])
        return obj.predict(X_test[:, obj._selectedvars])

    else:
        return np.array([.5 for _ in X_test])

def trainPLSDA(X_train,y_train,X_train_blank,mol_names,params):
    #get feature selection parameters

    feat_params = params["feat_selection"]

    #get estimator parameters
    otherParams = {key:val for key,val in params.items() if key != "feat_selection"}

    #make object
    obj = sklearn.cross_decomposition.PLSRegression(**otherParams)

    num_c = otherParams["n_components"]
    #selected features
    selected_feats = featureSelection(X_train, y_train, X_train_blank,mol_names,feat_params,False, False)

    #train model
    if len(selected_feats) > 0:
        if len(selected_feats) < num_c:
            obj.n_components = len(selected_feats)
        X_train_sig = X_train[:, selected_feats]
        obj.fit(X_train_sig, y_train)
    obj._selectedvars = selected_feats

    return obj

def predictPLSDA(obj,X_test):
    if len(obj._selectedvars) > 0:
        #return sklearn.preprocessing.binarize(obj.predict(X_test[:, obj._selectedvars]), .5)[:, 0]
        return obj.predict(X_test[:, obj._selectedvars])

    else:
        return np.array([.5 for _ in X_test])

def trainSVM(X_train, y_train, X_train_blank, mol_names, params):
    # get feature selection parameters
    feat_params = params["feat_selection"]


    # get estimator parameters
    otherParams = {key:val for key,val in params.items() if key != "feat_selection"}

    # make object
    obj = sklearn.svm.SVC(class_weight="balanced",**otherParams)


    # selected features
    selected_feats = featureSelection(X_train, y_train, X_train_blank,mol_names,feat_params,False, False)


    # train model
    if len(selected_feats) > 0:
        X_train_sig = X_train[:, selected_feats]
        obj.fit(X_train_sig, y_train)
    obj._selectedvars = selected_feats

    return obj

def predictSVM(obj, X_test):
    if len(obj._selectedvars) > 0:
        return obj.predict(X_test[:, obj._selectedvars])
    else:
        return np.array([.5 for _ in X_test])

def trainRF(X_train, y_train, X_train_blank, mol_names, params):
    # get feature selection parameters
    feat_params = params["feat_selection"]


    # get estimator parameters
    otherParams = {key:val for key,val in params.items() if key != "feat_selection"}

    # make object
    obj = sklearn.ensemble.RandomForestClassifier(**otherParams)


    # selected features
    selected_feats = featureSelection(X_train, y_train, X_train_blank,mol_names,feat_params,False, False)


    # train model
    if len(selected_feats) > 0:
        X_train_sig = X_train[:, selected_feats]
        obj.fit(X_train_sig, y_train)
    obj._selectedvars = selected_feats

    return obj

def predictRF(obj, X_test):
    if len(obj._selectedvars) > 0:
        return obj.predict(X_test[:, obj._selectedvars])
    else:
        return np.array([.5 for _ in X_test])

def normalize_data(samp_train_int,blank_train_int,qc_train_int,samp_test_int, blank_test_int, qc_test_int,samp_train_batch,
                   blank_train_batch, qc_train_batch, samp_test_batch, blank_test_batch, qc_test_batch,samp_train_vent,samp_test_vent,baseline="none",plot=False):
    # QC baseline

    # normalize and imput

    train_whole = np.concatenate((samp_train_int, blank_train_int, qc_train_int))
    train_whole_imput = imputeRowMin(train_whole, 4)

    # log2
    train_whole_imput = np.log2(train_whole_imput)

    # normalize and imput testing data

    test_whole = np.concatenate((samp_test_int, blank_test_int, qc_test_int))
    test_whole_imput = imputeRowMin(test_whole, 4)

    # log2
    test_whole_imput = np.log2(test_whole_imput)

    # normalize
    temp = np.concatenate((train_whole_imput, test_whole_imput))
    tempBatch = np.concatenate(
        (samp_train_batch, blank_train_batch, qc_train_batch, samp_test_batch, blank_test_batch, qc_test_batch))

    tempNorm = combat.combat(pd.DataFrame(temp.transpose()), tempBatch).to_numpy().transpose()
    train_whole_norm = tempNorm[:len(train_whole_imput)]

    test_whole_norm = tempNorm[len(train_whole_imput):]

    if baseline == "qc":
        # #baseline to qcSamps
        baseInt = [np.median(x) for x in train_whole_norm[len(samp_train_int) + len(blank_train_int):].transpose()]
        train_whole_norm = np.array([[v - med for v, med in zip(row, baseInt)] for row in train_whole_norm])

        # #baseline to qcSamps
        baseInt = [np.median(x) for x in test_whole_norm[len(samp_test_int) + len(blank_test_int):].transpose()]
        test_whole_norm = np.array([[v - med for v, med in zip(row, baseInt)] for row in test_whole_norm])
    if baseline == "all":
        baseInt = [np.median(x) for x in train_whole_norm.transpose()]
        train_whole_norm = np.array([[v - med for v, med in zip(row, baseInt)] for row in train_whole_norm])

        baseInt = [np.median(x) for x in test_whole_norm.transpose()]
        test_whole_norm = np.array([[v - med for v, med in zip(row, baseInt)] for row in test_whole_norm])

    # do pca
    pca = PCA(n_components=2)
    pca.fit(train_whole_norm)
    loads = pca.transform(train_whole_norm)

    # get vent pos/neg loadings
    negSamples = [x for x in range(len(samp_train_vent)) if samp_train_vent[x] == 0]
    posSamples = [x for x in range(len(samp_train_vent)) if samp_train_vent[x] == 1]
    samp_load_vent_neg = loads[:len(samp_train_int)][negSamples]
    samp_load_vent_pos = loads[:len(samp_train_int)][posSamples]

    # get blanks and qc
    blank_load = loads[len(samp_train_int):len(samp_train_int) + len(blank_train_int)]
    qc_load = loads[len(samp_train_int) + len(blank_train_int):]

    if plot:
        # make pca plot divided by sample type
        plt.scatter(samp_load_vent_neg[:, 0], samp_load_vent_neg[:, 1], label="vent- covid+ d0")
        plt.scatter(samp_load_vent_pos[:, 0], samp_load_vent_pos[:, 1], label="vent+ covid+ d0")
        plt.scatter(blank_load[:, 0], blank_load[:, 1], label="blanks")
        plt.scatter(qc_load[:, 0], qc_load[:, 1], label="qc")
        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        # make pca plot by batch
        plt.figure()
        batch_con = np.concatenate((samp_train_batch, blank_train_batch, qc_train_batch))
        plt.scatter(loads[:, 0], loads[:, 1], c=batch_con)
        plt.colorbar()
        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")

    # separate dataset for research samples
    X_train = train_whole_norm[:len(samp_train_int)]
    y_train = samp_train_vent

    X_train_blank = train_whole_norm[len(samp_train_int):len(samp_train_int) + len(blank_train_int)]

    # make pca of research samples only
    pca = PCA(n_components=2)
    pca.fit(X_train)
    loads = pca.transform(X_train)

    if plot:
        plt.figure()
        plt.scatter(loads[:, 0], loads[:, 1], c=y_train)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar()

        plt.figure()

    # do pca
    pca = PCA(n_components=2)
    pca.fit(test_whole_norm)
    loads = pca.transform(test_whole_norm)

    # get vent pos/neg loadings
    negSamples = [x for x in range(len(samp_test_vent)) if samp_test_vent[x] == 0]
    posSamples = [x for x in range(len(samp_test_vent)) if samp_test_vent[x] == 1]
    samp_load_vent_neg = loads[:len(samp_test_int)][negSamples]
    samp_load_vent_pos = loads[:len(samp_test_int)][posSamples]

    # get blanks and qc
    blank_load = loads[len(samp_test_int):len(samp_test_int) + len(blank_test_int)]
    qc_load = loads[len(samp_test_int) + len(blank_test_int):]

    if plot:
        # make pca plot divided by sample type
        plt.scatter(samp_load_vent_neg[:, 0], samp_load_vent_neg[:, 1], label="vent- covid+ d0")
        plt.scatter(samp_load_vent_pos[:, 0], samp_load_vent_pos[:, 1], label="vent+ covid+ d0")
        plt.scatter(blank_load[:, 0], blank_load[:, 1], label="blanks")
        plt.scatter(qc_load[:, 0], qc_load[:, 1], label="qc")
        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        # make pca plot by batch
        plt.figure()
        batch_con = np.concatenate((samp_test_batch, blank_test_batch, qc_test_batch))
        plt.scatter(loads[:, 0], loads[:, 1], c=batch_con)
        plt.colorbar()
        plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")

    # separate dataset for research samples
    X_test = test_whole_norm[:len(samp_test_int)]
    y_test = samp_test_vent

    if plot:
        # make pca of research samples only
        pca = PCA(n_components=2)
        pca.fit(X_test)
        loads = pca.transform(X_test)

        plt.figure()
        plt.scatter(loads[:, 0], loads[:, 1], c=y_test)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar()

    return X_train,y_train,X_test,y_test,X_train_blank