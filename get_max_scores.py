import numpy as np
import pandas as pd
import sys
import os



def getMaxScoreMethod(label, arr):
    max_score = 0
    max_score_method = None
    for experiment in arr:
        if (experiment[label] >= max_score):
            max_score = experiment[label]
            max_score_method = experiment
    return max_score_method

def removeRandomStateParam(model):
    if (model != None):
        d=model.get_params()
        if ('random_state' in d):
            d['random_state'] = None
        model.set_params(**d)
    return model


def renderMaxScoreMethod(label, arr):
    d = getMaxScoreMethod(label, arr)
    html = "<h3>Method for: maximum of " + label + "</h3>"
    html += "<p><ul>"
    for key, value in d.items():
        if (key == 'oversampler' or key == 'classifier'):
            value = removeRandomStateParam(value)
        if (key == 'cm'): 
            html += "<li>" + str(key) + ":<br>"  + showCM(value) +"</li>"
        else:
            html += "<li>" + key + " " + str(value) + "</li>"
    html+= "</ul></p>"
    return html

def showCM(cm):
    cm /= cm.sum()
    cm = cm*100
    cm = cm.round(2)
    html = "<table style=''>\
    <tr>\
        <th></th>\
        <th style='font-weight: bold;'>Negative predicted</th>\
        <th style='font-weight: bold;'>True predicted</th>\
    </tr>\
    <tr>\
        <td style='font-weight: bold;'>Negative known</td>\
        <td>TN: {}%</td>\
        <td>FP: {}%</td>\
    </tr>\
    <tr>\
        <td style='font-weight: bold;'>True known</td>\
        <td>FN: {}%</td>\
        <td>TP: {}%</td>\
    </tr>\
    </table>".format(cm[0,0], cm[1,0], cm[0,1], cm[1,1])
    return html


def renderHTML(name,arr,dataset):
    if (len(arr) == 0): 
        return "Cannot render"
    prettyName =  os.path.basename(os.path.normpath(name))
    html = "<html><head><meta name='viewport' content='width'>\
            <link rel='stylesheet' href='https://unpkg.com/marx-css/css/marx.min.css'><title>"
    html += prettyName
    html += " dataset</title></head><body><h1>"
    html += prettyName
    html += " dataset</h1>"


    unique, counts = np.unique(dataset.iloc[:, -1], return_counts=True)
    amount = dict(zip(unique, counts))
    total = sum(counts)
    html += "<p>Total: " + str(total)
    for key, value in amount.items():
        html += "<br>" + str(key) + ": " + str(value) + " , " + str(round(value/total, 4))
    html += "</p>"


    for key, value in arr[0].items():
        if (key != "classifier" and key != "oversampler" and key != "cm"):
            html += renderMaxScoreMethod(key, arr)
    html += "</body></html>"
    return html


def writeHTML(name, arr, dataset):
    f = open("reports/" + name + ".html", "w")
    f.write(renderHTML(name, arr, dataset))
    f.close()



if __name__ == "__main__":
    default_path = ''
    if (len(sys.argv) != 1):
        default_path = sys.argv[1]

    if (default_path == ''):
        exit('No dataset specified.')
    dataset = pd.read_csv('datasets/' + default_path + '.csv')
    array = np.load('results/' + default_path + '.npy', allow_pickle=True)
    writeHTML(default_path, array, dataset)