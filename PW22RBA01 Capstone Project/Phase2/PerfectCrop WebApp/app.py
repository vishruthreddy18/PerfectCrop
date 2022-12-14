from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')


@app.route('/about')
def about_page():
    return render_template('about.html')


@app.route('/districts')
def district_page():
    return render_template('districts.html')


@app.route('/Anantapur')
def anantapur_page():
    return render_template('anantapur.html')


@app.route('/Anantapur_predict', methods=['POST'])
def predict_Anantapur():
    
    model = pickle.load(open('Anantapur_model.pkl', 'rb'))
    scaler = pickle.load(open('Anantapur_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    Zn = request.form['Zn']
    Fe = request.form['Fe']
    Mn = request.form['Mn']
    SoilType = request.form['SoilType']
    crop_dict = dict()
    crop_dict['Mixed Soil'] = 0
    crop_dict['Red Soil'] = 0
    crop_dict['Sandy Soil'] = 0
    crop_dict['Black Soil'] = 0
    crop_dict[SoilType] += 1

    list1 = np.array([pH, EC, OC, Zn, Fe, Mn, crop_dict['Black Soil'], crop_dict['Mixed Soil'], crop_dict['Red Soil'], crop_dict['Sandy Soil']])
    list1 = np.array([list1])
    list1[:,0:6] = scaler.transform(list1[:,0:6])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)


@app.route('/Chittoor')
def chittoor_page():
    return render_template('chittoor.html')


@app.route('/Chittoor_predict', methods=['POST'])
def predict_Chittoor():
    model = pickle.load(open('Chittoor_model.pkl', 'rb'))
    scaler = pickle.load(open('Chittoor_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    K = request.form['K']
    Zn = request.form['Zn']
    Fe = request.form['Fe']
    Mn = request.form['Mn']

    list1 = np.array([pH, EC, OC, K, Zn, Fe, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)


@app.route('/East_Godavari')
def East_Godavari_page():
    return render_template('East_Godavari.html')


@app.route('/East_Godavari_predict', methods=['POST'])
def predict_East_Godavari():
    model = pickle.load(open('EG_model.pkl', 'rb'))
    scaler = pickle.load(open('EG_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    P = request.form['P']
    K = request.form['K']
    Mn = request.form['Mn']

    list1 = np.array([pH, EC, P, K, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)



@app.route('/Guntur')
def Guntur_page():
    return render_template('Guntur.html')


@app.route('/Guntur_predict', methods=['POST'])
def predict_Guntur():
    model = pickle.load(open('Guntur_model.pkl', 'rb'))
    scaler = pickle.load(open('Guntur_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    K = request.form['K']
    Ca = request.form['Ca']
    Mg = request.form['Mg']
    Zn = request.form['Zn']

    list1 = np.array([pH, EC, OC, K, Ca, Mg, Zn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)


if __name__ == "__main__":
    app.run(debug=True)


@app.route('/Kadapa')
def Kadapa_page():
    return render_template('Kadapa.html')


@app.route('/Kadapa_predict', methods=['POST'])
def predict_Kadapa():
    model = pickle.load(open('Kadapa_model.pkl', 'rb'))
    scaler = pickle.load(open('Kadapa_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    P = request.form['P']
    K = request.form['K']
    Mn = request.form['Mn']

    list1 = np.array([pH, EC, OC, P, K, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)


@app.route('/Krishna')
def Krishna_page():
    return render_template('Krishna.html')


@app.route('/Krishna_predict', methods=['POST'])
def predict_Krishna():
    model = pickle.load(open('Krishna_model.pkl', 'rb'))
    scaler = pickle.load(open('Krishna_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    K = request.form['K']
    Ca = request.form['Ca']
    Zn = request.form['Zn']
    
    SoilType = request.form['SoilType']
    crop_dict = dict()
    crop_dict['Black Soil'] = 0
    crop_dict['Red Soil'] = 0
    crop_dict[SoilType] += 1

    list1 = np.array([pH, EC, OC, K, Ca, Zn, crop_dict['Black Soil'], crop_dict['Red Soil']])
    list1 = np.array([list1])
    list1[:,0:6] = scaler.transform(list1[:,0:6])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)

@app.route('/Kurnool')
def Kurnool_page():
    return render_template('Kurnool.html')


@app.route('/Kurnool_predict', methods=['POST'])
def predict_Kurnool():
    model = pickle.load(open('Kurnool_model.pkl', 'rb'))
    scaler = pickle.load(open('Kurnool_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    P = request.form['P']
    Zn = request.form['Zn']
    Fe = request.form['Fe']
    Cu = request.form['Cu']

    list1 = np.array([pH, EC, OC, P, Zn, Fe, Cu])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)


@app.route('/Nellore')
def Nellore_page():
    return render_template('Nellore.html')


@app.route('/Nellore_predict', methods=['POST'])
def predict_Nellore():
    model = pickle.load(open('Nellore_model.pkl', 'rb'))
    scaler = pickle.load(open('Nellore_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    P = request.form['P']
    K = request.form['K']
    Ca = request.form['Ca']
    Mg = request.form['Mg']
    Zn = request.form['Zn']

    SoilType = request.form['SoilType']
    crop_dict = dict()
    crop_dict['Black Soil'] = 0
    crop_dict['Red Soil'] = 0
    crop_dict[SoilType] += 1

    list1 = np.array([pH, EC, OC, P, K, Ca, Mg, Zn, crop_dict['Black Soil'], crop_dict['Red Soil']])
    list1 = np.array([list1])
    list1[:,0:8] = scaler.transform(list1[:,0:8])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)



@app.route('/Prakasam')
def Prakasam_page():
    return render_template('Prakasam.html')


@app.route('/Prakasam_predict', methods=['POST'])
def predict_Prakasam():
    model = pickle.load(open('Prakasam_model.pkl', 'rb'))
    scaler = pickle.load(open('Prakasam_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    P = request.form['P']
    K = request.form['K']
    Zn = request.form['Zn']
    Fe = request.form['Fe']
    Mn = request.form['Mn']

    list1 = np.array([pH, EC, P, K, Zn, Fe, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)


@app.route('/Srikakulam')
def Srikakulam_page():
    return render_template('Srikakulam.html')


@app.route('/Srikakulam_predict', methods=['POST'])
def predict_Srikakulam():
    model = pickle.load(open('Srikakulam_model.pkl', 'rb'))
    scaler = pickle.load(open('Srikakulam_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    P = request.form['P']
    Fe = request.form['Fe']
    Mn = request.form['Mn']

    SoilType = request.form['SoilType']
    crop_dict = dict()
    crop_dict['Black Soil'] = 0
    crop_dict['Red Sandy Soil'] = 0
    crop_dict['Red Soil'] = 0
    crop_dict['Saline Soil'] = 0
    crop_dict['Sandy Loam Soil'] = 0
    crop_dict[SoilType] += 1

    list1 = np.array([pH, EC, OC, P, Fe, Mn, crop_dict['Black Soil'], crop_dict['Red Sandy Soil'], crop_dict['Red Soil'], crop_dict['Saline Soil'], crop_dict['Sandy Loam Soil']])
    list1 = np.array([list1])
    list1[:, 0:6] = scaler.transform(list1[:, 0:6])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)



@app.route('/Visakhapatnam')
def Visakhapatnam_page():
    return render_template('Visakhapatnam.html')


@app.route('/Visakhapatnam_predict', methods=['POST'])
def predict_Visakhapatnam():
    model = pickle.load(open('Visakhapatnam_model.pkl', 'rb'))
    scaler = pickle.load(open('Visakhapatnam_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    P = request.form['P']
    Fe = request.form['Fe']

    SoilType = request.form['SoilType']
    crop_dict = dict()
    crop_dict['Black Sandy Soil'] = 0
    crop_dict['Black Soil'] = 0
    crop_dict['Red Sandy Soil'] = 0
    crop_dict['Red Soil'] = 0
    crop_dict[SoilType] += 1

    list1 = np.array([pH, EC, OC, P, Fe, crop_dict['Black Sandy Soil'], crop_dict['Black Soil'], crop_dict['Red Sandy Soil'], crop_dict['Red Soil']])
    list1 = np.array([list1])
    list1[:, 0:5] = scaler.transform(list1[:, 0:5])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)


@app.route('/Vizianagaram')
def Vizianagaram_page():
    return render_template('Vizianagaram.html')


@app.route('/Vizianagaram_predict', methods=['POST'])
def predict_Vizianagaram():
    model = pickle.load(open('Vizianagaram_model.pkl', 'rb'))
    scaler = pickle.load(open('Vizianagaram_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    OC = request.form['OC']
    P = request.form['P']
    K = request.form['K']
    Fe = request.form['Fe']
    Mn = request.form['Mn']

    list1 = np.array([pH, EC, OC, P, K, Fe, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)


@app.route('/West_Godavari')
def West_Godavari_page():
    return render_template('West_Godavari.html')


@app.route('/West_Godavari_predict', methods=['POST'])
def predict_West_Godavari():
    model = pickle.load(open('WG_model.pkl', 'rb'))
    scaler = pickle.load(open('WG_scaler.pkl', 'rb'))
    pH = request.form['pH']
    EC = request.form['EC']
    Mn = request.form['Mn']

    SoilType = request.form['SoilType']
    crop_dict = dict()
    crop_dict['Black Clay Soil'] = 0
    crop_dict['Black Soil'] = 0
    crop_dict['Brown Clay Soil'] = 0
    crop_dict['Brown Soil'] = 0
    crop_dict['Clay Soil'] = 0
    crop_dict['Light Brown Soil'] = 0
    crop_dict['Red Sandy Loam Soil'] = 0
    crop_dict['Red Sandy Soil'] = 0
    crop_dict['Red Soil'] = 0
    crop_dict['Sandy Clay Soil'] = 0
    crop_dict['Sandy Loam Soil'] = 0
    crop_dict['Sandy Soil'] = 0
    crop_dict[SoilType] += 1

    list1 = np.array([pH, EC, Mn, crop_dict['Black Clay Soil'], crop_dict['Black Soil'], crop_dict['Brown Clay Soil'], crop_dict['Brown Soil'],crop_dict['Clay Soil'], crop_dict['Light Brown Soil'], crop_dict['Red Sandy Loam Soil'], crop_dict['Red Sandy Soil'], crop_dict['Red Soil'],crop_dict['Sandy Clay Soil'], crop_dict['Sandy Loam Soil'], crop_dict['Sandy Soil']])
    list1 = np.array([list1])
    list1[:, 0:3] = scaler.transform(list1[:, 0:3])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(k, v) for k, v in ans.items()]
    ans_list.sort(key = lambda x: x[1], reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    return render_template('output.html', data=res)


if __name__ == "__main__":
    app.run(debug=True)