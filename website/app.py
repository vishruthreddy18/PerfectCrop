from flask import Flask, render_template, request
import pickle
import numpy as np
import folium
import random

app = Flask(__name__)

@app.route('/')
def home_page():
    random_image_selected="img"+str(random.randint(1,7))+".jpg"
    return render_template('home.html',random_image=random_image_selected)

@app.route('/ourgoal')
def our_goal():
    return render_template('our_goal.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

# @app.route('/districts')
# def district_page():
#     return render_template('districts.html')

@app.route('/districts')
def district_page():
    places=["Anantapur","Chittoor","East_Godavari","Guntur","Kadapa","Krishna","Kurnool","Nellore","Prakasam","Srikakulam","Visakhapatnam","Vizianagaram","West_Godavari"]
    return render_template('test.html',places=places)

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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    Zn = Zn.strip('-')
    Zn = float(Zn)
    Fe = Fe.strip('-')
    Fe = float(Fe)
    Mn = Mn.strip('-')
    Mn = float(Mn)

    list1 = np.array([pH, EC, OC, Zn, Fe, Mn, crop_dict['Black Soil'], crop_dict['Mixed Soil'], crop_dict['Red Soil'], crop_dict['Sandy Soil']])
    list1 = np.array([list1])
    list1[:,0:6] = scaler.transform(list1[:,0:6])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Anantapur"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    K = K.strip('-')
    K = float(K)
    Zn = Zn.strip('-')
    Zn = float(Zn)
    Fe = Fe.strip('-')
    Fe = float(Fe)
    Mn = Mn.strip('-')
    Mn = float(Mn)

    list1 = np.array([pH, EC, OC, K, Zn, Fe, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Chittoor"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    P = P.strip('-')
    P = float(P)
    K = K.strip('-')
    K = float(K)
    Mn = Mn.strip('-')
    Mn = float(Mn)

    list1 = np.array([pH, EC, P, K, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "East Godavari"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    K = K.strip('-')
    K = float(K)
    Ca = Ca.strip('-')
    Ca = float(Ca)
    Mg = Mg.strip('-')
    Mg = float(Mg)
    Zn = Zn.strip('-')
    Zn = float(Zn)

    list1 = np.array([pH, EC, OC, K, Ca, Mg, Zn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Guntur"
    return render_template('output.html', data=res)

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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    P = P.strip('-')
    P = float(P)
    K = K.strip('-')
    K = float(K)
    Mn = Mn.strip('-')
    Mn = float(Mn)

    list1 = np.array([pH, EC, OC, P, K, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Kadapa"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    K = K.strip('-')
    K = float(K)
    Ca = Ca.strip('-')
    Ca = float(Ca)
    Zn = Zn.strip('-')
    Zn = float(Zn)
    
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
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Krishna"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    P = P.strip('-')
    P = float(P)
    Zn = Zn.strip('-')
    Zn = float(Zn)
    Fe = Fe.strip('-')
    Fe = float(Fe)
    Cu = Cu.strip('-')
    Cu = float(Cu)

    list1 = np.array([pH, EC, OC, P, Zn, Fe, Cu])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Kurnool"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    P = P.strip('-')
    P = float(P)
    K = K.strip('-')
    K = float(K)
    Ca = Ca.strip('-')
    Ca = float(Ca)
    Mg = Mg.strip('-')
    Mg = float(Mg)
    Zn = Zn.strip('-')
    Zn = float(Zn)

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
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Nellore"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    P = P.strip('-')
    P = float(P)
    K = K.strip('-')
    K = float(K)
    Zn = Zn.strip('-')
    Zn = float(Zn)
    Fe = Fe.strip('-')
    Fe = float(Fe)
    Mn = Mn.strip('-')
    Mn = float(Mn)
    

    list1 = np.array([pH, EC, P, K, Zn, Fe, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Prakasam"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    P = P.strip('-')
    P = float(P)
    Fe = Fe.strip('-')
    Fe = float(Fe)
    Mn = Mn.strip('-')
    Mn = float(Mn)

    

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
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Srikakulam"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    P = P.strip('-')
    P = float(P)
    Fe = Fe.strip('-')
    Fe = float(Fe)

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
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Visakhapatnam"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    OC = OC.strip('-')
    OC = float(OC)
    P = P.strip('-')
    P = float(P)
    K = K.strip('-')
    K = float(K)
    Fe = Fe.strip('-')
    Fe = float(Fe)
    Mn = Mn.strip('-')
    Mn = float(Mn)

    list1 = np.array([pH, EC, OC, P, K, Fe, Mn])
    list1 = np.array([list1])
    list1[:,:] = scaler.transform(list1[:,:])
    pred = model.predict(list1)
    data = pred[0]
    probab_list = model.predict_proba(list1)[0]

    ans = dict()
    for i in range(len(model.classes_)):
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "Vizianagaram"
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

    pH = pH.strip('-')
    pH = float(pH)
    EC = EC.strip('-')
    EC = float(EC)
    Mn = Mn.strip('-')
    Mn = float(Mn)

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
        model.classes_[i] = model.classes_[i].replace("+", "or")
        ans[model.classes_[i]] = "{:.2f}".format((probab_list[i] * 100))
    
    ans_list = [(float(v), k) for k, v in ans.items()]
    ans_list.sort(reverse=True)
    res = dict()
    res['final_ans'] = ans_list
    res['district_name'] = "West Godavari"
    return render_template('output.html', data=res)

@app.route('/map')
def map_display():
    m = folium.Map(location=[16.196333585459023, 80.84786303681183], zoom_start=7)
    folium.Marker([14.683886299171148, 77.60021860852216], popup='<a target="_blank" href=/Anantapur>Predict!</a>', tooltip= "Anantapur").add_to(m)
    folium.Marker([13.21786796676931, 79.09944733592019],  popup='<a target="_blank" href=/Chittoor>Predict!</a>', tooltip= "Chittoor").add_to(m)
    folium.Marker([17.370341225421864, 82.01079117056173], popup='<a target="_blank" href=/East_Godavari>Predict!</a>', tooltip= "East Godavari").add_to(m)
    folium.Marker([16.31796271270806, 80.43880920394956],  popup='<a target="_blank" href=/Guntur>Predict!</a>', tooltip= "Guntur").add_to(m)
    folium.Marker([14.469685257965102, 78.82439366455279], popup='<a target="_blank" href=/Kadapa>Predict!</a>', tooltip= "Kadapa").add_to(m)
    folium.Marker([16.572103857283086, 80.85059936784116], popup='<a target="_blank" href=/Krishna>Predict!</a>', tooltip= "Krishna").add_to(m)
    folium.Marker([15.829016820646583, 78.03499797070828], popup='<a target="_blank" href=/Kurnool>Predict!</a>', tooltip= "Kurnool").add_to(m)
    folium.Marker([14.434108393791231, 79.98104886833782], popup='<a target="_blank" href=/Nellore>Predict!</a>', tooltip= "Nellore").add_to(m)
    folium.Marker([15.56634776682381, 79.55101034086451],  popup='<a target="_blank" href=/Prakasam>Predict!</a>', tooltip= "Prakasam").add_to(m)
    folium.Marker([16.196333585459023, 80.84786303681183], popup='<a target="_blank" href=/Srikakulam>Predict!</a>', tooltip= "Srikakulam").add_to(m)
    folium.Marker([17.695435246008262, 83.21449697148302], popup='<a target="_blank" href=/Visakhapatnam>Predict!</a>', tooltip= "Visakhapatnam").add_to(m)
    folium.Marker([18.106841627670757, 83.39686820653758], popup='<a target="_blank" href=/Vizianagaram>Predict!</a>', tooltip= "Vizianagaram").add_to(m)
    folium.Marker([16.871745407600862, 81.41923950492824], popup='<a target="_blank" href=/West_Godavari>Predict!</a>', tooltip= "West Godavari").add_to(m)
    m
    # start_coords = (46.9540700, 142.7360300)
    # folium_map = folium.Map(location=start_coords, zoom_start=14)
    return m._repr_html_()
