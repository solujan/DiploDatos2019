import sys, os
import pandas as pd
import numpy as np

rename_col_dict = {
    'area1': 'zona_urbana',
    'area2': 'zona_rural',
    'v2a1': 'monthly_rent',
    'lugar1': 'region_central',
    'lugar2': 'region_chorotega',
    'lugar3': 'region_pacifico_central',
    'lugar4': 'region_brunca',
    'lugar5': 'region_huetar_atlantica',
    'lugar6': 'region_huetar_norte'}


def get_filter_by_row(input, columns):
    filter_data = {}
    for col in columns:
        if "columns" in col:
            for df_column in col["columns"]:
                if input[df_column] == 1:
                    filter_data[df_column] = 1
        elif "name" in col:
            filter_data[col["name"]] = input[col["name"]]
    return filter_data


techo_columns = ['techozinc', 'techoentrepiso', 'techocane', 'techootro']

techo_variables_check = [
    {"columns": ["lugar1", "lugar2", "lugar3", "lugar4", "lugar5", "lugar6"]},
    {"columns": ["area1", "area2"]},
    {"columns": ["paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc",
                 "paredfibras", "paredother"]},
    {"columns": ["pisomoscer", "pisocemento", "pisoother", "pisonatur", "pisonotiene",
                 "pisomadera"]},
]


def techo_mode(input, house_parent, list_houses_techo_issue):
    if input.idhogar not in list_houses_techo_issue:
        return input
    filter_data = {}
    for n in range(len(techo_variables_check)):
        filter_data = None
        if n == 0:
            filter_data = get_filter_by_row(input, techo_variables_check)
        elif n > 0:
            filter_data = get_filter_by_row(input, techo_variables_check[:-n])
        filtered_ds = house_parent.loc[
            (house_parent[list(filter_data)] == pd.Series(filter_data)).all(axis=1)]
        if filtered_ds.shape[0] > 0:
            input[filtered_ds[techo_columns].sum().idxmax()] = 1
            return input


def fix_techo(ds):
    idhogar_list = pd.Series(ds['idhogar'].unique())
    techo_ds = ds[['idhogar', 'techozinc', 'techoentrepiso', 'techocane', 'techootro']]
    id = idhogar_list.apply(
        lambda x: x if techo_ds[techo_ds.idhogar == x].all().value_counts()[True] != 2 else None)
    print("Cantidad de familias sin caracteristicas comunes: ", len(id[id.notnull()]))
    list_houses_techo_issue = list(id[id.notnull()])
    house_parent = ds[(ds.parentesco1 == 1) & ~(ds.idhogar.isin(list_houses_techo_issue))]
    return ds.apply(lambda x: techo_mode(x, house_parent, list_houses_techo_issue), axis=1)


electricity_variables = [
    {"columns": ["lugar1", "lugar2", "lugar3", "lugar4", "lugar5", "lugar6"]},
    {"columns": ["area1", "area2"]},
    {"columns": ["energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4"]},
    {"name": "cielorazo"},
    {"columns": ["eviv1", "eviv2", "eviv3"]},
    {"columns": ["etecho1", "etecho2", "etecho3"]},
    {"columns": ["epared1", "epared2", "epared3"]},

]

electricity_columns = ["public", "planpri", "noelec", "coopele"]


def electricity_mode(input, house_parent, list_houses_issue):
    if input.idhogar not in list_houses_issue:
        return input
    for n in range(len(electricity_variables)):
        filter_data = {}
        if n == 0:
            filter_data = get_filter_by_row(input, electricity_variables)
        elif n > 0:
            filter_data = get_filter_by_row(input, electricity_variables[:-n])
        filtered_ds = house_parent.loc[
            (house_parent[list(filter_data)] == pd.Series(filter_data)).all(axis=1)]
        if filtered_ds.shape[0] > 0:
            input[filtered_ds[electricity_columns].sum().idxmax()] = 1
            return input


def fix_electricity(ds):
    idhogar_list = pd.Series(ds['idhogar'].unique())
    elect_ds = ds[['idhogar', "public", "planpri", "noelec", "coopele"]]
    id = idhogar_list.apply(
        lambda x: x if elect_ds[elect_ds.idhogar == x].all().value_counts()[True] != 2 else None)

    print("Cantidad de familias sin caracteristicas comunes: ", len(id[id.notnull()]))
    list_houses_issue = list(id[id.notnull()])
    house_parent = ds[(ds.parentesco1 == 1) & ~(ds.idhogar.isin(list_houses_issue))]
    return ds.apply(lambda x: electricity_mode(x, house_parent, list_houses_issue), axis=1)

def fix_v18q1(ds):
    ds.loc[ds.v18q1.isna(), 'v18q1'] = 0
    return ds

costo_oportunidad_check_columns = [
    {"columns": ["region_central", "region_chorotega", "region_pacifico_central", "region_brunca",
                 "region_huetar_atlantica", "region_huetar_norte"]},
    {"columns": ["zona_urbana", "zona_rural"]},
    {"name": "cielorazo"},
    {"columns": ["eviv1", "eviv2", "eviv3"]},
    {"name": "rooms"},
    {"columns": ["etecho1", "etecho2", "etecho3"]},
    {"columns": ["epared1", "epared2", "epared3"]},
    {"columns": ["paredblolad", "paredzocalo", "paredpreb", "pareddes", "paredmad", "paredzinc",
                 "paredfibras", "paredother"]},
    {"columns": ["pisomoscer", "pisocemento", "pisoother", "pisonatur", "pisonotiene",
                 "pisomadera"]},
    {"columns": ["techozinc", "techoentrepiso", "techocane", "techootro"]},
]


def get_costo_de_oportunidad(input, ds_paid_rent):
    if input.monthly_rent > 0:
        return input
    for n in range(len(costo_oportunidad_check_columns)):
        filter_data = {}
        if n == 0:
            filter_data = get_filter_by_row(input, costo_oportunidad_check_columns)
        elif n > 0:
            filter_data = get_filter_by_row(input, costo_oportunidad_check_columns[:-n])
        filtered_ds = ds_paid_rent.loc[
            (ds_paid_rent[list(filter_data)] == pd.Series(filter_data)).all(axis=1)]
        if filtered_ds.shape[0] > 0:
            input["monthly_rent"] = filtered_ds.monthly_rent.mean()
            return input

def get_educacion_jefe(x, _ds):
    x['edu_jefe'] = (_ds.loc[(_ds['parentesco1']==1) & (_ds['idhogar']==x['idhogar']), 'escolari'].item())**2
    return x
        
def add_synthetic_features(_ds):
    _ds['tech_individuo'] = (_ds['mobilephone'] + _ds['v18q'])**2
    _ds['tech_hogar'] = (_ds['television'] + _ds['qmobilephone'] +
                      _ds['computer'] + _ds['v18q1'])**2
    _ds['monthly_rent_log'] = np.log(_ds['monthly_rent'])
    _ds = _ds.apply(lambda x: get_educacion_jefe(x, _ds), axis=1)
    _ds['bedrooms_to_rooms'] = _ds['bedrooms']/_ds['rooms']
    _ds['rent_to_rooms'] = _ds['monthly_rent']/_ds['rooms']
    _ds['SQBage'] = _ds['age'] ** 2
    _ds['SQBhogar_total'] = (_ds['hogar_nin'] + _ds['hogar_mayor'] +_ds['hogar_adul']) ** 2
    _ds['child_dependency'] = _ds['hogar_nin'] / (_ds['hogar_nin'] + _ds['hogar_mayor']+_ds['hogar_adul']) 
    _ds['rooms_per_person'] = (_ds['hogar_nin'] + _ds['hogar_mayor'] +_ds['hogar_adul'])  / (_ds['rooms'])
    _ds['female_weight'] = ((_ds['r4m1'] + _ds['r4m2'])/_ds['tamhog'])**2
    _ds['male_weight'] = ((_ds['r4h1'] + _ds['r4h2'])/_ds['tamhog'])**2
    return _ds

        
def clean(ds):
    # Step 1.1
    _calc_feat = ds.loc[:, 'SQBescolari':'agesq'].columns
    print('Columnas eliminadas: ', _calc_feat.values)
    ds.drop(columns=_calc_feat, inplace=True)
    ds.drop(columns=["edjefe", "edjefa", "dependency", "meaneduc"], inplace=True)
    print("Columnas eliminadas: edjefe, edjefa, dependency, meaneduc, rez_esc, hhsize, r4t1, r4t2, r4t3,r4m3, r4h3, hogar_total")
    # Step 2.2
    ds.drop(columns=["rez_esc"], inplace=True)

    # Step 2.5
    ds.drop(columns=["hhsize", 'r4t1', 'r4t2', 'r4t3', 'r4m3', 'r4h3', "hogar_total"], inplace=True)

    ds = fix_techo(ds)
    ds = fix_electricity(ds)
    ds = fix_v18q1(ds)

    hogares = ds[["parentesco1", "idhogar"]].groupby(['idhogar']).sum()
    array_hogares = hogares[hogares.parentesco1 != 1].index.values
    ds = ds[ds.idhogar.isin(list(array_hogares)) == False]

    # Step 2.6
    v2a1_max = ds.v2a1.std() * 3 + ds.v2a1.mean()
    ds = ds[(ds.v2a1 < v2a1_max) | (ds.v2a1.isnull())]

    # Step 3.3
    rename_col_dict = {
        'area1': 'zona_urbana',
        'area2': 'zona_rural',
        'v2a1': 'monthly_rent',
        'lugar1': 'region_central',
        'lugar2': 'region_chorotega',
        'lugar3': 'region_pacifico_central',
        'lugar4': 'region_brunca',
        'lugar5': 'region_huetar_atlantica',
        'lugar6': 'region_huetar_norte'}
    ds.rename(columns=rename_col_dict, inplace=True)

    # Step 5
    ds_paid_rent = ds[(ds.monthly_rent > 0) & (ds.parentesco1 == 1)]
    ds = ds.apply(lambda x: get_costo_de_oportunidad(x, ds_paid_rent), axis=1)
    
    #step 6 add new feautures
    add_synthetic_features(ds)

    cat = len(ds.select_dtypes(include=['object']).columns)
    num = len(ds.select_dtypes(include=['int64', 'float64']).columns)
    print('Total Features: ', cat, 'objetos', '+',
          num, 'numerical', '=', cat + num, 'features')

    return ds

def clean_datadrame(ds):
    return clean(ds)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("No argument supplied")
        exit()
    filename = sys.argv[1]
    if os.path.exists(filename):
        ds = pd.read_csv(filename)
        ds = clean(ds)
        ds.to_csv(os.path.splitext(filename)[0] + "_out.csv")
    else:
        print("File not exists")
