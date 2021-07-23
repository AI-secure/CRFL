import csv
import pandas as pd
import numpy as np

filepath = './data/loan.csv'
df = pd.read_csv(filepath)
# Copy Dataframe
data= df.copy()
data = data.drop(['policy_code','id','member_id','emp_title','issue_d','zip_code','emp_length','title','earliest_cr_line','last_pymnt_d','hardship_start_date','desc','hardship_end_date','payment_plan_start_date','next_pymnt_d','settlement_date','last_credit_pull_d','debt_settlement_flag_date','sec_app_earliest_cr_line'], axis=1)
data = data.drop(['url','mths_since_last_delinq','mths_since_last_major_derog','mths_since_last_record','annual_inc_joint','dti_joint','verification_status_joint','mths_since_recent_bc_dlq','mths_since_recent_revol_delinq','revol_bal_joint','sec_app_inq_last_6mths','sec_app_mort_acc','sec_app_open_acc','sec_app_revol_util','sec_app_open_act_il',
                'sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med','sec_app_mths_since_last_major_derog','hardship_type','hardship_reason','hardship_status','deferral_term','hardship_amount','hardship_length','hardship_dpd','hardship_loan_status','orig_projected_additional_accrued_interest','hardship_payoff_balance_amount',
                'hardship_last_payment_amount','settlement_status','settlement_amount','settlement_percentage','settlement_term'], axis=1)
data = data.fillna(0)

columns = data.columns
df = data.copy()
list_obj = []

list_value=[]


print("isnull", df.isnull().any())

for i in range(len(columns)):
   
    if (df.loc[:, columns[i]].dtype == 'object') and (columns[i] != 'addr_state'):
        print(columns[i])
        list_obj.append(columns[i])
        value = list(df.drop_duplicates(columns[i]).loc[:, columns[i]])
        print(value)
        if  columns[i] == 'loan_status': # label!!
            for j in range(len(value)):
                df.loc[df[columns[i]] == value[j], columns[i]] = j
        else:
            for j in range(len(value)):
                df.loc[df[columns[i]] == value[j], columns[i]] = float(j/len(value))
    elif (df.loc[:, columns[i]].dtype == 'float64') or (df.loc[:, columns[i]].dtype == 'int64'):
        list_value.append(columns[i])
        print(columns[i])
        df[columns[i]]= ( df[columns[i]] -  df[columns[i]].min()) / ( df[columns[i]].max() -  df[columns[i]].min()) 


print("isnull", df.isnull().any())

state_set = list(set(df.loc[:,'addr_state']))

save_dirs = './data/loan/'

import os
if not os.path.exists(save_dirs):
    os.makedirs(save_dirs)

for j in range(len(state_set)):
    print('saving: ', state_set[j])
    data_new = df.loc[df['addr_state']== state_set[j]].drop(['addr_state'], axis=1)
    with open(save_dirs+'/loan_'+ str(state_set[j]) + '.csv', 'w', newline='',encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(data_new.columns)
        for i in range(data_new.shape[0]):
            csv_writer.writerow(data_new.iloc[[i][0]])
    
print('done')