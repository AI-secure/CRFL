
import csv
import copy
train_fileHeader = ["local_model", "round", "epoch", "internal_epoch", "average_loss", "accuracy", "correct_data",
                    "total_data"]
test_fileHeader = ["model", "epoch", "average_loss", "accuracy", "correct_data", "total_data"]
train_result = []  # train_fileHeader
test_result = []  # test_fileHeader
posiontest_result = []  # test_fileHeader
weight_result=[]
norm_result=[]
global_acc_result= []

def save_result_csv(epoch,folder_path):
    train_csvFile = open(f'{folder_path}/train_result.csv', "w")
    train_writer = csv.writer(train_csvFile)
    train_writer.writerow(train_fileHeader)
    train_writer.writerows(train_result)
    train_csvFile.close()

    test_csvFile = open(f'{folder_path}/test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(test_result)
    test_csvFile.close()

    if len(weight_result)>0:
        weight_csvFile=  open(f'{folder_path}/weight_result.csv', "w")
        weight_writer = csv.writer(weight_csvFile)
        weight_writer.writerows(weight_result)
        weight_csvFile.close()
    
    if len(norm_result)>0:
        norm_csvFile=  open(f'{folder_path}/norm_result.csv', "w")
        norm_writer = csv.writer(norm_csvFile)

        norm_writer.writerows(norm_result)
        norm_csvFile.close()


    if len(global_acc_result)>0:
        global_acc_csvFile=  open(f'{folder_path}/global_acc_result.csv', "w")
        global_acc_writer = csv.writer(global_acc_csvFile)
    
        global_acc_writer.writerows(global_acc_result)
        global_acc_csvFile.close()

    if len(posiontest_result)>0:
        test_csvFile = open(f'{folder_path}/posiontest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(posiontest_result)
        test_csvFile.close()

     

def add_global_acc_result(acc):
    global_acc_result.append([acc])
 

def add_norm_result(norm):
    norm_result.append([norm])


def add_weight_result(name,weight,alpha):

    weight_result.append(weight)



