import csv

#write file
nfile = open('fnout.csv','w',newline='',encoding='utf8')
pwriter = csv.writer(nfile,delimiter = ',')

with open('fnoutJAN.csv',newline='',encoding='utf8') as csvfile:
    preader = csv.reader(csvfile,delimiter=',')
    for row in preader:
        pwriter.writerow([row[0]])
with open('fnoutFEB.csv',newline='',encoding='utf8') as csvfile:
    preader = csv.reader(csvfile,delimiter=',')
    for row in preader:
        pwriter.writerow([row[0]])
with open('fnoutMAR.csv',newline='',encoding='utf8') as csvfile:
    preader = csv.reader(csvfile,delimiter=',')
    for row in preader:
        pwriter.writerow([row[0]])
with open('fnoutJAN.csv',newline='',encoding='utf8') as csvfile:
    preader = csv.reader(csvfile,delimiter=',')
    for row in preader:
        pwriter.writerow([row[0]])
with open('fnoutJAN.csv',newline='',encoding='utf8') as csvfile:
    preader = csv.reader(csvfile,delimiter=',')
    for row in preader:
        pwriter.writerow([row[0]])
nfile.close()
