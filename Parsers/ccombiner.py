import csv

#write file
nfile = open('out08_02_2019.csv','w',newline='',encoding='utf8')
pwriter = csv.writer(nfile,delimiter = ',')

with open('out.csv',newline='',encoding='utf8') as csvfile:
    preader=csv.reader(csvfile,delimiter=',')
    for row in preader:
        pwriter.writerow([row[0]])
with open('fnout.csv',newline='',encoding='utf8') as csvfile:
    preader=csv.reader(csvfile,delimiter=',')
    for row in preader:
        pwriter.writerow([row[0]])
        
nfile.close()