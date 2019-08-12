import csv

nfile = open('out.csv','w',newline='',encoding='utf8')
pwriter = csv.writer(nfile,delimiter = ',')

#abc news
with open('abcnews-date-text.csv',newline='',encoding='utf8') as csvfile:
    preader = csv.reader(csvfile, delimiter= ',')
    line_count = 0
    for row in preader:
        if line_count == 0:
            line_count+=1
        else:
            line_count+=1
            #print('{}'.format(row[1]))
            pwriter.writerow(['{}'.format(row[1])])
    print(line_count)
#examiner use
with open ('examiner-date-text.csv',newline='',encoding='utf8') as csvfile2:
    preader = csv.reader(csvfile2,delimiter=',')
    line_count=0
    for row in preader:
        if line_count == 0:
            line_count+=1
        else:
            line_count+=1
            #print('{}'.format(row[1]))
            pwriter.writerow([row[1]])
    print(line_count)
#irish times
with open('irishtimes-date-text1.csv',newline='',encoding='utf8') as csvfile3:
    preader = csv.reader(csvfile3,delimiter=',')
    line_count=0
    for row in preader:
        if line_count == 0:
            line_count+=1
        else:
            line_count+=1
            #print('{}'.format(row[2]))
            pwriter.writerow([row[2]])
    print(line_count)
#irish times part 2
with open('w3-latnigrin-text.csv',newline='',encoding='utf8') as csvfile4:
    preader = csv.reader(csvfile4,delimiter=',')
    line_count = 0
    for row in preader:
        if line_count == 0:
            line_count+=1
        else:
            line_count+=1
            pwriter.writerow([row[1]])
    print(line_count)
#india news headlines
with open('india-news-headlines.csv',newline='',encoding='utf8') as csvfile5:
    preader = csv.reader(csvfile5,delimiter=',')
    line_count = 0
    for row in preader:
        if line_count ==0:
            line_count+=1
        else:
            line_count+=1
            pwriter.writerow([row[2]])
    print(line_count)
    
nfile.close()