import csv
import numpy as np
import pandas as pd

def resultsToEHData():
	with open('EHdata.csv', 'a' ,newline='') as ok:
		writer = csv.writer(ok)
		writer.writerow(['','E11','E22','E33','E44','E55','E66','u12','u23','u13'])
		with open('CHresults.csv', 'r') as f:
			data = list(csv.reader(f))
			for i in range(1000):
				label = data[i*7][0].split('.')[0]
				CH = np.array(data[i*7+1:i*7+7]).astype(float)
				EH = np.linalg.inv(CH)
				writer.writerow([label,1/EH[0,0],1/EH[1,1],1/EH[2,2],1/EH[3,3],1/EH[4,4],1/EH[5,5],-EH[0,1]/EH[0,0],-EH[1,2]/EH[1,1],-EH[0,2]/EH[0,0]])

def CHdataToCHresults():
	with open('CHresults.csv', 'a', newline='') as f:
		writer = csv.writer(f)
		CHdata = pd.read_csv('CHdata.csv',index_col=0)
		for i in CHdata.index:
			CH = np.zeros((6,6))
			for j in range(6):
				CH[j,j] = CHdata.loc[i].iloc[j]
			CH[0,1]=CH[1,0]=CHdata.loc[i,'C12']
			CH[1,2]=CH[2,1]=CHdata.loc[i,'C23']
			CH[0,2]=CH[2,0]=CHdata.loc[i,'C13']
			writer.writerow([i,])
			writer.writerows(CH)


if __name__ == '__main__':
	CHdataToCHresults()
'''
with open('results.csv', 'r') as f:
	data = list(csv.reader(f))
	CH = np.array(data[(0*7+1):(0*7+7)]).astype(float)
	EH = np.linalg.inv(CH)
	print(EH)
'''