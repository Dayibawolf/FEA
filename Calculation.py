import sys
import csv
from abaqus import *
from abaqusConstants import *
from caeModules import *
from odbAccess import *
import os

def getRF(odbname, nodeLabel, RF1):
    odb=openOdb(odbname)
    instance = odb.rootAssembly.instances['PART-1-1']
    node = instance.getNodeFromLabel(nodeLabel)
    lastFrame = odb.steps['Step-1'].frames[-1]
    RF=lastFrame.fieldOutputs['RF'].getSubset(region = node).values[0].data[RF1]
    odb.close()
    return RF
#l = 'echo '+str(sys.argv)
#os.system(l)
argv = sys.argv[10:]
Name = argv[0]
names = [Name+'-x', Name+'-y', Name+'-z']
nodeLabel = [int(argv[1]), int(argv[2]), int(argv[3])]
i = 0
RF = {}
for name in names:
    source = '{}.inp'.format(name)
    mdb.JobFromInputFile(name=name, inputFileName=source, 
        type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
        memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=6,
        numDomains=6, numGPUs=0)
    mdb.jobs[name].submit(consistencyChecking=OFF)
    mdb.jobs[name].waitForCompletion()
    odbname = '{}.odb'.format(name)
    RF[i] = getRF(odbname, nodeLabel[i], i)/0.5
    i += 1
with open('results.csv', 'ab') as f:
    writer = csv.writer(f)
    writer.writerow([Name, RF[0], RF[1], RF[2]])
    
