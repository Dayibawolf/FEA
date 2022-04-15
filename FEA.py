import os
import numpy as np
from datetime import datetime

def changeToDir(folder = 'Simulation_results'):
	path = os.getcwd()
	if os.path.exists(folder):
		os.chdir(os.path.join(path, folder))
	else:
		os.makedirs(folder)
		os.chdir(os.path.join(path, folder))
	return path, os.path.join(path, folder)

def delExcept(file = ['.odb', '.inp', '.py']):
    path = os.getcwd()
    fileName = os.listdir(path)
    for f in fileName:
        if os.path.splitext(f)[1] not in file:
            os.remove(os.path.join(path, f))
    return file

def createMesh(dim = (10,10,10), size = 1.0):
	dim = np.array(dim)
	ndim = dim+1
	#创建node和element编号
	nodeId = np.reshape(range(1,np.prod(ndim)+1),ndim, order = 'F')
	eleId = np.reshape(range(1,np.prod(dim)+1),dim, order = 'F')

	nodes = {}
	nodeflag = {}
	if(len(dim) == 2):
		#element关联node编号
		node = np.pad(nodeId[:-1,:-1].reshape((np.prod(dim),1),order = 'F'),((0,0),(0,3)),'edge')
		add  = np.array([[ 0, 1, ndim[0]+1, ndim[0] ]])
		add  = np.pad(add,((0,np.prod(dim)-1),(0,0)),'edge')
		elements = node+add
		#关联node与node坐标，并初始化nodeflag
		for y in range(ndim[1]):
			for x in range(ndim[0]):
				nodes[nodeId[x,y]] = (size*x, size*y, 0.0)
				nodeflag[nodeId[x,y]] = 0
		#标记node使用次数nodeflag
		for ele in elements:
			for nId in ele:
				nodeflag[nId] = nodeflag[nId]+1
	elif(len(dim) == 3):
		#element关联node编号
		node = np.pad(nodeId[:-1,:-1,:-1].reshape((np.prod(dim),1),order = 'F'),((0,0),(0,7)),'edge')
		add  = np.array([[ 0, 1, ndim[0]+1, ndim[0], ndim[0]*ndim[1], ndim[0]*ndim[1]+1, ndim[0]*ndim[1]+ndim[0]+1, ndim[0]*ndim[1]+ndim[0] ]])
		add  = np.pad(add,((0,np.prod(dim)-1),(0,0)),'edge')
		elements = node+add
		#关联node与node坐标，并初始化nodeflag
		for z in range(ndim[2]):
			for y in range(ndim[1]):
				for x in range(ndim[0]):
					nodes[nodeId[x,y,z]] = (size*x, size*y, size*z)
					nodeflag[nodeId[x,y,z]] = 0
		#标记node使用次数nodeflag
		for ele in elements:
			for nId in ele:	nodeflag[nId] += 1
		#角点node标记+1
		node = [nodeId[0,0, 0],nodeId[-1,0, 0],nodeId[-1,-1, 0],nodeId[0,-1, 0],
	   		    nodeId[0,0,-1],nodeId[-1,0,-1],nodeId[-1,-1,-1],nodeId[0,-1,-1]]
		for n in node: nodeflag[n] += 1
	else:
		print('error:createMesh()-->dim')
		return 0, 0, 0, 0, 0
	return nodes, nodeId, nodeflag, elements, eleId

def writeInp(nodes, elements, type = 'S4R', sets = '', equation = '', Fix = (1, 1, 1), BC = ('U2', 1, 1, 0.1), file = 'test'):
	#write inp
	file = '{}.inp'.format(file)
	with open(file, 'w') as f:
		f.write('*Node\n')
		for i in nodes:
			f.write('{:>d}, {: >f}, {: >f}, {: >f}\n'.format(i,nodes[i][0],nodes[i][1],nodes[i][2]))
		f.write('*Element,type={},Elset=Micro\n'.format(type))
		for i in range(len(elements)):
			line = '{:>}'.format(i+1)
			for j in elements[i]:
				line += ', {:>}'.format(j)
			f.write('{}\n'.format(line))
		f.write('*SOLID SECTION, ELSET=Micro, MATERIAL=Micro\n')
		for i in sets:
			f.write('*Nset,Nset={}\n{: >10d},\n'.format(i,sets[i]))
		for i in equation:
			f.write('*Equation\n{}\n'.format(len(i)))
			for j in i:
				f.write('{}, {}, {}\n'.format(j[0],j[1],j[2]))
		f.write('*Material, name=Micro\n*Elastic\n{},{}\n'.format(100,0.3))
		f.write('*Boundary\n{},1,1\n{},2,2\n{},3,3\n'.format(Fix[0],Fix[1],Fix[2]))
		f.write('*Step, name=Step-1, nlgeom=NO\n')
		f.write('*Static\n{},{},{},{}\n'.format(1.0, 1.0, 1e-5, 1.0))
		f.write('*Boundary\n{},{},{},{}\n'.format(BC[0], BC[1], BC[2], BC[3]))
		f.write('*Output, field, variable=PRESELECT\n')
		f.write('*Output, history, variable=PRESELECT\n')
		f.write('*End Step\n')

def delElement(nodes, nodeflag, elements, eleId, tensor):
	#删除空单元和相关节点
	Id = eleId[np.where(tensor == 1)]-1
	for ele in elements[Id]:
		for n in ele:
			nodeflag[n] -= 1
			if (nodeflag[n] < 1): nodes.pop(n)
	elements = np.delete(elements, Id, axis = 0)
	return nodes, nodeflag, elements

def Equation(equation, a, b, m, axis = 1):
	#根据规则创建约束方程
	if axis == 1: index = [1,]
	if axis == 2: index = [1,2]
	if axis == 3: index = [1, 2, 3]
	for i in [1,2,3]:
		eq = []
		eq.append([a, i, 1.0])
		eq.append([b, i,-1.0])
		if(i in index):
			eq.append([m, i, -1.0])
		equation.append(eq)

def EquationSYS(equation, b, o, i):
	eq = []
	eq.append([b, i, 1.0])
	eq.append([o, i,-1.0])
	equation.append(eq)

def isActiveN(A, nodeflag):
	#判断节点是否存在
	tA = {}
	for i in range(len(A)):
		tA[i] = []
	for i in range(len(A[0])):
		if nodeflag[A[0][i]] > 0:
			for j in range(len(A)):
				tA[j].append(A[j][i])
	return (tA[i] for i in tA)

def createEquation3D(nodeId, nodeflag):
	equation = []
	#6个面
	A = nodeId[-1,1:-1,1:-1].reshape(nodeId[-1,1:-1,1:-1].size)
	B = nodeId[0,1:-1,1:-1].reshape(nodeId[0,1:-1,1:-1].size)
	A, B = isActiveN([A, B], nodeflag)

	C = nodeId[1:-1,-1,1:-1].reshape(nodeId[1:-1,-1,1:-1].size)
	D = nodeId[1:-1,0,1:-1].reshape(nodeId[1:-1,0,1:-1].size)
	C, D = isActiveN([C, D], nodeflag)

	E = nodeId[1:-1,1:-1,-1].reshape(nodeId[1:-1,1:-1,-1].size)
	F = nodeId[1:-1,1:-1,0].reshape(nodeId[1:-1,1:-1,0].size)
	E, F = isActiveN([E, F], nodeflag)

	#12条边
	ei    = nodeId[0,0,1:-1]
	eii   = nodeId[-1,0,1:-1]
	eiii  = nodeId[-1,-1,1:-1]
	eiv   = nodeId[0,-1,1:-1]
	ei, eii, eiii, eiv = isActiveN([ei, eii, eiii, eiv], nodeflag)
	ev    = nodeId[0,1:-1,0]
	evi   = nodeId[-1,1:-1,0]
	evii  = nodeId[-1,1:-1,-1]
	eviii = nodeId[0,1:-1,-1]
	ev, evi, evii, eviii = isActiveN([ev, evi, evii, eviii], nodeflag)
	eix   = nodeId[1:-1,0,0]
	ex    = nodeId[1:-1,-1,0]
	exi   = nodeId[1:-1,-1,-1]
	exii  = nodeId[1:-1,0,-1]
	eix, ex, exi, exii = isActiveN([eix, ex, exi, exii], nodeflag)

	#8个顶点
	node = [nodeId[0,0, 0],nodeId[-1,0, 0],nodeId[-1,-1, 0],nodeId[0,-1, 0],
	   		 nodeId[0,0,-1],nodeId[-1,0,-1],nodeId[-1,-1,-1],nodeId[0,-1,-1]]

	#创建坐标系，o：坐标原点，x,y,z：三个轴向载荷施加参考点
	o, x, y, z, = node[0], node[1], node[3], node[4]
	csys = (o, x, y, z)

	#创建约束方程（根据04文献）
	if nodeflag[o]>1:
		Equation(equation, node[2], node[3], x, 1)
		Equation(equation, node[5], node[4], x, 1)
		Equation(equation, node[6], node[2], z, 3)
		Equation(equation, node[7], node[4], y, 2)
	else:
		EquationSYS(equation, B[0], o, 1)
		EquationSYS(equation, D[0], o, 2)
		EquationSYS(equation, F[0], o, 3)
	#边和面
	eqNode = [(A  , B  , 1),(C   , D    , 2),(E    , F  , 3),
			  (eii, ei , 1),(eiii, eiv  , 1),(eiv  , ei , 2),
			  (evi, ev , 1),(evii, eviii, 1),(eviii, ev , 3),
			  (ex , eix, 2),(exi , exii , 2),(exii , eix, 3)]
	for n in eqNode:
		for a,b in zip(n[0], n[1]):
			Equation(equation, a, b, csys[n[2]], n[2])
	return equation, o, x, y, z

def run(tensor, fileName = ''):
	dim = tensor.shape
	nodes, nodeId, nodeflag, elements, eleId = createMesh(dim = dim, size = 1.0)
	nodes, nodeflag, elements = delElement(nodes, nodeflag, elements, eleId, tensor)
	equation, o, x, y, z = createEquation3D(nodeId, nodeflag)
	oldPath, newPath = changeToDir(folder = 'Simulation_results')
	if not fileName:
		fileName = 'voxel-{}'.format(datetime.now().strftime('%Y-%m-%d-%H_%M_%S'))
	writeInp(nodes, elements, type = 'C3D8I', equation = equation, 
		     Fix = (o, o, o), BC = (x, 1, 1, 0.5), file = fileName+'-x')
	writeInp(nodes, elements, type = 'C3D8I', equation = equation, 
		     Fix = (o, o, o), BC = (y, 2, 2, 0.5), file = fileName+'-y')
	writeInp(nodes, elements, type = 'C3D8I', equation = equation, 
		     Fix = (o, o, o), BC = (z, 3, 3, 0.5), file = fileName+'-z')
	os.system('abaqus cae noGui={} -- {} {} {} {}'.format(oldPath+'\\Calculation.py', fileName, x, y, z))
	delExcept(file = ['.odb', '.inp', '.py', '.cae', '.csv'])
	os.chdir(oldPath)

if __name__ == '__main__':
#	os.system('abaqus cae script=test.py -- ttt')
	#tensor = np.load('voxelfea-2021-11-24-15_21_54.npy')
	'''
	tensor = np.array([[[1,1,1],[1,0,1],[1,1,1]],
                       [[1,0,1],[0,0,0],[1,0,1]],
                       [[1,1,1],[1,0,1],[1,1,1]]])
	tensor = np.array([[[0,0,0],[0,0,0],[0,0,0]],
                       [[0,0,0],[0,0,0],[0,0,0]],
                       [[0,0,0],[0,0,0],[0,0,0]]])
	tensor = postpro_matrix(tensor,3,3,3)
	'''
	tensor = np.array([[[1,1,1],[1,0,1],[1,1,1]],
                       [[1,0,1],[0,0,0],[1,0,1]],
                       [[1,1,1],[1,0,1],[1,1,1]]])
	run(tensor, fileName = 'test1')
#	print(tensor)

#	tensor = np.array([[[0,0,0],[0,0,0],[0,0,0]],
#                         [[1,1,1],[1,1,1],[1,1,1]],
#                         [[0,0,0],[0,0,0],[0,0,0]]])
#	dim = tensor.shape
#	nodes, nodeId, nodeflag, elements, eleId = createMesh(dim = (10,10,10), size = 1.0)
#	nodes, nodeflag, elements = delElement(nodes, nodeflag, elements, eleId, tensor)
#	sets, equation = createEquation3D(nodeId, nodeflag)
#	writeInp(nodes, elements, type = 'C3D8I',sets = sets, equation = equation)
#	print(nodeId)
#	print(nodes)
#	print(eleId)
#	print(elements)
	#os.system('abaqus cae noGui=Calculation.py')
#	print(nodeflag)
#	os.system('pause')
