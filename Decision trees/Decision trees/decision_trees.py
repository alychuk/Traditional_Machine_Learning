import numpy as np
import math
# BEGIN inputing first set of data
X_Training1 = np.array( [ [0,1], [0,0], [1,0], [0,0], [1,1] ] )
Y_Training1 = np.array( [ [1], [0], [0], [0], [1] ] )

X_Validation1 = np.array( [ [0,0], [0,1], [1,0], [1,1] ] )
Y_Validation1 = np.array( [ [0], [1], [0], [1] ] )

X_Testing1 = np.array( [ [0,0], [0,1], [1,0], [1,1] ] )
Y_Testing1 = np.array( [ [1], [1], [0], [1] ] )
# END inputing first set of data


# BEGIN inputing 2nd set of data
X_Training2 = np.array( [ [0, 1, 0, 0], [0, 0, 0, 1],[1, 0, 0, 0],[0, 0, 1, 1], [1, 1, 0, 1],[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0] ] )
Y_Training2 = np.array( [ [0], [1], [0], [0], [1], [0], [1], [1], [1] ] )

X_Validation2 = np.array( [ [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0] ] )
Y_Validation2 = np.array( [ [0], [0], [1], [0], [1], [1] ] )


X_Testing2 = np.array( [ [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1],[1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0] ] )
Y_Testing2 = np.array( [ [1], [1], [0], [0], [1], [0], [1], [1], [1] ] )
# END inputing 2nd set of data



# BEGIN Data for generating the decision tree (last part of the project)
X_real = np.array([	[4.8, 3.4, 1.9, 0.2],
					[5.0, 3.0, 1.6, 1.2],
				   	[5.0, 3.4, 1.6, 0.2],
				   	[5.2, 3.5, 1.5, 0.2],
				   	[5.2, 3.4, 1.4, 0.2],
				   	[4.7, 3.2, 1.6, 0.2],
				   	[4.8, 3.1, 1.6, 0.2],
				   	[5.4, 3.4, 1.5, 0.4],
				   	[7.0, 3.2, 4.7, 1.4],
				   	[6.4, 3.2, 4.7, 1.5],
				   	[6.9, 3.1, 4.9, 1.5],
				   	[5.5, 2.3, 4.0, 1.3],
				   	[6.5, 2.8, 4.6, 1.5],
				   	[5.7, 2.8, 4.5, 1.3],
				   	[6.3, 3.3, 4.7, 1.6],
				   	[4.9, 2.4, 3.3, 1.0]  ])

Y_real = np.array( [ [1], [1], [1], [1], [1], [1], [1], [1],[0], [0], [0], [0], [0], [0], [0], [0] ] )
# END Data for generating the decision tree (last part of the project)

def mylog2(side):
	return 0 if (side==0) else -side * math.log2(side)

def Proportion(Set, Label):
	return 0 if ((len(Set)) == 0) else (np.count_nonzero(Set == Label)) / len(Set)

def Proportion_real(Set, Threshold):
	count = 0
	for i in range(np.size(Set, 1)):
		count = (Threshold <= Set[:,i])
		print(count)
	return (count / len(Set))

def entropy(Set):
	return mylog2(Proportion(Set, 0)) + mylog2(Proportion(Set, 1))

def informationGain(Xset, label, i):
	return entropy(label) - (Proportion(Xset[:,i], 0) *  entropy(label[Xset[:,i] == 0]) + Proportion(Xset[:,i], 1) * entropy(label[Xset[:,i] == 1]) )

def SplitData(XSet, YSet):
	Position = 0
	BetterInfoGain = 0
	for i in range (np.size(XSet,1)):
		if informationGain(XSet, YSet, i) > BetterInfoGain:
			Position = i
			BetterInfoGain = informationGain(XSet, YSet, i)
	return Position

def ChildTree(Set, label, Side, max_depth):
	if (entropy(Set) == label):
		if (len(Set) !=0):
			Child = Set[0]
		else:
			Child = [0]
	else:
		Child = DT_train_binary(Side, Set, max_depth - 1)
	return Child

def CheckSides(Set,features, RootIndex, value1, value2):
	Left = Set[features[:, RootIndex] == value1]
	Right = Set[features[:, RootIndex] == value2]
	return Left, Right

def delete(Side, Side2, index, value):
	Side  = np.delete(Side, index, value)
	Side2 = np.delete(Side2, index, value)
	return Side, Side2

def DT_train_binary(X,Y,max_depth):
	if (max_depth != 0) and ( X.size != 0):
		index = SplitData(X, Y)
		Left, Right = CheckSides(X,X,index, 0, 1)
		LabelonLeft, LabelonRight = CheckSides(Y, X, index, 0, 1)
		Left, Right = delete(Left, Right, index, 1)

		return [index,ChildTree(LabelonLeft, 0, Left, max_depth-1), ChildTree(LabelonRight, 0, Right, max_depth-1)]
	else:
		return 0 if Proportion(Y, 0) > Proportion(Y, 1) else 1

def DT_test_binary(X,Y,DT):
	Prediction = []
	for sample in X:
		DecisionTree = DT
		while True:
			if (len(DecisionTree) >= 3):
				if sample[DecisionTree[0]] == 0:
					DecisionTree = DecisionTree[1]
				else:
					DecisionTree = DecisionTree[2]
			else:
				break
		Prediction.append(DecisionTree)
	correct = 0
	for i in range(len(Y)):
		if Prediction[i] == Y[i]:
			correct = correct + 1
	return correct/len(Y)

def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
	CurrentDepth = 0
	BestDecisionTree = DT_train_binary(X_train,Y_train, CurrentDepth)
	BestAccuracy = 0
	CurrentAccuracy = -1
	while BestAccuracy > CurrentAccuracy:
		CurrentDecisionTree = BestDecisionTree
		BestDecisionTree = DT_train_binary(X_train,Y_train, CurrentDepth + 1)
		CurrentAccuracy = BestAccuracy
		BestAccuracy = DT_test_binary(X_val,Y_val, BestDecisionTree)
	return CurrentDecisionTree

def RightChild(DT):
	return DT[2]

def LeftChild(DT):
	return DT[1]

def DT_make_prediction(x,DT):
	DecisionTree = DT
	while True:
			if (len(DecisionTree) >= 3):
				if x[DecisionTree[0]!=0]:
					DecisionTree = RightChild(DecisionTree)
				else:
					DecisionTree = LeftChild(DecisionTree)
			else:
				break
	return DecisionTree

def DT_find_thresholds(X):
	return 0 if ((len(Set)) == 0) else (np.count_nonzero(Set == Label)) / len(Set)


def DT_train_real(X,Y,max_depth):
	if (max_depth != 0) and ( X.size != 0):
		index = SplitData(X, Y)
		Left, Right = CheckSides(X,X,index, 0, 1)
		LabelonLeft, LabelonRight = CheckSides(Y, X, index, 0, 1)
		Left, Right = delete(Left, Right, index, 1)
		return [index,ChildTree(LabelonLeft, 0, Left, max_depth-1), ChildTree(LabelonRight, 0, Right, max_depth-1)]
	else:
		return 0 if Proportion(Y, 0) > Proportion(Y, 1) else 1

def DT_test_real(X,Y,DT):
	Prediction = []
	for sample in X:
		DecisionTree = DT
		while True:
			if (len(DecisionTree) >= 3):
				if sample[DecisionTree[0]] == 0:
					DecisionTree = DecisionTree[1]
				else:
					DecisionTree = DecisionTree[2]
			else:
				break
		Prediction.append(DecisionTree)
	correct = 0
	for i in range(len(Y)):
		if Prediction[i] == Y[i]:
			correct = correct + 1
	return correct/len(Y)

def DT_train_real_best(X_train, Y_train, X_val, Y_val):
	CurrentDepth = 0
	BestDecisionTree = DT_train_binary(X_train,Y_train, CurrentDepth)
	BestAccuracy = 0
	CurrentAccuracy = -1
	while BestAccuracy > CurrentAccuracy:
		CurrentDecisionTree = BestDecisionTree
		BestDecisionTree = DT_train_binary(X_train,Y_train, CurrentDepth + 1)
		CurrentAccuracy = BestAccuracy
		BestAccuracy = DT_test_binary(X_val,Y_val, BestDecisionTree)
	return CurrentDecisionTree

def main():
	#proportion = Proportion_real(X_real, 3.1)
	#print(proportion)
	infoGain = informationGain(X_Training2, Y_Training2, 1)
	print(infoGain)

if __name__ == "__main__":
	main()
