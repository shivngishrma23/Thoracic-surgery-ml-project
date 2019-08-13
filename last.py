import numpy as np
import pickle

def abc(val):
	val = np.array([val]).astype(np.float64)
	obj= pickle.load(open('static/model.pkl','rb'))

	lr=obj

	pred= lr.predict(val)
	# if pred == [1]:
	# 	return "The person will die"
	# elif pred == [2]:
	# 	return "The person will not die"
	return pred 
