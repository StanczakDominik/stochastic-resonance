ostateczny_wykres.png: wykres.csv read_plot.py
	python read_plot.py

wykres.csv: data3.hdf5 fourier.py
	python fourier.py

data3.hdf5: stochastic.py
	python stochastic.py
