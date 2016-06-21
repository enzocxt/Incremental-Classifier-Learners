##
## Makefile
##  
## Made by Benjamin Negrevergne
## Login   <benjaminneb.dyn.cs.kuleuven.be>
##
## Started on  Sat Nov 15 12:19:51 2014 Benjamin Negrevergne
## Last update <2015-11-18 Wed> Jessa Bekker
## 

ZOO_TRAIN=../data/zoo-train.dat
ZOO_TEST=../data/zoo-test.dat

# Big synthetic dataset, no noise
BD_TRAIN=../data/synthetic/bigdata.0.25.0.00.train
BD_TEST=../data/synthetic/bigdata.0.25.0.00.test

# Big synthetic dataset, with 25% noise
BND_TRAIN=../data/synthetic/bigdata.0.25.0.25.train
BND_TEST=../data/synthetic/bigdata.0.25.0.25.test

ALPHA=0.001
DELTA=0.0001
TAU=0.05

all: LogisticRegression.class Vfdt.class

Data.class: Data.java
	javac $<

LogisticRegression.class: LogisticRegression.java Data.class
	javac $<


VfdtNode.class: Vfdt.java
	javac $<

Vfdt.class: Vfdt.java VfdtNode.class Data.class
	javac $<

testlr: LogisticRegression.class
	@echo "Testing LR on Zoo"
	rm -f out.lr.acc ;
	time java  LogisticRegression $(ALPHA) $(ZOO_TRAIN) $(ZOO_TEST) out 2
	@echo "Testing LR on BD"	
	rm -f out.lr.acc ;				
	time java  LogisticRegression $(ALPHA) $(BD_TRAIN) $(BD_TEST) out 2
	@echo "Testing LR on BND"	
	rm -f out.lr.acc ;				
	time java  LogisticRegression $(ALPHA) $(BND_TRAIN) $(BND_TEST) out 2

testvfdt: Vfdt.class
	@echo "Testing VFDT on Zoo" 
	rm -f out.vfdt.acc ; 
	time java  Vfdt $(DELTA) $(TAU) $(ZOO_TRAIN) $(ZOO_TEST) out 2 
	@echo "Testing VFDT on BD" 
	rm -f out.vfdt.acc ; 
	time java  Vfdt $(DELTA) $(TAU)  $(BD_TRAIN) $(BD_TEST) out 2 
	@echo "Testing VFDT on BND" 
	rm -f out.vfdt.acc ; 
	time java  Vfdt $(DELTA) $(TAU) $(BND_TRAIN) $(BND_TEST) out 2
