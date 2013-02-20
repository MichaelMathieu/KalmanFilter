CXX=g++ -g -W -Wall -Wextra
LINK=`pkg-config --libs opencv`

all:
	${CXX} test.cpp ${LINK}