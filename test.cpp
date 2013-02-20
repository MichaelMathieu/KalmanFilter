#include "kalman.hpp"
#include<iostream>
using namespace std;

class TestKalman : public KalmanFilter<float> {
public:
  TestKalman()
    :KalmanFilter<float>(1, 0.1f, 0.2f) {
  };
  virtual ~TestKalman() {};
protected:
  virtual int nCommandParams() const { return 1;};
  virtual int nNoise1Params() const { return 1;};
  virtual int nObsParams() const { return 1;};
  virtual int nNoise2Params() const { return 1;};
public:
  virtual matr f(const matr & u, const matr & w, const void* p = NULL) const;
  virtual matr h(const matr & v, const void* = NULL) const;
};

TestKalman::matr TestKalman::f(const matr & u, const matr & w, const void*) const {
  return matr(1, 1, x(0) + 2*u(0) + 0.1*w(0));
};

TestKalman::matr TestKalman::h(const matr & v, const void*) const {
  return matr(1, 1, 0.7*x(0) + 0.2*v(0));
};



int main () {
  TestKalman test;
  cout << test.getX() << " " << test.getCov() << endl;
  test.update(TestKalman::matr(1,1,0.5f), TestKalman::matr(1,1,0.8f*0.7f));
  cout << test.getX() << " " << test.getCov() << endl;
  test.update(TestKalman::matr(1,1,0.0f), TestKalman::matr(1,1,0.9f*0.7f));
  cout << test.getX() << " " << test.getCov() << endl;
  test.update(TestKalman::matr(1,1,1.0f), TestKalman::matr(1,1,2.7f*0.7f));
  cout << test.getX() << " " << test.getCov() << endl;
  return 0;
}
