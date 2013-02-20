#ifndef __KALMAN_FILTER_HPP_190213__
#define __KALMAN_FILTER_HPP_190213__

#include<opencv/cv.h>
#include<cstdlib>

// x_k = f(x_{k-1},u_k,w_{k-1})
//  x : state
//  u : command
//  w : noise1
// y_k = h(x_k, v_k)
//  y : observation
//  v : noise2
template<typename real>
class KalmanFilter {
public:
  typedef cv::Mat_<real> matr;
protected:
  mutable matr x;
  matr cov;
  real covw, covv; // TODO: these could be cov matrices
public:
  inline KalmanFilter(int nStateParams, real covw, real covv);
  inline KalmanFilter(const  KalmanFilter & src);
  inline KalmanFilter & operator=(const KalmanFilter & src);
  virtual ~KalmanFilter() {};
protected:
  inline int nStateParams() const {
    return x.size().height;
  }
  virtual int nCommandParams() const =0;
  virtual int nNoise1Params() const =0;
  virtual int nObsParams() const =0;
  virtual int nNoise2Params() const =0;
public:
  virtual matr f(const matr & u, const matr & w, const void* p = NULL) const =0;
  virtual matr h(const matr & v, const void* p = NULL) const =0;
  // A = df/dx
  virtual matr getA(const void* p = NULL) const {return getA_perturbation(p);};
  // W = df/dw
  virtual matr getW(const void* p = NULL) const {return getW_perturbation(p);};
  // H = dh/dx
  virtual matr getH(const void* p = NULL) const {return getH_perturbation(p);};
  // V = dh/dv
  virtual matr getV(const void* p = NULL) const {return getV_perturbation(p);};
public:
  matr getA_perturbation(const void* p = NULL) const;
  matr getW_perturbation(const void* p = NULL) const;
  matr getH_perturbation(const void* p = NULL) const;
  matr getV_perturbation(const void* p = NULL) const;
  bool testDerivatives(const void* p = NULL) const;
public:
  void update(const matr & u, const matr & y, const void* p = NULL);
  inline const matr & getX() const { return x; };
  inline const matr & getCov() const { return cov; };
};

template<typename real>
KalmanFilter<real>::KalmanFilter(int nStateParams, real covw, real covv)
  :x(nStateParams, 1, (real)0.0f),
   cov(matr::eye(nStateParams, nStateParams)),
   covw(covw), covv(covv) {
}

template<typename real>
KalmanFilter<real>::KalmanFilter(const KalmanFilter<real> & src)
  :x(), cov(), covw(src.covw), covv(src.covv) {
  src.x.copyTo(x);
  src.cov.copyTo(cov);
}

template<typename real>
KalmanFilter<real> & KalmanFilter<real>::operator=(const KalmanFilter<real> & src) {
  if (&src != *this) {
    src.x.copyTo(x);
    src.cov.copyTo(cov);
    covv = src.covv;
    covw = src.covw;
  }
  return *this;
};


template<typename real>
typename KalmanFilter<real>::matr KalmanFilter<real>::getA_perturbation(const void* p) const {
  real eps = 1e-3;
  matr out(nStateParams(), nStateParams());
  matr xm, xp;
  matr matrnull1(nCommandParams(), 1, (real)0.f);
  matr matrnull2(nNoise1Params(), 1, (real)0.f);
  for (int j = 0; j < nStateParams(); ++j) {
    x(j) -= eps;
    xm = f(matrnull1, matrnull2, p);
    x(j) += eps;
    x(j) += eps;
    xp = f(matrnull1, matrnull2, p);
    x(j) -= eps;
    for (int i = 0; i < nStateParams(); ++i)
      out(i, j) = (xp(i) - xm(i)) / (((real)2.f)*eps);
  }
  return out;
}

template<typename real>
typename KalmanFilter<real>::matr KalmanFilter<real>::getW_perturbation(const void* p) const {
  real eps = 1e-3;
  matr out(nStateParams(), nNoise1Params());
  matr noisep(nNoise1Params(), 1, 0.0f), noisem(nNoise1Params(), 1, 0.0f);
  matr xm, xp;
  matr matrnull(nCommandParams(), 1, (real)0.f);
  for (int j = 0; j < nNoise1Params(); ++j) {
    noisem(j) -= eps;
    noisep(j) += eps;
    xm = f(matrnull, noisem, p);
    xp = f(matrnull, noisep, p);
    noisem(j) += eps;
    noisep(j) -= eps;
    for (int i = 0; i < nStateParams(); ++i)
      out(i, j) = (xp(i) - xm(i)) / (((real)2.f)*eps);
  }
  return out;
}

template<typename real>
typename KalmanFilter<real>::matr KalmanFilter<real>::getH_perturbation(const void* p) const {
  real eps = 1e-3;
  matr out(nObsParams(), nStateParams());
  matr xm, xp;
  matr matrnull(nNoise2Params(), 1, (real)0.f);
  for (int j = 0; j < nStateParams(); ++j) {
    x(j) -= eps;
    xm = h(matrnull, p);
    x(j) += eps;
    x(j) += eps;
    xp = h(matrnull, p);
    x(j) -= eps;
    for (int i = 0; i < nObsParams(); ++i)
      out(i, j) = (xp(i) - xm(i)) / (((real)2.f)*eps);
  }
  return out;
}

template<typename real>
typename KalmanFilter<real>::matr KalmanFilter<real>::getV_perturbation(const void* p) const {
  real eps = 1e-3;
  matr out(nObsParams(), nNoise2Params());
  matr noisep(nNoise2Params(), 1, 0.0f), noisem(nNoise2Params(), 1, 0.0f);
  matr xm, xp;
  for (int j = 0; j < nNoise1Params(); ++j) {
    noisem(j) -= eps;
    noisep(j) += eps;
    xm = h(noisem, p);
    xp = h( noisep, p);
    noisem(j) += eps;
    noisep(j) -= eps;
    for (int i = 0; i < nObsParams(); ++i)
      out(i, j) = (xp(i) - xm(i)) / (((real)2.f)*eps);
  }
  return out;
}

template<typename real>
bool KalmanFilter<real>::testDerivatives(const void* p) const {
  matr oldx;
  x.copyTo(oldx);
  double m, eps = 1e-3;
  matr diff;
  bool out = true;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < nStateParams(); ++j)
      x(j) = (float)rand()/(0.25*RAND_MAX)-2.f;
    diff = abs(getA(p) - getA_perturbation(p));
    minMaxLoc(diff, NULL, &m);
    if (m > eps) {
      std::cerr << "KalmanFilter::testDerivatives: getA failed" << std::endl;
      out = false;
      break;
    }
    diff = abs(getW(p) - getW_perturbation(p));
    minMaxLoc(diff, NULL, &m);
    if (m > eps) {
      std::cerr << "KalmanFilter::testDerivatives: getW failed" << std::endl;
      out = false;
      break;
    }
    diff = abs(getH(p) - getH_perturbation(p));
    minMaxLoc(diff, NULL, &m);
    if (m > eps) {
      std::cerr << "KalmanFilter::testDerivatives: getH failed" << std::endl;
      out = false;
      break;
    }
    diff = abs(getV(p) - getV_perturbation(p));
    minMaxLoc(diff, NULL, &m);
    if (m > eps) {
      std::cerr << "KalmanFilter::testDerivatives: getV failed" << std::endl;
      out = false;
      break;
    }
  }
  oldx.copyTo(x);
  return out;
}

template<typename real>
void KalmanFilter<real>::update(const matr & u, const matr & y, const void* p) {
  matr A = getA(p);
  matr W = getW(p);
  // 1) prediction
  x = f(u, matr(nNoise1Params(), 1, (real)0.0f), p);
  cov = A*cov*A.t() + covw*W*W.t();
  // 2) update
  matr H = getH(p);
  matr V = getV(p);
  // TODO: come products are redundants, and do not use inv()
  matr K = cov * H.t() * (H * cov * H.t() + covv * V * V.t()).inv();
  x += K * (y - h(matr(nNoise2Params(), 1, (real)0.0f), p));
  cov = (matr::eye(nStateParams(), nStateParams()) - K * H) * cov;
}  

#endif
