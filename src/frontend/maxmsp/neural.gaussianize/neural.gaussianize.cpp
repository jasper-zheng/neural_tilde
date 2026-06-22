// neural.gaussianize — map a uniform distribution to a standard-normal one.
//
// Takes values assumed to be uniform on [min, max] and outputs the corresponding 
// standard-normal N(0,1) samples via the probit (inverse normal CDF), z = Phi^-1((x - min) / (max - min)).
//
// Why: generative models (e.g. [neural.gen~]) want N(0,1) noise, but Max's
// `jit.noise` emits uniform [0,1].
//
//   [jit.noise 7 float32 32 256] -> [neural.gaussianize] -> [neural.gen~] (noises inlet)
//   @min 0.  @max 1.    (the assumed input uniform range; defaults match jit.noise)
//
// Accepts EITHER a Jitter matrix (any shape; output type matches the input,
// float32 in -> float32 out, float64 in -> float64 out) OR a list / single
// number. Two fixed-type outlets: left = matrix (for matrix input), right = list
// (for list/number input).

#include "c74_min.h"

#include <cmath>
#include <string>

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

using namespace c74::min;
namespace mx = c74::max;

// Acklam's rational approximation of the inverse standard-normal CDF (probit);
// Maps a uniform(0,1) value to its N(0,1) quantile (the exact inverse-CDF transform: uniform -> normal); the
// input is clamped just inside (0,1) so the tails don't blow up.
static double inv_normal_cdf(double p) {
  if (p <= 0.0)
    p = 1e-9;
  else if (p >= 1.0)
    p = 1.0 - 1e-9;
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                             -2.759285104469687e+02, 1.383577518672690e+02,
                             -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                             -1.556989798598866e+02, 6.680131188771972e+01,
                             -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                             -2.400758277161838e+00, -2.549732539343734e+00,
                             4.374664141464968e+00,  2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                             2.445134137142996e+00, 3.754408661907416e+00};
  const double plow = 0.02425, phigh = 1.0 - plow;
  double q, r;
  if (p < plow) {
    q = std::sqrt(-2.0 * std::log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else if (p <= phigh) {
    q = p - 0.5;
    r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) *
           q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  }
  q = std::sqrt(-2.0 * std::log(1.0 - p));
  return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
         ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
}

class gaussianize : public object<gaussianize> {
public:
  MIN_DIGEST{"Gaussinize a uniform distribution"};
  MIN_DESCRIPTION{"Map a uniform distribution to a standard-Gaussian N(0,1) distribution (mean = 0, std = 1). "};
  MIN_TAGS{"neural audio synthesis, generative models, jitter, math"};
  MIN_AUTHOR{"Jasper Shuoyang Zheng"};
  MIN_RELATED{"neural.gen~, neural.live~, jit.noise, jit.matrix"};

  gaussianize(const atoms &args = {}) {}
  ~gaussianize();

  inlet<> m_in{this, "(matrix/list) uniform input", "matrix"};
  // Two fixed-type outlets: a matrix output (left) for matrix input, and a list
  // output (right) for list/number input. Declaration order = left-to-right.
  // The matrix outlet MUST be typed "jit_matrix" (the real Jitter outlet type) so
  // Max recognises it as a matrix outlet (matrix-coloured patch cords); a plain
  // object<> creates the outlet verbatim via outlet_new(x, type), and "matrix" is
  // only a min-internal marker used by the MOP path (which we don't take).
  outlet<> m_matrix_out{this, "(matrix) gaussianized matrix", "jit_matrix"};
  outlet<> m_list_out{this, "(list) gaussianized list", "list"};

  // The assumed input uniform range [min, max]. Defaults match jit.noise [0,1].
  attribute<number> min{this, "min", 0.0,
                        description{"Low end of the assumed input uniform range."}};
  attribute<number> max{this, "max", 1.0,
                        description{"High end of the assumed input uniform range."}};

  message<> jit_matrix_msg{
      this, "jit_matrix", "Gaussianize a Jitter float32/float64 matrix.",
      MIN_FUNCTION {
        if (!args.empty())
          receive_matrix(args[0]);
        return {};
      }};

  message<> list_msg{
      this, "list", "Gaussianize each number; output a list of the same length.",
      MIN_FUNCTION {
        double lo, hi;
        const bool map = range(lo, hi);
        atoms out;
        out.reserve(args.size());
        for (const auto &a : args)
          out.push_back(map ? inv_normal_cdf(((double)a - lo) / (hi - lo))
                            : (double)a);
        m_list_out.send(out);
        return {};
      }};

  message<> number_msg{
      this, "number", "Gaussianize a single number.", MIN_FUNCTION {
        double lo, hi;
        const bool map = range(lo, hi);
        m_list_out.send(map ? inv_normal_cdf(((double)args[0] - lo) / (hi - lo))
                            : (double)args[0]);
        return {};
      }};

  message<> maxclass_setup{
      this, "maxclass_setup", [this](const atoms &, const int) -> atoms {
        cout << "neural.gaussianize " << VERSION << " - Jasper Shuoyang Zheng"
             << endl;
        return {};
      }};

private:
  // Reusable, registered output matrix (created on first matrix input, resized
  // and re-typed to match each input's geometry and type, freed in the
  // destructor). We emit its name so the downstream reads it just like any
  // jit_matrix.
  void *m_outmat{nullptr};
  symbol m_outname;
  bool m_warned_range{false};

  // Read [min, max]; returns false (and warns once) if the range is degenerate,
  // in which case callers pass values through unchanged.
  bool range(double &lo, double &hi) {
    lo = min;
    hi = max;
    if (hi <= lo) {
      if (!m_warned_range) {
        cerr << "max must be greater than min — passing values through" << endl;
        m_warned_range = true;
      }
      return false;
    }
    m_warned_range = false;
    return true;
  }

  bool ensure_outmat(mx::t_jit_matrix_info want);
  void receive_matrix(symbol name);
};

gaussianize::~gaussianize() {
  if (m_outmat)
    mx::jit_object_free(m_outmat);
}

// (Re)configure the reusable output matrix to `want` (type set by the caller to
// match the input). Created lazily and registered under a unique name on first
// use; resized/re-typed thereafter.
bool gaussianize::ensure_outmat(mx::t_jit_matrix_info want) {
  if (!m_outmat) {
    m_outmat = mx::jit_object_new(mx::_jit_sym_jit_matrix, &want);
    if (!m_outmat) {
      cerr << "could not create output matrix" << endl;
      return false;
    }
    m_outname = mx::jit_symbol_unique();
    m_outmat = mx::jit_object_register(m_outmat, (mx::t_symbol *)m_outname);
  } else {
    mx::object_method(m_outmat, mx::_jit_sym_setinfo, &want);
  }
  return true;
}

// Read the named input matrix, gaussianize every cell into our output matrix
// (same dimcount/dim/planecount and same type as the input), and emit it. Any
// shape is handled by a flat odometer over the dims honoring dimstride (bytes).
// float32/float64 in -> same type out.
void gaussianize::receive_matrix(symbol name) {
  void *in = mx::jit_object_findregistered((mx::t_symbol *)name);
  if (!in) {
    cerr << "matrix '" << std::string(name) << "' not found" << endl;
    return;
  }
  if (mx::object_classname((mx::t_object *)in) != mx::_jit_sym_jit_matrix) {
    cerr << "'" << std::string(name) << "' is not a jit_matrix" << endl;
    return;
  }

  void *savelock = mx::object_method(in, mx::_jit_sym_lock, (void *)1);
  mx::t_jit_matrix_info ininfo;
  mx::object_method(in, mx::_jit_sym_getinfo, &ininfo);

  const bool is_f32 = ininfo.type == mx::_jit_sym_float32;
  const bool is_f64 = ininfo.type == mx::_jit_sym_float64;
  char *inbp = nullptr;
  if (is_f32 || is_f64)
    mx::object_method(in, mx::_jit_sym_getdata, &inbp);

  bool ok = (is_f32 || is_f64) && inbp;
  mx::t_jit_matrix_info real_out;
  char *outbp = nullptr;
  if (ok) {
    mx::t_jit_matrix_info outinfo = ininfo; // same geometry and type as input
    ok = ensure_outmat(outinfo);
    if (ok) {
      mx::object_method(m_outmat, mx::_jit_sym_getinfo, &real_out);
      mx::object_method(m_outmat, mx::_jit_sym_getdata, &outbp);
      ok = (outbp != nullptr);
    }
  }

  if (ok) {
    double lo, hi;
    const bool map = range(lo, hi);
    const double inv = map ? 1.0 / (hi - lo) : 0.0;
    const long dc = ininfo.dimcount;
    const long pc = ininfo.planecount;
    long total = 1;
    for (long k = 0; k < dc; k++)
      total *= ininfo.dim[k];

    long coord[mx::JIT_MATRIX_MAX_DIMCOUNT] = {0};
    for (long n = 0; n < total; n++) {
      const char *ip = inbp;
      char *op = outbp;
      for (long k = 0; k < dc; k++) {
        ip += (size_t)coord[k] * ininfo.dimstride[k];
        op += (size_t)coord[k] * real_out.dimstride[k];
      }
      for (long pl = 0; pl < pc; pl++) {
        const double x = is_f32 ? (double)reinterpret_cast<const float *>(ip)[pl]
                                : reinterpret_cast<const double *>(ip)[pl];
        const double z = map ? inv_normal_cdf((x - lo) * inv) : x;
        if (is_f32)
          reinterpret_cast<float *>(op)[pl] = (float)z;
        else
          reinterpret_cast<double *>(op)[pl] = z;
      }
      for (long k = 0; k < dc; k++) {
        if (++coord[k] < ininfo.dim[k])
          break;
        coord[k] = 0;
      }
    }
  }

  mx::object_method(in, mx::_jit_sym_lock, savelock);

  if (!(is_f32 || is_f64))
    cerr << "matrix must be float32 or float64 - ignored" << endl;
  else if (!inbp)
    cerr << "matrix '" << std::string(name) << "' has no data - ignored" << endl;
  else if (ok)
    m_matrix_out.send("jit_matrix", m_outname);
}

MIN_EXTERNAL(gaussianize);
  