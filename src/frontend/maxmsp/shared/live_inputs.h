#pragma once
// Condition + matrix-noise inputs for the live objects (neural.live~,
// mc.neural.live~, mcs.neural.live~). The objects create the inlets; this header
// owns the per-load metadata, the control-thread state, the audio-thread
// snapshot, and the `<name>_inlet` toggle attributes.
//
// CONDITION INPUTS: a `condition`-role input is a held control vector the patch
// supplies as a `list` / bare number on the condition's control inlet (by
// position) or a `dictionary` (matched by name). It is held across blocks until
// updated; unsupplied conditions are zero-filled by the backend, and a supplied
// one is replicated across every batch. The control thread writes `cond` under
// `cond_mutex` and sets `cond_dirty`; the audio thread snapshots it into
// `supplied` (try_lock, reuse last on contention).
//
// MATRIX NOISE INPUTS: a `noise`-role input whose shape folds to a Jitter matrix
// gets its own jit_matrix inlet + a `<name>_inlet` boolean attribute. When the
// toggle is on and a valid matrix has arrived, perform feeds it (via `supplied`,
// matched by name) instead of the seeded per-input stream. Same threading as the
// conditions (noise_mutex / noise_dirty / noise_snapshot).
//
// THREADING: snapshot() must only run while the worker is NOT reading
// `supplied` — i.e. in the single-threaded perform branch, or right after the
// audio thread acquired the "result available" semaphore (the worker is then
// parked until the next "data available" release). build() / the attribute
// rebuild run only while the perform thread is quiesced (ctor / reload).
//
// The owning object class T must expose a public `LiveInputs m_inputs` member;
// the raw Max accessors for the `<name>_inlet` attributes recover the instance
// through the min wrapper.

#include "../../../backend/backend.h"
#include "c74_min.h"
#include "cond_bytes.h"   // cond_atoms_to_bytes
#include "cond_label.h"   // cond_inlet_label
#include "matrix_noise.h" // MatrixNoise, detect_matrix_noise, read_jit_matrix
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// ScalarType element size in bytes (Long is 8; Int / Float are 4). Turns a
// condition's byte buffer back into an element count for the backend.
inline size_t live_dtype_size(executorch::aten::ScalarType t) {
  return t == executorch::aten::ScalarType::Long ? 8 : 4;
}

struct LiveInputs {
  // --- conditions ---
  std::vector<std::string> cond_names; // condition names, declared order
  std::map<std::string, std::pair<executorch::aten::ScalarType, size_t>>
      cond_meta;                                 // name -> (dtype, per-batch numel)
  std::map<std::string, std::vector<char>> cond; // control thread (under mutex)
  std::mutex cond_mutex;
  std::atomic<bool> cond_dirty{false};
  std::map<std::string, std::vector<char>> cond_snapshot; // audio thread

  // --- matrix noise ---
  std::vector<MatrixNoise> noise;                         // foldable noise inputs
  std::vector<std::string> noise_attr_slots;              // <name>_inlet attrs
  std::map<std::string, std::vector<float>> noise_data;   // control thread
  std::map<std::string, bool> noise_enabled;              // <name>_inlet, by name
  std::mutex noise_mutex;
  std::atomic<bool> noise_dirty{false};
  std::map<std::string, std::vector<float>> noise_snapshot; // audio thread

  // -> Backend::perform (points into cond_snapshot / noise_snapshot)
  std::map<std::string, SuppliedInput> supplied;

  // Resolve the method's condition inputs (name, dtype, per-batch numel) and its
  // matrix-drivable noise inputs (folded geometry). Clears any held values.
  void build(Backend *model, const std::string &method, c74::min::logger &err) {
    cond_names.clear();
    cond_meta.clear();
    {
      std::lock_guard<std::mutex> lk(cond_mutex);
      cond.clear();
    }
    cond_snapshot.clear();
    cond_dirty.store(false);

    noise.clear();
    {
      std::lock_guard<std::mutex> lk(noise_mutex);
      noise_data.clear();
      noise_enabled.clear();
    }
    noise_snapshot.clear();
    noise_dirty.store(false);

    supplied.clear();
    if (!model || !model->has_metadata())
      return;

    const auto &inputs = model->method_inputs(method);
    for (const auto &in : inputs) {
      if (in.role != Role::Condition)
        continue;
      // Per-batch numel = product of shape[1:] (the leading dim is the batch the
      // backend replicates to n_batches).
      size_t per_batch = 1;
      for (size_t k = 1; k < in.shape.size(); k++)
        per_batch *= (size_t)in.shape[k];
      cond_names.push_back(in.name);
      cond_meta[in.name] = {in.dtype, per_batch};
    }
    noise = detect_matrix_noise(inputs, err);
  }

  // Assist labels for the condition inlets (one per condition, declared order).
  static std::vector<std::string> condition_labels(Backend *model,
                                                   const std::string &method) {
    std::vector<std::string> labels;
    if (!model || !model->has_metadata())
      return labels;
    for (const auto &in : model->method_inputs(method))
      if (in.role == Role::Condition)
        labels.push_back(cond_inlet_label(in) + " (list, or dictionary by name)");
    return labels;
  }

  // Assist labels for the matrix-noise inlets (one per entry of `noise`).
  std::vector<std::string> noise_labels() const {
    std::vector<std::string> labels;
    for (const auto &nz : noise)
      labels.push_back(nz.name + " noise (jit_matrix " + std::to_string(nz.n0) +
                       " float32/float64 " + std::to_string(nz.n2) + " " +
                       std::to_string(nz.n1) + "); enable @" + nz.name + "_inlet");
    return labels;
  }

  // Audio-thread side of the handoff (see THREADING above).
  void snapshot() {
    snapshot_conditions();
    snapshot_noise();
  }

  void snapshot_conditions() {
    if (!cond_dirty.load())
      return;
    if (!cond_mutex.try_lock())
      return;
    cond_snapshot = cond;
    cond_dirty.store(false);
    cond_mutex.unlock();
    // Set/refresh only the condition keys (held — never removed in steady
    // state), so the matrix-noise entries in `supplied` are left intact.
    for (auto &kv : cond_snapshot) {
      auto mit = cond_meta.find(kv.first);
      if (mit == cond_meta.end())
        continue;
      supplied[kv.first] = {mit->second.first, kv.second.data(),
                            kv.second.size() / live_dtype_size(mit->second.first)};
    }
  }

  void snapshot_noise() {
    if (!noise_dirty.load())
      return;
    if (!noise_mutex.try_lock())
      return;
    noise_snapshot = noise_data;
    std::map<std::string, bool> enabled = noise_enabled;
    noise_dirty.store(false);
    noise_mutex.unlock();
    for (const auto &nz : noise) {
      auto en = enabled.find(nz.name);
      auto sn = noise_snapshot.find(nz.name);
      if (en != enabled.end() && en->second && sn != noise_snapshot.end() &&
          sn->second.size() == nz.numel) {
        supplied[nz.name] = {executorch::aten::ScalarType::Float,
                             sn->second.data(), sn->second.size()};
      } else {
        supplied.erase(nz.name); // disabled / invalid => backend reseeds
      }
    }
  }

  // A bare list / number on a condition inlet: assign it to the condition at
  // this inlet's position. `base` is the index of the first condition inlet.
  void receive_list(const c74::min::atoms &args, int inlet, int base,
                    c74::min::logger &err) {
    const int j = inlet - base;
    if (j < 0 || j >= (int)cond_names.size()) {
      err << "list on inlet " << inlet << " has no condition input"
          << c74::min::endl;
      return;
    }
    const std::string &name = cond_names[j];
    auto mit = cond_meta.find(name);
    if (mit == cond_meta.end())
      return;
    if ((size_t)args.size() != mit->second.second)
      err << "condition '" << name << "' expects " << mit->second.second
          << " value(s), got " << args.size() << c74::min::endl;
    std::vector<char> bytes = cond_atoms_to_bytes(
        mit->second.first, (long)args.size(),
        [&](long k) { return (long long)args[k]; },
        [&](long k) { return (double)args[k]; });
    {
      std::lock_guard<std::mutex> lk(cond_mutex);
      cond[name] = std::move(bytes);
    }
    cond_dirty.store(true);
  }

  // A Max dictionary: store every entry whose key names a condition input
  // (merged, so several dicts accumulate). The dict may arrive as a direct
  // object atom or a registered name.
  void receive_dict(const c74::min::atoms &args, c74::min::logger &err) {
    if (args.empty()) {
      err << "dictionary message has no payload" << c74::min::endl;
      return;
    }
    c74::max::t_atom *a0 = (c74::max::t_atom *)&args[0];
    c74::max::t_dictionary *d = nullptr;
    bool retained = false;
    if (c74::max::atomisdictionary(a0)) {
      d = (c74::max::t_dictionary *)c74::max::atom_getobj(a0);
    } else {
      c74::min::symbol name = args[0];
      d = c74::max::dictobj_findregistered_retain((c74::max::t_symbol *)name);
      retained = true;
      if (!d) {
        err << "dictionary '" << std::string(name.c_str()) << "' not found"
            << c74::min::endl;
        return;
      }
    }
    if (!d)
      return;

    bool matched_any = false;
    for (const auto &name : cond_names) {
      auto mit = cond_meta.find(name);
      if (mit == cond_meta.end())
        continue;
      c74::max::t_symbol *key = c74::max::gensym(name.c_str());
      if (!c74::max::dictionary_hasentry(d, key))
        continue;
      long argc = 0;
      c74::max::t_atom *argv = nullptr;
      if (c74::max::dictionary_getatoms(d, key, &argc, &argv) !=
              c74::max::MAX_ERR_NONE ||
          !argv)
        continue;
      std::vector<char> bytes = cond_atoms_to_bytes(
          mit->second.first, argc,
          [&](long k) { return c74::max::atom_getlong(argv + k); },
          [&](long k) { return c74::max::atom_getfloat(argv + k); });
      {
        std::lock_guard<std::mutex> lk(cond_mutex);
        cond[name] = std::move(bytes);
      }
      matched_any = true;
    }
    if (retained)
      c74::max::dictobj_release(d);
    if (matched_any)
      cond_dirty.store(true);
    else
      err << "dictionary held no keys matching a condition input"
          << c74::min::endl;
  }

  // A Jitter matrix on a matrix-noise inlet: decode it into noise_data[name].
  // `base` is the index of the first matrix-noise inlet.
  void receive_matrix(const c74::min::atoms &args, int inlet, int base,
                      c74::min::logger &err) {
    const int j = inlet - base;
    if (j < 0 || j >= (int)noise.size()) {
      err << "jit_matrix arrived on inlet " << inlet
          << ", which has no matrix-noise input" << c74::min::endl;
      return;
    }
    const MatrixNoise &nz = noise[j];
    std::vector<float> buf;
    if (!read_jit_matrix(args, nz, buf, err))
      return; // shared reader logged the reason
    bool enabled;
    {
      std::lock_guard<std::mutex> lk(noise_mutex);
      noise_data[nz.name] = std::move(buf);
      auto en = noise_enabled.find(nz.name);
      enabled = (en != noise_enabled.end() && en->second);
    }
    noise_dirty.store(true);
    if (!enabled)
      err << "noise '" << nz.name << "' received but not enabled; set @"
          << nz.name << "_inlet 1 to use it (seeded until then)"
          << c74::min::endl;
  }
};

// Strip the fixed "_inlet" suffix to recover the noise input name an attribute
// names; returns false if `attr_name` is not a `<name>_inlet`.
inline bool live_noise_name_of(const char *attr_name, std::string &out) {
  std::string s(attr_name);
  static const std::string suf = "_inlet";
  if (s.size() <= suf.size() || s.compare(s.size() - suf.size(), suf.size(), suf))
    return false;
  out = s.substr(0, s.size() - suf.size());
  return true;
}

template <class T>
c74::max::t_max_err live_noise_inlet_set(c74::max::t_object *x,
                                         c74::max::t_object *attr, long ac,
                                         c74::max::t_atom *av) {
  if (!x || ac < 1 || !av)
    return 0;
  LiveInputs &in = reinterpret_cast<c74::min::minwrap<T> *>(x)->m_min_object.m_inputs;
  auto *nm = (c74::max::t_symbol *)c74::max::object_method(
      attr, c74::max::gensym("getname"));
  std::string name;
  if (nm && live_noise_name_of(nm->s_name, name)) {
    const bool on = c74::max::atom_getlong(av) != 0;
    {
      std::lock_guard<std::mutex> lk(in.noise_mutex);
      in.noise_enabled[name] = on;
    }
    in.noise_dirty.store(true); // re-snapshot so perform picks up the toggle
  }
  return 0;
}

template <class T>
c74::max::t_max_err live_noise_inlet_get(c74::max::t_object *x,
                                         c74::max::t_object *attr, long *ac,
                                         c74::max::t_atom **av) {
  LiveInputs &in = reinterpret_cast<c74::min::minwrap<T> *>(x)->m_min_object.m_inputs;
  auto *nm = (c74::max::t_symbol *)c74::max::object_method(
      attr, c74::max::gensym("getname"));
  long v = 0;
  std::string name;
  if (nm && live_noise_name_of(nm->s_name, name)) {
    std::lock_guard<std::mutex> lk(in.noise_mutex);
    auto it = in.noise_enabled.find(name);
    v = (it != in.noise_enabled.end() && it->second) ? 1 : 0;
  }
  if (*ac < 1 || !*av) {
    if (*av)
      c74::max::sysmem_freeptr(*av);
    *ac = 1;
    *av = (c74::max::t_atom *)c74::max::sysmem_newptr(sizeof(c74::max::t_atom));
  }
  c74::max::atom_setlong(*av, v);
  return 0;
}

// Re-create the `<name>_inlet` boolean attributes from the current matrix-noise
// inputs (call after LiveInputs::build, with the perform thread quiesced).
template <class T>
void rebuild_noise_inlet_attrs(c74::max::t_object *self, LiveInputs &in) {
  for (const auto &n : in.noise_attr_slots) // drop a previous model's toggles
    c74::max::object_deleteattr(self, c74::max::gensym(n.c_str()));
  in.noise_attr_slots.clear();
  for (const auto &nz : in.noise) {
    {
      std::lock_guard<std::mutex> lk(in.noise_mutex);
      in.noise_enabled[nz.name] = false;
    }
    // Custom thread-safe accessors so the audio thread can read the toggle
    // (via noise_enabled) every block.
    std::string an = make_noise_inlet_attr(
        self, nz, (c74::max::method)&live_noise_inlet_get<T>,
        (c74::max::method)&live_noise_inlet_set<T>);
    if (!an.empty())
      in.noise_attr_slots.push_back(an);
  }
}
