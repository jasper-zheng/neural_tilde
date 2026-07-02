#include "../../../backend/backend.h"
#include "../shared/circular_buffer.h"
#include "../shared/cond_bytes.h"
#include "../shared/cond_label.h"   // cond_inlet_label
#include "../shared/matrix_noise.h" // MatrixNoise, detect_matrix_noise, read_jit_matrix
#include "../shared/neural_common.h"
#include "c74_min.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

using namespace c74::min;

// ScalarType element size in bytes (Long is 8; Int / Float are 4). Used to turn
// a condition's byte buffer back into an element count for the backend.
static size_t live_dtype_size(executorch::aten::ScalarType t) {
  return t == executorch::aten::ScalarType::Long ? 8 : 4;
}

class live : public object<live>, public vector_operator<> {
public:
  MIN_DIGEST{"Real-time (live) neural audio synthesis"};
  MIN_DESCRIPTION{"Load and run a neural audio synthesis model for real-time signal inputs. "
                  "Supports models exported from PyTorch via ExecuTorch. "
                  "Hardware-accelerated on CPU / GPU / ANE (Apple Neural Engine)."};
  MIN_TAGS{"neural audio synthesis, generative models, diffusion, autoencoders"};
  MIN_AUTHOR{"Jasper Shuoyang Zheng"};
  MIN_RELATED{"mc.neural.live~, mcs.neural.live~, neural.tokenizer~, neural.gaussianize"};

  live(const atoms &args = {});
  ~live();

  // INLETS OUTLETS
  std::vector<std::unique_ptr<inlet<>>> m_inlets;
  std::vector<std::unique_ptr<outlet<>>> m_outlets;

  // BACKEND RELATED MEMBERS
  std::unique_ptr<Backend> m_model;
  bool m_is_backend_init = false;
  std::string m_method;
  // Absolute path to the model's .pte program (derived from its .json metadata via
  // resolve_model_pte; the .pte itself may be absent — see backend load()).
  std::string m_path;
  int m_in_dim, m_in_ratio, m_out_dim, m_out_ratio, m_higher_ratio;

  // DYNAMIC ATTRIBUTES --------------------------------------------------------
  // One Max instance attribute per settable scalar the model's method declares,
  // created from the metadata at load (the same design as neural.gen~). The model
  // consumes each as a trailing scalar input to its forward; the host owns the
  // value. m_attr_slots holds the metadata (name/range/default/description);
  // m_attr_live is the parallel live value, written by the attribute setter
  // (control thread) and snapshotted by perform (audio/worker thread).
  struct AttrSlot {
    std::string name;
    double min, max, def;
    std::string description;
  };
  std::vector<AttrSlot> m_attr_slots;
  std::vector<float> m_attr_live;
  void add_dynamic_attributes();
  bool is_attr(const std::string &name) const;
  void set_attr(const std::string &name, double value);
  void attr_store(const char *name, double value); // setter -> m_attr_live
  double attr_load(const char *name) const;        // getter <- m_attr_live
  // perform feeds m_attr_live to the model directly (by const-ref) in slot order;
  // the per-float race with the setter is benign (a value-copy was never atomic).
  // Raw Max attribute accessors for the dynamically-created attributes (recover
  // the instance via the Min wrapper; identify the attribute by its name).
  static c74::max::t_max_err attr_get(c74::max::t_object *x,
                                      c74::max::t_object *attr, long *ac,
                                      c74::max::t_atom **av);
  static c74::max::t_max_err attr_set(c74::max::t_object *x,
                                      c74::max::t_object *attr, long ac,
                                      c74::max::t_atom *av);

  // CONDITION INPUTS --------------------------------------------------------
  // A `condition`-role input is a held control vector the patch supplies as a
  // `list` on the condition's control inlet (by position) or a `dictionary`
  // (matched by name). It is held across blocks until updated; unsupplied
  // conditions are zero-filled by the backend. The control thread writes
  // m_cond under m_cond_mutex and sets m_cond_dirty; the audio thread snapshots
  // it into m_supplied (try_lock, reuse last on contention) and hands it to
  // perform — the snapshot/perform handoff is ordered by the worker semaphores,
  // so the worker reads a stable m_supplied while the next block waits.
  std::vector<std::string> m_cond_names;           // condition names, declared order
  std::map<std::string, std::pair<executorch::aten::ScalarType, size_t>>
      m_cond_meta;                                 // name -> (dtype, per-batch numel)
  std::map<std::string, std::vector<char>> m_cond; // control thread (under mutex)
  std::mutex m_cond_mutex;
  std::atomic<bool> m_cond_dirty{false};
  std::map<std::string, std::vector<char>> m_cond_snapshot; // audio thread
  std::map<std::string, SuppliedInput> m_supplied;          // -> perform()
  void build_condition_meta();
  void snapshot_conditions();
  void receive_list(const atoms &args, int inlet);
  void receive_dict(const atoms &args);

  // MATRIX NOISE INPUTS ------------------------------------------------------
  // A `noise`-role input whose shape folds to a Jitter matrix gets its own
  // jit_matrix inlet (after the condition inlets) + a `<name>_inlet` boolean
  // attribute. When the toggle is on and a valid matrix has arrived, perform
  // feeds it (via the shared m_supplied map, matched by name) instead of the
  // seeded per-input stream. Threading mirrors the conditions: the scheduler
  // thread writes m_noise / m_noise_enabled under m_noise_mutex and sets
  // m_noise_dirty; the audio thread snapshots into m_noise_snapshot (which
  // m_supplied points into) at the worker handoff. The geometry + matrix read
  // live in shared/matrix_noise.h (also used by neural.gen~).
  std::vector<MatrixNoise> m_matrix_noise;       // foldable noise inputs (per load)
  std::map<int, std::string> m_noise_inlet_name; // absolute inlet idx -> noise name
  std::vector<std::string> m_noise_attr_slots;   // <name>_inlet attrs (for cleanup)
  std::map<std::string, std::vector<float>> m_noise;   // control thread (under mutex)
  std::map<std::string, bool> m_noise_enabled;        // <name>_inlet toggle, by name
  std::mutex m_noise_mutex;
  std::atomic<bool> m_noise_dirty{false};
  std::map<std::string, std::vector<float>> m_noise_snapshot; // audio thread
  void build_matrix_noise_meta();
  void snapshot_noise();
  void receive_matrix(const atoms &args, int inlet);
  void add_noise_inlet_attributes();
  // Raw accessors for the `<name>_inlet` boolean attributes (strip the suffix
  // to key m_noise_enabled by the noise input name).
  static c74::max::t_max_err noise_inlet_get(c74::max::t_object *x,
                                             c74::max::t_object *attr, long *ac,
                                             c74::max::t_atom **av);
  static c74::max::t_max_err noise_inlet_set(c74::max::t_object *x,
                                             c74::max::t_object *attr, long ac,
                                             c74::max::t_atom *av);

  // BUFFER RELATED MEMBERS
  int m_buffer_size;
  std::unique_ptr<circular_buffer<double, float>[]> m_in_buffer;
  std::unique_ptr<circular_buffer<float, double>[]> m_out_buffer;
  std::vector<std::unique_ptr<float[]>> m_in_model, m_out_model;

  // AUDIO PERFORM
  bool m_use_thread, m_should_stop_perform_thread;
  std::unique_ptr<std::thread> m_compute_thread;
  std::binary_semaphore m_data_available_lock, m_result_available_lock;

  void operator()(audio_bundle input, audio_bundle output);
  void perform(audio_bundle input, audio_bundle output);

  // ONLY FOR DOCUMENTATION
  argument<symbol> path_arg{this, "model path",
                            "Path to the .pte model (with its .json metadata beside it)."};
  argument<symbol> method_arg{this, "method",
                              "Name of the method to call during synthesis."};
  argument<int> buffer_arg{
      this, "buffer size",
      "Size of the internal buffer (can't be lower than the method's ratio)."};

  // ENABLE / DISABLE ATTRIBUTE
    attribute<bool> enable {this, "enable_model", false,
        description{"Enable / disable tensor computation"},
        setter{
            [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
                if (m_is_backend_init && m_model->is_loaded() && args[0]){
                    return {true};
                } else {
                    return {false};
                }
                return args;
            }
        }
    };

    // RNG SEED ATTRIBUTE
    // Base seed for the model's noise inputs (each noise input draws from its own
    // (seed, name) stream; see Backend::set_seed). The metadata `seed` is the
    // default, synced onto this attribute after load. Changing it restarts the
    // seeded noise deterministically.
    attribute<int> seed {this, "seed", 0,
        description{"RNG seed for noise inputs (reproducible noise)."},
        setter{
            [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
                if (m_model)
                    m_model->set_seed((int)args[0]);
                return args;
            }
        }
    };

    static void model_perform_loop(live *nn_instance);
    
    message<> reload_m {this, "reload", "Reload a model",
        MIN_FUNCTION {
            
            if (args.size() != 1) {
                cerr << "wrong number of argments" << endl;
                return {};
            }
            if (args.size() == 1) {
                if (args[0].a_type == 3){
                    m_should_stop_perform_thread = true;
                    if (m_compute_thread)
                      m_compute_thread->join();
                    
                    try {
//                        m_model.reset();
                        m_model = std::make_unique<Backend>();
                    } catch (...) {
                        }
                    // Resolve the metadata and derive the sibling .pte (which may
                    // be absent — load() then keeps metadata-only, model disabled).
                    m_path = resolve_model_pte(std::string(args[0]));
                    if (m_path.empty()) {
                        cerr << "could not find model .json metadata for "
                             << std::string(args[0]) << endl;
                        error();
                        return {};
                    }
                    try {
                        if (m_model->load(m_path)) {
                            cerr << "error during loading" << endl;
                            error();
                            return {};
                        }
                    }
                        catch (...) {
                        }
                    if (m_model->has_metadata() && !m_model->is_loaded()) {
                        cerr << "no .pte program is loaded, inference is disabled"
                             << endl;
                    }
                    
                    try {
                        m_higher_ratio = m_model->get_higher_ratio();
                    } catch (...) {
                        
                    }
                    // GET MODEL'S METHOD PARAMETERS
                    auto params = m_model->get_method_params(m_method);
                    if (!params.size()) {
                      error("method " + m_method + " not found !");
                    }
                    // Rebuild the model's settable scalars as Max attributes (the
                    // perform thread is stopped above, so this is safe).
                    add_dynamic_attributes();
                    // Re-resolve the condition inputs for the new model (the
                    // condition inlets, fixed at construction, do not change).
                    build_condition_meta();
                    // Re-resolve the matrix-noise inputs + their toggle attributes
                    // (the jit_matrix inlets, fixed at construction, do not change);
                    // re-sync the seed attribute onto the new metadata default.
                    build_matrix_noise_meta();
                    add_noise_inlet_attributes();
                    seed = m_model->gen_seed();
                    // Adopt the model's exported (fixed) buffer size; see ctor.
                    m_buffer_size = negotiate_buffer_size(
                        m_buffer_size, m_model->get_buffer_size(), m_higher_ratio,
                        m_use_thread, cerr);
                    m_should_stop_perform_thread = false;
                    if (m_use_thread)
                      m_compute_thread = std::make_unique<std::thread>(model_perform_loop, this);
                } else {
                    cerr << "argument should be a path" << endl;
                    return {};
                }
            }
            cout << "neural.live~ setup finished" << endl;
            return {};
        }
    };
    
    

  // BOOT STAMP
  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        cout << "neural.live~ " << VERSION << " - " << NN_ENGINE_NAME
             << " - July-2026 - Jasper Shuoyang Zheng " << endl;
        cout << "docs: https://github.com/jasper-zheng/neural_tilde" << endl;
        return {};
      }};

  // Conditioning as a bare list on a condition inlet: assigned to the condition
  // input at THIS inlet's position (the alternative to the by-name dictionary
  // path). `inlet` is the arriving inlet index (see receive_list).
  message<> list_msg{
      this, "list", "Conditioning values for this inlet's condition input.",
      MIN_FUNCTION {
        receive_list(args, inlet);
        return {};
      }};

  // Conditioning as a bare float/int on a condition inlet: treated as a
  // single-element list for the condition input at THIS inlet's position. min-api
  // maps "number" -> the "float" method and auto-generates the matching "int"
  // method routing to it, so both bare floats and bare ints land here (see
  // receive_list; `inlet` is the arriving inlet index).
  message<> number_msg{
      this, "number",
      "A single conditioning value for this inlet's condition input.",
      MIN_FUNCTION {
        receive_list(args, inlet);
        return {};
      }};

  // Conditioning as a Max dictionary, matched BY KEY NAME against the model's
  // condition inputs and MERGED (so several dicts on any inlet accumulate).
  message<> dictionary_msg{
      this, "dictionary",
      "Receive conditioning as a dictionary; merged by key name.", MIN_FUNCTION {
        receive_dict(args);
        return {};
      }};

  // A Jitter matrix on a matrix-noise inlet: snapshot the named float32/float64
  // matrix as this inlet's noise input (used when its `<name>_inlet` toggle is on). `inlet`
  // is the arriving inlet index (see receive_matrix).
  message<> jit_matrix_msg{
      this, "jit_matrix",
      "Float32 Jitter matrix used as this inlet's noise input "
      "(enable with attribute: '[name]_inlet').",
      [this](const atoms &args, const int inlet) -> atoms {
        receive_matrix(args, inlet);
        return {};
      }};

  message<> anything{
      this, "anything", "callback for attributes",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        // reload/get_methods are model-level controls. The model's settable
        // scalars are dynamic Max attributes (see add_dynamic_attributes) added
        // per-instance AFTER class registration, so Max's message dispatch
        // doesn't route to them natively (the inspector / @box-args reach them
        // directly) — this catch-all backs the bare `<name> <value>` form.
        std::string sel = args[0];
        if (sel == "reload") {
          m_model->reload();
        } else if (sel == "get_methods") {
          for (std::string method : m_model->get_available_methods())
            cout << method << endl;
        } else if (is_attr(sel) && args.size() >= 2) {
          set_attr(sel, (double)args[1]); // bare `<name> <value>`
        } else {
          cerr << "no method/attribute '" << sel << "'" << endl;
        }
        return {};
      }};
};

void model_perform(live *nn_instance) {
  std::vector<float *> in_model, out_model;
  for (int c(0); c < nn_instance->m_in_dim; c++)
    in_model.push_back(nn_instance->m_in_model[c].get());
  for (int c(0); c < nn_instance->m_out_dim; c++)
    out_model.push_back(nn_instance->m_out_model[c].get());

  nn_instance->m_model->perform(in_model, out_model, nn_instance->m_buffer_size,
                                nn_instance->m_method, 1,
                                nn_instance->m_attr_live, nn_instance->m_supplied);
}

void live::model_perform_loop(live *nn_instance) {
  std::vector<float *> in_model, out_model;

  for (auto &ptr : nn_instance->m_in_model)
    in_model.push_back(ptr.get());

  for (auto &ptr : nn_instance->m_out_model)
    out_model.push_back(ptr.get());

  while (!nn_instance->m_should_stop_perform_thread) {
    if (nn_instance->m_data_available_lock.try_acquire_for(
            std::chrono::milliseconds(200))) {
      nn_instance->m_model->perform(in_model, out_model,
                                    nn_instance->m_buffer_size,
                                    nn_instance->m_method, 1,
                                    nn_instance->m_attr_live,
                                    nn_instance->m_supplied);
      nn_instance->m_result_available_lock.release();
    }
  }
}

live::live(const atoms &args)
    : m_compute_thread(nullptr), m_in_dim(1), m_in_ratio(1), m_out_dim(1),
      m_out_ratio(1), m_buffer_size(4096), m_method("forward"),
      m_use_thread(true), m_data_available_lock(0), m_result_available_lock(1),
      m_should_stop_perform_thread(false) {

  m_model = std::make_unique<Backend>();
  m_is_backend_init = true;

  // CHECK ARGUMENTS
  if (!args.size()) {
    return;
  }
  if (args.size() > 0) { // ONE ARGUMENT IS GIVEN
    // Resolve the mandatory .json metadata and derive the sibling .pte path (which
    // may be absent — the backend then loads metadata-only and stays disabled).
    m_path = resolve_model_pte(std::string(args[0]));
    if (m_path.empty()) {
      cerr << "could not find model .json metadata for " << std::string(args[0]) << endl;
      error();
      return;
    }
  }
  if (args.size() > 1) { // TWO ARGUMENTS ARE GIVEN
    m_method = std::string(args[1]);
  }
  if (args.size() > 2) { // THREE ARGUMENTS ARE GIVEN
    m_buffer_size = int(args[2]);
  }

  // TRY TO LOAD MODEL (returns non-zero only if the metadata itself can't parse).
  if (m_model->load(m_path)) {
    cerr << "error during loading" << endl;
    error();
    return;
  }

  // The metadata parsed but the .pte program may be missing: build the I/O below
  // from metadata, but the model stays disabled (enable_model can't turn on).
  if (m_model->has_metadata() && !m_model->is_loaded()) {
      cerr << "no .pte program is loaded, inference is disabled"
         << endl;
  }

  m_higher_ratio = m_model->get_higher_ratio();

  // GET MODEL'S METHOD PARAMETERS
  auto params = m_model->get_method_params(m_method);

  if (!params.size()) {
    error("method " + m_method + " not found !");
  }

  m_in_dim = params[0];
  m_in_ratio = params[1];
  m_out_dim = params[2];
  m_out_ratio = params[3];

  // Resolve the model's condition-role inputs (their names/dtypes/sizes) so the
  // ctor can give each its own control inlet and perform() can bind them.
  build_condition_meta();
  // Resolve the matrix-drivable noise inputs (geometry per load) so the ctor can
  // give each its own jit_matrix inlet + `<name>_inlet` toggle.
  build_matrix_noise_meta();

  // ExecuTorch programs are exported at a fixed buffer size (the time dim is
  // baked into the .pte unless dynamic); the host block size must match it.
  // Adopt the model's exported buffer_size, overriding the user's buffer arg.
  m_buffer_size = negotiate_buffer_size(m_buffer_size, m_model->get_buffer_size(),
                                        m_higher_ratio, m_use_thread, cerr);

  // CREATE INLETS, OUTLETS and BUFFERS
  m_in_buffer = std::make_unique<circular_buffer<double, float>[]>(m_in_dim);
  auto input_labels = m_model->get_input_labels(m_method);
  for (int i(0); i < m_in_dim; i++) {
    std::string input_label =
        (i < (int)input_labels.size())
            ? input_labels[i]
            : "(signal) model input " + std::to_string(i);
    m_inlets.push_back(std::make_unique<inlet<>>(this, input_label, "signal"));
    m_in_buffer[i].initialize(m_buffer_size);
    m_in_model.push_back(std::make_unique<float[]>(m_buffer_size));
  }

  // One control inlet per condition-role input, after the signal inlets (fixed
  // at construction; a later reload to a different condition set keeps these,
  // while the by-name dictionary path still reaches every condition). Label each
  // with the tensor it expects (name + shape + dtype + optional description),
  // built from the same method_inputs() the condition order derives from.
  for (const auto &in : m_model->method_inputs(m_method)) {
    if (in.role != Role::Condition)
      continue;
    m_inlets.push_back(std::make_unique<inlet<>>(
        this, cond_inlet_label(in) + " (list, or dictionary by name)"));
  }

  // One jit_matrix inlet per matrix-drivable noise input, after the condition
  // inlets (also fixed at construction; the `<name>_inlet` toggle picks the
  // matrix over the seeded stream per block — see receive_matrix / snapshot_noise).
  {
    int base = (int)m_inlets.size();
    for (size_t i = 0; i < m_matrix_noise.size(); i++) {
      const auto &nz = m_matrix_noise[i];
      std::string label = nz.name + " noise (jit_matrix " +
                          std::to_string(nz.n0) + " float32/float64 " +
                          std::to_string(nz.n2) + " " + std::to_string(nz.n1) +
                          "); enable @" + nz.name + "_inlet";
      m_inlets.push_back(std::make_unique<inlet<>>(this, label));
      m_noise_inlet_name[base + (int)i] = nz.name;
    }
  }

  m_out_buffer = std::make_unique<circular_buffer<float, double>[]>(m_out_dim);
  auto output_labels = m_model->get_output_labels(m_method);
  for (int i(0); i < m_out_dim; i++) {
    std::string output_label =
        (i < (int)output_labels.size())
            ? output_labels[i]
            : "(signal) model output " + std::to_string(i);
    m_outlets.push_back(
        std::make_unique<outlet<>>(this, output_label, "signal"));
    m_out_buffer[i].initialize(m_buffer_size);
    m_out_model.push_back(std::make_unique<float[]>(m_buffer_size));
  }

  // Expose the model's settable scalars as dynamic Max attributes (inspector /
  // @box-args / get-set). Done before the perform thread starts.
  add_dynamic_attributes();
  // One `<name>_inlet` boolean attribute per matrix-noise input (toggles the
  // matrix inlet over the seeded stream).
  add_noise_inlet_attributes();
  // Adopt the metadata's default seed onto the `seed` attribute (fires set_seed).
  seed = m_model->gen_seed();

  if (m_use_thread)
    m_compute_thread = std::make_unique<std::thread>(model_perform_loop, this);
}

live::~live() {
  m_should_stop_perform_thread = true;
  if (m_compute_thread)
    m_compute_thread->join();
}

// --- Dynamic attributes -----------------------------------------------------
// The model's settable scalars become Max instance attributes (the same design
// as neural.gen_tilde.cpp's add_dynamic_attributes), each fed to the model as a
// trailing forward input. The custom getter/setter make a thread-shared
// m_attr_live the single source of truth so the audio/worker perform thread can
// read the current value without touching the Max API.

bool live::is_attr(const std::string &name) const {
  for (const auto &s : m_attr_slots)
    if (s.name == name)
      return true;
  return false;
}

void live::set_attr(const std::string &name, double value) {
  if (!is_attr(name)) {
    cerr << "no attribute '" << name << "'" << endl;
    return;
  }
  // Max applies the clip filter, then calls attr_set (-> attr_store).
  c74::max::object_attr_setfloat((c74::max::t_object *)(*this),
                                 c74::max::gensym(name.c_str()), value);
}

// Setter side: store the (clamped) value into the parallel live buffer.
void live::attr_store(const char *name, double value) {
  for (size_t i = 0; i < m_attr_slots.size(); i++)
    if (m_attr_slots[i].name == name) {
      const AttrSlot &s = m_attr_slots[i];
      if (s.max > s.min)
        value = std::min(std::max(value, s.min), s.max); // defensive clamp
      m_attr_live[i] = (float)value;
      return;
    }
}

double live::attr_load(const char *name) const {
  for (size_t i = 0; i < m_attr_slots.size(); i++)
    if (m_attr_slots[i].name == name)
      return (double)m_attr_live[i];
  return 0.0;
}

c74::max::t_max_err live::attr_set(c74::max::t_object *x, c74::max::t_object *attr,
                                 long ac, c74::max::t_atom *av) {
  if (!x || ac < 1 || !av)
    return 0;
  live &self = reinterpret_cast<c74::min::minwrap<live> *>(x)->m_min_object;
  auto *nm = (c74::max::t_symbol *)c74::max::object_method(
      attr, c74::max::gensym("getname"));
  if (nm)
    self.attr_store(nm->s_name, c74::max::atom_getfloat(av));
  return 0;
}

c74::max::t_max_err live::attr_get(c74::max::t_object *x, c74::max::t_object *attr,
                                 long *ac, c74::max::t_atom **av) {
  live &self = reinterpret_cast<c74::min::minwrap<live> *>(x)->m_min_object;
  auto *nm = (c74::max::t_symbol *)c74::max::object_method(
      attr, c74::max::gensym("getname"));
  double v = nm ? self.attr_load(nm->s_name) : 0.0;
  // Reuse the caller's atom if one was provided, else allocate (Max frees it).
  if (*ac < 1 || !*av) {
    if (*av)
      c74::max::sysmem_freeptr(*av);
    *ac = 1;
    *av = (c74::max::t_atom *)c74::max::sysmem_newptr(sizeof(c74::max::t_atom));
  }
  c74::max::atom_setfloat(*av, v);
  return 0;
}

// Re-derive the dynamic attributes from the loaded model's method. Called from
// the ctor and on reload (the perform thread is quiesced at both points).
void live::add_dynamic_attributes() {
  c74::max::t_object *self = (c74::max::t_object *)(*this);
  for (const auto &s : m_attr_slots) // drop a previous model's attributes
    c74::max::object_deleteattr(self, c74::max::gensym(s.name.c_str()));
  m_attr_slots.clear();
  m_attr_live.clear();
  if (!m_model || !m_model->has_metadata())
    return;

  // Populate slots + live values first so the setter (fired by the default-set
  // below, and by @box-args after the ctor) can resolve each slot by name.
  for (const auto &a : m_model->method_attributes(m_method)) {
    m_attr_slots.push_back({a.name, a.min, a.max, a.def, a.description});
    m_attr_live.push_back((float)a.def);
  }

  for (const auto &a : m_model->method_attributes(m_method)) {
    c74::max::t_object *attr = (c74::max::t_object *)c74::max::attribute_new(
        a.name.c_str(), c74::max::gensym("float64"), 0,
        (c74::max::method)&live::attr_get, (c74::max::method)&live::attr_set);
    if (!attr)
      continue;
    if (a.max > a.min)
      c74::max::attr_addfilter_clip(attr, a.min, a.max, 1, 1);
    c74::max::object_addattr(self, attr);
    c74::max::object_attr_setfloat(self, c74::max::gensym(a.name.c_str()), a.def);
    if (!a.description.empty())
      c74::max::object_attr_addattr_format(
          self, a.name.c_str(), "label", c74::max::gensym("symbol"), 0, "s",
          c74::max::gensym(a.description.c_str()));
  }
}

void live::operator()(audio_bundle input, audio_bundle output) {
  auto dsp_vec_size = output.frame_count();

  // CHECK IF MODEL IS LOADED AND ENABLED
  if (!m_model->is_loaded() || !enable) {
    fill_with_zero(output);
    return;
  }

  // CHECK IF DSP_VEC_SIZE IS LARGER THAN BUFFER SIZE
  if (dsp_vec_size > m_buffer_size) {
    cerr << "vector size (" << dsp_vec_size << ") ";
    cerr << "larger than buffer size (" << m_buffer_size << "). ";
    cerr << "disabling model.";
    cerr << endl;
    enable = false;
    fill_with_zero(output);
    return;
  }

  perform(input, output);
}

void live::perform(audio_bundle input, audio_bundle output) {
  auto vec_size = input.frame_count();

  // COPY INPUT TO CIRCULAR BUFFER
  // Min's vector_operator turns EVERY inlet into an MSP signal channel
  // (dsp_setup is called with inlets().size(), and the perform wrapper sets
  // audio_bundle.channel_count = numins), so input.channel_count() also counts
  // the condition / matrix-noise control inlets that follow the signal inlets.
  // Only the first m_in_dim channels are the model's audio, and m_in_buffer is
  // sized to m_in_dim — clamp so the extra control inlets don't run us off the
  // end of m_in_buffer (an out-of-bounds circular_buffer whose null storage
  // pointer segfaults the moment the model is enabled with a condition input).
  const long n_in = std::min<long>(input.channel_count(), (long)m_in_dim);
  for (int c(0); c < n_in; c++) {
    auto in = input.samples(c);
    m_in_buffer[c].put(in, vec_size);
  }

  if (m_in_buffer[0].full()) { // BUFFER IS FULL
    if (!m_use_thread) {
      // Single-threaded: this same thread runs inference below, so refreshing the
      // held conditioning + matrix noise here cannot race the model's read of
      // m_supplied. Both write into m_supplied, which model_perform then reads.
      snapshot_conditions();
      snapshot_noise();

      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_in_dim; c++)
        m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);

      // CALL MODEL PERFORM IN CURRENT THREAD
      model_perform(this);

      // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_out_dim; c++)
        m_out_buffer[c].put(m_out_model[c].get(), m_buffer_size);

    } else if (m_result_available_lock.try_acquire()) {
      // Acquiring m_result_available_lock means the worker finished the previous
      // block and is now parked on m_data_available_lock — i.e. it is NOT reading
      // m_supplied. Refreshing the held conditioning + matrix noise is therefore
      // safe ONLY here: the worker won't touch m_supplied again until we release
      // m_data_available_lock below. (Doing this before the try_acquire — as the
      // code previously did — lets the audio thread mutate the m_supplied map
      // while a busy worker is traversing it in Backend::perform, corrupting the
      // tree. That crash only surfaced once a condition/noise input populated
      // m_supplied; with an empty map there were no nodes to race over.)
      snapshot_conditions();
      snapshot_noise();

      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_in_dim; c++)
        m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);

      // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_out_dim; c++)
        m_out_buffer[c].put(m_out_model[c].get(), m_buffer_size);

      // SIGNAL PERFORM THREAD THAT DATA IS AVAILABLE
      m_data_available_lock.release();
    }
  }

  // COPY CIRCULAR BUFFER TO OUTPUT (clamped to the model's signal outlets;
  // defensive — conditions add no outlets today, so numouts == m_out_dim, but
  // this keeps perform correct if the two ever diverge).
  const long n_out = std::min<long>(output.channel_count(), (long)m_out_dim);
  for (int c(0); c < n_out; c++) {
    auto out = output.samples(c);
    m_out_buffer[c].get(out, vec_size);
  }
}

// --- Conditioning -----------------------------------------------------------
// Resolve the loaded method's condition-role inputs: their names (in declared
// order, mapping a control inlet position) and per-input (dtype, per-batch
// numel). Called from the ctor and on reload (the perform thread is quiesced at
// both points). Clears any previously cached conditioning.
void live::build_condition_meta() {
  m_cond_names.clear();
  m_cond_meta.clear();
  {
    std::lock_guard<std::mutex> lk(m_cond_mutex);
    m_cond.clear();
  }
  m_cond_snapshot.clear();
  m_supplied.clear();
  m_cond_dirty.store(false);
  if (!m_model || !m_model->has_metadata())
    return;
  for (const auto &in : m_model->method_inputs(m_method)) {
    if (in.role != Role::Condition)
      continue;
    // Per-batch numel = product of shape[1:] (the leading dim is the batch the
    // host replicates to n_batches; matches Backend::perform's expectation).
    size_t per_batch = 1;
    for (size_t k = 1; k < in.shape.size(); k++)
      per_batch *= (size_t)in.shape[k];
    m_cond_names.push_back(in.name);
    m_cond_meta[in.name] = {in.dtype, per_batch};
  }
}

// Snapshot the held conditioning into the map perform() reads. Audio-thread
// side of the handoff: only rebuilds when the control thread flagged a change,
// and only if it can grab the lock without blocking (reuse the last snapshot on
// contention). m_supplied points into m_cond_snapshot, which is rewritten only
// here — and only while the worker is idle — so the worker reads it safely.
void live::snapshot_conditions() {
  if (!m_cond_dirty.load())
    return;
  if (!m_cond_mutex.try_lock())
    return;
  m_cond_snapshot = m_cond;
  m_cond_dirty.store(false);
  m_cond_mutex.unlock();
  // Set/refresh only the condition keys (held — never removed in steady state),
  // so the matrix-noise entries in m_supplied are left intact (snapshot_noise
  // owns those). build_condition_meta clears m_supplied wholesale on (re)load.
  for (auto &kv : m_cond_snapshot) {
    auto mit = m_cond_meta.find(kv.first);
    if (mit == m_cond_meta.end())
      continue;
    m_supplied[kv.first] = {mit->second.first, kv.second.data(),
                            kv.second.size() / live_dtype_size(mit->second.first)};
  }
}

// A bare list on a condition's control inlet: assign it to the condition at this
// inlet's position (the condition inlets follow the m_in_dim signal inlets).
void live::receive_list(const atoms &args, int inlet) {
  const int j = inlet - m_in_dim;
  if (j < 0 || j >= (int)m_cond_names.size()) {
    cerr << "list on inlet " << inlet << " has no condition input" << endl;
    return;
  }
  const std::string &name = m_cond_names[j];
  auto mit = m_cond_meta.find(name);
  if (mit == m_cond_meta.end())
    return;
  if ((size_t)args.size() != mit->second.second)
    cerr << "condition '" << name << "' expects " << mit->second.second
         << " value(s), got " << args.size() << endl;
  std::vector<char> bytes = cond_atoms_to_bytes(
      mit->second.first, (long)args.size(),
      [&](long k) { return (long long)args[k]; },
      [&](long k) { return (double)args[k]; });
  {
    std::lock_guard<std::mutex> lk(m_cond_mutex);
    m_cond[name] = std::move(bytes);
  }
  m_cond_dirty.store(true);
}

// A Max dictionary: store every entry whose key names a condition input (merged,
// so several dicts accumulate). The dict may arrive as a direct object atom or a
// registered name. Mirrors neural.gen~'s by-name path.
void live::receive_dict(const atoms &args) {
  if (args.empty()) {
    cerr << "dictionary message has no payload" << endl;
    return;
  }
  c74::max::t_atom *a0 = (c74::max::t_atom *)&args[0];
  c74::max::t_dictionary *d = nullptr;
  bool retained = false;
  if (c74::max::atomisdictionary(a0)) {
    d = (c74::max::t_dictionary *)c74::max::atom_getobj(a0);
  } else {
    symbol name = args[0];
    d = c74::max::dictobj_findregistered_retain((c74::max::t_symbol *)name);
    retained = true;
    if (!d) {
      cerr << "dictionary '" << std::string(name) << "' not found" << endl;
      return;
    }
  }
  if (!d)
    return;

  bool matched_any = false;
  for (const auto &name : m_cond_names) {
    auto mit = m_cond_meta.find(name);
    if (mit == m_cond_meta.end())
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
      std::lock_guard<std::mutex> lk(m_cond_mutex);
      m_cond[name] = std::move(bytes);
    }
    matched_any = true;
  }
  if (retained)
    c74::max::dictobj_release(d);
  if (matched_any)
    m_cond_dirty.store(true);
  else
    cerr << "dictionary held no keys matching a condition input" << endl;
}

// --- Matrix noise -----------------------------------------------------------
// Resolve the loaded method's matrix-drivable noise inputs (their folded
// geometry), so a Jitter matrix can drive them instead of the seeded stream.
// Called from the ctor and on reload (the perform thread is quiesced at both
// points). The jit_matrix inlets + `<name>_inlet` toggles are fixed at
// construction; only the geometry is re-derived here. Clears any cached matrix.
void live::build_matrix_noise_meta() {
  m_matrix_noise.clear();
  {
    std::lock_guard<std::mutex> lk(m_noise_mutex);
    m_noise.clear();
    m_noise_enabled.clear();
  }
  m_noise_snapshot.clear();
  m_noise_dirty.store(false);
  if (!m_model || !m_model->has_metadata())
    return;
  m_matrix_noise = detect_matrix_noise(m_model->method_inputs(m_method), cerr);
}

// Snapshot the enabled matrix noise into the map perform() reads. Audio-thread
// side of the handoff (only when the control thread flagged a change and the
// lock is free): m_supplied points into m_noise_snapshot, rewritten only here —
// and only while the worker is idle — so the worker reads it safely. A noise
// input that is disabled or has no valid matrix is removed from m_supplied, so
// the backend falls back to its seeded per-input stream.
void live::snapshot_noise() {
  if (!m_noise_dirty.load())
    return;
  if (!m_noise_mutex.try_lock())
    return;
  m_noise_snapshot = m_noise;
  std::map<std::string, bool> enabled = m_noise_enabled;
  m_noise_dirty.store(false);
  m_noise_mutex.unlock();
  for (const auto &nz : m_matrix_noise) {
    auto en = enabled.find(nz.name);
    auto sn = m_noise_snapshot.find(nz.name);
    if (en != enabled.end() && en->second && sn != m_noise_snapshot.end() &&
        sn->second.size() == nz.numel) {
      m_supplied[nz.name] = {executorch::aten::ScalarType::Float,
                             sn->second.data(), sn->second.size()};
    } else {
      m_supplied.erase(nz.name); // disabled / invalid => backend reseeds
    }
  }
}

// A Jitter matrix on a matrix-noise inlet (selector "jit_matrix"): decode it via
// the shared reader into the control-side m_noise[name]. `inlet` is the arriving
// absolute inlet index, mapped to a noise name by m_noise_inlet_name.
void live::receive_matrix(const atoms &args, int inlet) {
  if (!m_model || !m_model->has_metadata()) {
    cerr << "no model loaded" << endl;
    return;
  }
  auto it = m_noise_inlet_name.find(inlet);
  if (it == m_noise_inlet_name.end()) {
    cerr << "jit_matrix arrived on inlet " << inlet
         << ", which has no matrix-noise input" << endl;
    return;
  }
  const std::string &name = it->second;
  const MatrixNoise *nz = nullptr;
  for (const auto &z : m_matrix_noise)
    if (z.name == name) {
      nz = &z;
      break;
    }
  if (!nz) {
    cerr << "noise '" << name
         << "' is not a matrix-noise input in the loaded model" << endl;
    return;
  }
  std::vector<float> buf;
  if (!read_jit_matrix(args, *nz, buf, cerr))
    return; // shared reader logged the reason
  bool enabled;
  {
    std::lock_guard<std::mutex> lk(m_noise_mutex);
    m_noise[name] = std::move(buf);
    auto en = m_noise_enabled.find(name);
    enabled = (en != m_noise_enabled.end() && en->second);
  }
  m_noise_dirty.store(true);
  if (!enabled)
    cerr << "noise '" << name << "' received but not enabled; set @" << name
         << "_inlet 1 to use it (seeded until then)" << endl;
}

// Re-create the `<name>_inlet` boolean (long, 0/1) attributes from the current
// matrix-noise inputs. Mirrors add_dynamic_attributes; called from the ctor and
// on reload (perform thread quiesced at both points).
void live::add_noise_inlet_attributes() {
  c74::max::t_object *self = (c74::max::t_object *)(*this);
  for (const auto &n : m_noise_attr_slots) // drop a previous model's toggles
    c74::max::object_deleteattr(self, c74::max::gensym(n.c_str()));
  m_noise_attr_slots.clear();
  if (!m_model || !m_model->has_metadata())
    return;
  for (const auto &nz : m_matrix_noise) {
    {
      std::lock_guard<std::mutex> lk(m_noise_mutex);
      m_noise_enabled[nz.name] = false;
    }
    // Shared factory; pass live~'s custom thread-safe accessors so the audio
    // thread can read the toggle (via m_noise_enabled) every block. Adds the
    // on/off checkbox style + descriptive label, matching neural.gen~.
    std::string an = make_noise_inlet_attr(
        self, nz, (c74::max::method)&live::noise_inlet_get,
        (c74::max::method)&live::noise_inlet_set);
    if (!an.empty())
      m_noise_attr_slots.push_back(an);
  }
}

// Strip the fixed "_inlet" suffix to recover the noise input name an attribute
// names; returns false if `attr_name` is not a `<name>_inlet`.
static bool noise_name_of(const char *attr_name, std::string &out) {
  std::string s(attr_name);
  static const std::string suf = "_inlet";
  if (s.size() <= suf.size() || s.compare(s.size() - suf.size(), suf.size(), suf))
    return false;
  out = s.substr(0, s.size() - suf.size());
  return true;
}

c74::max::t_max_err live::noise_inlet_set(c74::max::t_object *x,
                                          c74::max::t_object *attr, long ac,
                                          c74::max::t_atom *av) {
  if (!x || ac < 1 || !av)
    return 0;
  live &self = reinterpret_cast<c74::min::minwrap<live> *>(x)->m_min_object;
  auto *nm = (c74::max::t_symbol *)c74::max::object_method(
      attr, c74::max::gensym("getname"));
  std::string name;
  if (nm && noise_name_of(nm->s_name, name)) {
    const bool on = c74::max::atom_getlong(av) != 0;
    {
      std::lock_guard<std::mutex> lk(self.m_noise_mutex);
      self.m_noise_enabled[name] = on;
    }
    self.m_noise_dirty.store(true); // re-snapshot so perform picks up the toggle
  }
  return 0;
}

c74::max::t_max_err live::noise_inlet_get(c74::max::t_object *x,
                                          c74::max::t_object *attr, long *ac,
                                          c74::max::t_atom **av) {
  live &self = reinterpret_cast<c74::min::minwrap<live> *>(x)->m_min_object;
  auto *nm = (c74::max::t_symbol *)c74::max::object_method(
      attr, c74::max::gensym("getname"));
  long v = 0;
  std::string name;
  if (nm && noise_name_of(nm->s_name, name)) {
    std::lock_guard<std::mutex> lk(self.m_noise_mutex);
    auto it = self.m_noise_enabled.find(name);
    v = (it != self.m_noise_enabled.end() && it->second) ? 1 : 0;
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

MIN_EXTERNAL(live);
