#include "../../../backend/backend.h"
#include "../shared/circular_buffer.h"
#include "../shared/live_attrs.h"  // LiveAttrs, rebuild_live_attrs
#include "../shared/live_inputs.h" // LiveInputs (conditions + matrix noise)
#include "../shared/neural_common.h"
#include "../shared/runtime_console.h" // console sink, prepare_and_report
#include "c74_min.h"
#include <atomic>
#include <chrono>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

using namespace c74::min;

long simplemc_multichanneloutputs(c74::max::t_object *x, long index,
                                  long count);
long simplemc_inputchanged(c74::max::t_object *x, long index, long count);

class mc_live : public object<mc_live>, public mc_operator<> {
public:
  MIN_DIGEST{"Real-time (live) neural audio synthesis (multi-channel over batches)"};
  MIN_DESCRIPTION{"Load and run a neural audio synthesis model for real-time signal inputs. "
                  "Supports models exported from PyTorch via ExecuTorch. "
                  "Hardware-accelerated on CPU / GPU / ANE (Apple Neural Engine)."};
  MIN_TAGS{"neural audio synthesis, generative models, diffusion, autoencoders"};
  MIN_AUTHOR{"Jasper Shuoyang Zheng"};
  MIN_RELATED{"neural.live~, mcs.neural.live~, neural.tokenizer~, neural.gaussianize"};

  mc_live(const atoms &args = {});
  ~mc_live();

  // INLETS OUTLETS
  std::vector<std::unique_ptr<inlet<>>> m_inlets;
  std::vector<std::unique_ptr<outlet<>>> m_outlets;

  // CHANNELS
  std::vector<int> chans;
  int get_batches();
  bool check_inputs();

  // BACKEND RELATED MEMBERS
  std::unique_ptr<Backend> m_model;
  bool m_is_backend_init = false;
  // The program is loaded and m_method is ready to run (prepared at load).
  // Gates enable_model and the audio callback.
  std::atomic<bool> m_prepared{false};
  void reload_model(); // bare `reload`
  std::string m_method;
  // The model's Attribute-role inputs as dynamic Max attributes (see
  // shared/live_attrs.h); m_attrs.live is fed to Backend::perform each block.
  LiveAttrs m_attrs;
  // Condition + matrix-noise inputs (see shared/live_inputs.h): one control
  // inlet per condition, then one jit_matrix inlet per matrix noise, after the
  // signal inlets. m_inputs.supplied is fed to Backend::perform each block.
  LiveInputs m_inputs;
  int m_cond_inlet_base = 0;  // index of the first condition inlet
  int m_noise_inlet_base = 0; // index of the first matrix-noise inlet
  // Absolute path to the model's .pte (derived from its .json metadata via
  // resolve_model_pte; the .pte may be absent — see backend load()).
  std::string m_path;
  int m_in_dim, m_in_ratio, m_out_dim, m_out_ratio, m_higher_ratio;

  // BUFFER RELATED MEMBERS
  int m_buffer_size;
  std::unique_ptr<circular_buffer<double, float>[]> m_in_buffer;
  std::unique_ptr<circular_buffer<float, double>[]> m_out_buffer;
  std::vector<std::unique_ptr<float[]>> m_in_model, m_out_model;
  void reset_buffers();

  // AUDIO PERFORM
  bool m_use_thread, m_should_stop_perform_thread;
  std::unique_ptr<std::thread> m_compute_thread;
  std::binary_semaphore m_data_available_lock, m_result_available_lock;

  void operator()(audio_bundle input, audio_bundle output);
  void perform(audio_bundle input, audio_bundle output);

  // using mc_operator::operator();

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
                if (m_is_backend_init && m_prepared && args[0]){
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
  // default, synced after load. A matrix-noise input fed from its jit_matrix
  // inlet (with `<name>_inlet` on) replaces its seeded stream.
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

  // BOOT STAMP
  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        // make stamp
        cout << "mc.neural.live~ " << VERSION << " - " << NN_ENGINE_NAME
             << " - July-2026 - Jasper Shuoyang Zheng " << endl;
        cout << "docs: https://github.com/jasper-zheng/neural_tilde" << endl;
        install_default_console_sink();
        // mc handle
        c74::max::t_class *c = args[0];
        c74::max::class_addmethod(
            c, (c74::max::method)simplemc_multichanneloutputs,
            "multichanneloutputs", c74::max::A_CANT, 0);
        c74::max::class_addmethod(c, (c74::max::method)simplemc_inputchanged,
                                  "inputchanged", c74::max::A_CANT, 0);
        return {};
      }};

  // Conditioning as a bare list on a condition inlet: assigned to the condition
  // input at THIS inlet's position (the alternative to the by-name dictionary
  // path). Shared across every batch (the backend replicates it).
  message<> list_msg{
      this, "list", "Conditioning values for this inlet's condition input.",
      MIN_FUNCTION {
        m_inputs.receive_list(args, inlet, m_cond_inlet_base, cerr);
        return {};
      }};

  // A bare float/int on a condition inlet: a single-element list (min-api maps
  // "number" -> "float" and auto-generates the matching "int").
  message<> number_msg{
      this, "number",
      "A single conditioning value for this inlet's condition input.",
      MIN_FUNCTION {
        m_inputs.receive_list(args, inlet, m_cond_inlet_base, cerr);
        return {};
      }};

  // Conditioning as a Max dictionary, matched BY KEY NAME against the model's
  // condition inputs and MERGED (so several dicts on any inlet accumulate).
  message<> dictionary_msg{
      this, "dictionary",
      "Receive conditioning as a dictionary; merged by key name.", MIN_FUNCTION {
        m_inputs.receive_dict(args, cerr);
        return {};
      }};

  // A Jitter matrix on a matrix-noise inlet: used as this inlet's noise input
  // (shared across every batch) while its `<name>_inlet` toggle is on.
  message<> jit_matrix_msg{
      this, "jit_matrix",
      "Float32 Jitter matrix used as this inlet's noise input "
      "(enable with attribute: '[name]_inlet').",
      [this](const atoms &args, const int inlet) -> atoms {
        m_inputs.receive_matrix(args, inlet, m_noise_inlet_base, cerr);
        return {};
      }};

  message<> anything{
      this, "anything", "Set a model attribute (e.g. gain 0.5), or reload / get_methods.",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        // The model's settable scalars are dynamic Max attributes (see
        // shared/live_attrs.h) added per-instance AFTER class registration, so
        // Max's message dispatch doesn't route to them natively (the inspector /
        // @box-args reach them directly) — this catch-all backs the bare
        // `<name> <value>` form, plus nn~-style `set` / `get` aliases.
        c74::max::t_object *self = (c74::max::t_object *)(*this);
        std::string sel = args[0];
        if (sel == "reload") {
          reload_model();
        } else if (sel == "get_methods") {
          for (std::string method : m_model->get_available_methods())
            cout << method << endl;
        } else if (sel == "get_attributes") {
          for (const auto &s : m_attrs.slots)
            cout << s.name << endl;
        } else if (sel == "get") {
          if (args.size() < 2) {
            cerr << "get must be given an attribute name" << endl;
            return {};
          }
          std::string name = args[1];
          if (m_attrs.is_attr(name))
            cout << name << ": " << m_attrs.load(name.c_str()) << endl;
          else
            cerr << "no attribute '" << name << "'" << endl;
        } else if (sel == "set") {
          if (args.size() < 3) {
            cerr << "set must be given an attribute name and a value" << endl;
            return {};
          }
          std::string name = args[1];
          if (!set_live_attr(self, m_attrs, name, (double)args[2]))
            cerr << "no attribute '" << name << "'" << endl;
        } else if (m_attrs.is_attr(sel) && args.size() >= 2) {
          set_live_attr(self, m_attrs, sel, (double)args[1]); // bare `<name> <value>`
        } else {
          cerr << "no method/attribute '" << sel << "'" << endl;
        }
        return {};
      }};
};

int mc_live::get_batches() {
  return *std::min_element(chans.begin(), chans.end());
}

void model_perform(mc_live *mc_nn_instance) {
  std::vector<float *> in_model, out_model;

  for (auto &ptr : mc_nn_instance->m_in_model)
    in_model.push_back(ptr.get());

  for (auto &ptr : mc_nn_instance->m_out_model)
    out_model.push_back(ptr.get());

  mc_nn_instance->m_model->perform(
      in_model, out_model, mc_nn_instance->m_buffer_size,
      mc_nn_instance->m_method, mc_nn_instance->get_batches(),
          mc_nn_instance->m_attrs.live, mc_nn_instance->m_inputs.supplied);
}

void check_loop_buffers(mc_live *mc_nn_instance, std::vector<float *> &in_model, std::vector<float *> &out_model) {
  if (mc_nn_instance->m_in_model.size() != in_model.size())
  {
    in_model.clear();
    for (auto &ptr : mc_nn_instance->m_in_model)
      in_model.push_back(ptr.get());

  }
  if (mc_nn_instance->m_out_model.size() != out_model.size())
  {
    out_model.clear();
    for (auto &ptr : mc_nn_instance->m_out_model)
      out_model.push_back(ptr.get());
  }
}

void model_perform_loop(mc_live *mc_nn_instance) {
  std::vector<float *> in_model, out_model;

  for (auto &ptr : mc_nn_instance->m_in_model)
    in_model.push_back(ptr.get());

  for (auto &ptr : mc_nn_instance->m_out_model)
    out_model.push_back(ptr.get());

  while (!mc_nn_instance->m_should_stop_perform_thread) {
    check_loop_buffers(mc_nn_instance, in_model, out_model);
    if (mc_nn_instance->m_data_available_lock.try_acquire_for(
            std::chrono::milliseconds(200))) {
      mc_nn_instance->m_model->perform(
          in_model, out_model, mc_nn_instance->m_buffer_size,
          mc_nn_instance->m_method, mc_nn_instance->get_batches(),
          mc_nn_instance->m_attrs.live, mc_nn_instance->m_inputs.supplied);
      mc_nn_instance->m_result_available_lock.release();
    }
  }
}

mc_live::mc_live(const atoms &args)
    : m_compute_thread(nullptr), m_in_dim(1), m_in_ratio(1), m_out_dim(1),
      m_out_ratio(1), m_buffer_size(4096), m_method("forward"),
      m_use_thread(true), m_data_available_lock(0), m_result_available_lock(1),
      m_should_stop_perform_thread(false) {
  m_model = std::make_unique<Backend>();
  m_model->set_log_sink(max_console_sink((c74::max::t_object *)(*this)));
  m_is_backend_init = true;

  // CHECK ARGUMENTS
  if (!args.size()) {
    return;
  }
  if (args.size() > 0) { // ONE ARGUMENT IS GIVEN
    // Resolve the mandatory .json metadata; derive the sibling .pte (which may be
    // absent — the backend then loads metadata-only and stays disabled).
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

  // Metadata parsed but the .pte may be missing: build the I/O below, but the
  // model stays disabled (enable_model can't turn on).
  if (m_model->has_metadata() && !m_model->is_loaded()) {
    cerr << "no .pte program is loaded, inference is disabled"
         << endl;
  }

  // FIND MINIMUM BUFFER SIZE GIVEN MODEL RATIO
  m_higher_ratio = 1;
  auto model_methods = m_model->get_available_methods();
  for (int i(0); i < model_methods.size(); i++) {
    auto params = m_model->get_method_params(model_methods[i]);
    if (!params.size())
      continue; // METHOD NOT USABLE, SKIPPING
    int max_ratio = std::max(params[1], params[3]);
    m_higher_ratio = std::max(m_higher_ratio, max_ratio);
  }

  // GET MODEL'S METHOD PARAMETERS
  auto params = m_model->get_method_params(m_method);


  if (!params.size()) {
    error("method " + m_method + " not found !");
  }

  for (int i(0); i < params[0]; i++)
    chans.push_back(1);
  m_in_dim = params[0];
  m_in_ratio = params[1];
  m_out_dim = params[2];
  m_out_ratio = params[3];

  // Load the method now, so its setup (CUDA kernels, weights) isn't paid by the
  // first audio block.
  if (m_model->is_loaded())
    m_prepared = prepare_and_report(*m_model, m_method, m_path,
                                    (c74::max::t_object *)(*this));

  // Adopt the model's exported (fixed) buffer size; see neural.live_tilde.cpp ctor.
  m_buffer_size = negotiate_buffer_size(m_buffer_size, m_model->get_buffer_size(),
                                        m_higher_ratio, m_use_thread, cerr);

  // Adopt the metadata's default seed onto the `seed` attribute (fires set_seed).
  seed = m_model->gen_seed();

  // CREATE INLETS, OUTLETS and BUFFERS
  m_in_buffer = std::make_unique<circular_buffer<double, float>[]>(
      m_in_dim * get_batches());
  auto input_labels = m_model->get_input_labels(m_method);
  for (int i(0); i < m_in_dim * get_batches(); i++) {
    std::string input_label =
        (i < (int)input_labels.size())
            ? input_labels[i]
            : "(signal) model input " + std::to_string(i);
    m_inlets.push_back(
        std::make_unique<inlet<>>(this, input_label, "multichannelsignal"));
    m_in_buffer[i].initialize(m_buffer_size);
    m_in_model.push_back(std::make_unique<float[]>(m_buffer_size));
  }

  // Control inlets after the signal inlets (fixed at construction): one per
  // condition-role input, then one jit_matrix inlet per matrix-drivable noise
  // input (see shared/live_inputs.h). Their values are shared across batches.
  m_inputs.build(m_model.get(), m_method, cerr);
  m_cond_inlet_base = (int)m_inlets.size();
  for (const auto &label : LiveInputs::condition_labels(m_model.get(), m_method))
    m_inlets.push_back(std::make_unique<inlet<>>(this, label));
  m_noise_inlet_base = (int)m_inlets.size();
  for (const auto &label : m_inputs.noise_labels())
    m_inlets.push_back(std::make_unique<inlet<>>(this, label));

  m_out_buffer = std::make_unique<circular_buffer<float, double>[]>(
      m_out_dim * get_batches());
  auto output_labels = m_model->get_output_labels(m_method);
  for (int i(0); i < m_out_dim * get_batches(); i++) {
    std::string output_label =
        (i < (int)output_labels.size())
            ? output_labels[i]
            : "(signal) model output " + std::to_string(i);
    m_outlets.push_back(
        std::make_unique<outlet<>>(this, output_label, "multichannelsignal"));
    m_out_buffer[i].initialize(m_buffer_size);
    m_out_model.push_back(std::make_unique<float[]>(m_buffer_size));
  }

  // Expose the model's settable scalars as dynamic Max attributes (inspector /
  // @box-args / bare `<name> <value>`). Done before the perform thread starts.
  rebuild_live_attrs<mc_live>((c74::max::t_object *)(*this), m_attrs, m_model.get(),
                              m_method);
  rebuild_noise_inlet_attrs<mc_live>((c74::max::t_object *)(*this), m_inputs);

  if (m_use_thread)
    m_compute_thread = std::make_unique<std::thread>(model_perform_loop, this);
}

// Re-size the buffers for a new batch count (the MC channel count changed).
// m_in_dim / m_out_dim stay the model's per-batch channel counts (= the number
// of signal inlets / outlets); the buffers hold dim * batches channels.
void mc_live::reset_buffers() {
  const int n_in = m_in_dim * get_batches();
  const int n_out = m_out_dim * get_batches();
  m_in_buffer = std::make_unique<circular_buffer<double, float>[]>(n_in);
  m_out_buffer = std::make_unique<circular_buffer<float, double>[]>(n_out);
  m_in_model.clear();
  m_out_model.clear();
  for (int i(0); i < n_in; i++) {
    m_in_buffer[i].initialize(m_buffer_size);
    m_in_model.push_back(std::make_unique<float[]>(m_buffer_size));
  }
  for (int i(0); i < n_out; i++) {
    m_out_buffer[i].initialize(m_buffer_size);
    m_out_model.push_back(std::make_unique<float[]>(m_buffer_size));
  }
}

// Re-read the current model from disk. The worker is stopped meanwhile, so it
// never runs against a half-loaded model.
void mc_live::reload_model() {
  m_prepared = false; // the audio callback outputs silence from here
  m_should_stop_perform_thread = true;
  if (m_compute_thread)
    m_compute_thread->join();
  m_compute_thread.reset();
  if (!m_model->reload() && m_model->is_loaded())
    m_prepared = prepare_and_report(*m_model, m_method, m_path,
                                    (c74::max::t_object *)(*this));
  // Rebuild the attributes and re-resolve the condition / matrix-noise inputs
  // (clears held values, as neural.live~ does on reload). The worker is
  // stopped, so this is safe; the inlets, fixed at construction, do not change.
  rebuild_live_attrs<mc_live>((c74::max::t_object *)(*this), m_attrs, m_model.get(),
                              m_method);
  m_inputs.build(m_model.get(), m_method, cerr);
  rebuild_noise_inlet_attrs<mc_live>((c74::max::t_object *)(*this), m_inputs);
  m_should_stop_perform_thread = false;
  if (m_use_thread)
    m_compute_thread = std::make_unique<std::thread>(model_perform_loop, this);
}

mc_live::~mc_live() {
  m_should_stop_perform_thread = true;
  if (m_compute_thread)
    m_compute_thread->join();
}

bool mc_live::check_inputs() {
  bool check = true;
  for (int i = 1; i < chans.size(); i++) {
    if (chans[i] != chans[0])
      check = false;
  }
  return check;
}

void mc_live::operator()(audio_bundle input, audio_bundle output) {
  auto dsp_vec_size = output.frame_count();

  // CHECK IF MODEL IS LOADED AND ENABLED
  if (!m_prepared || !enable) {
    fill_with_zero(output);
    return;
  }

  // CHECK IF DSP_VEC_SIZE IS LARGER THAN BUFFER SIZE
  if (dsp_vec_size > m_buffer_size) {
    // Audio thread: Max's posting is thread-safe, min's cerr is not.
    c74::max::object_error((c74::max::t_object *)(*this),
                           "vector size (%d) larger than buffer size (%d), disabling model",
                           (int)dsp_vec_size, m_buffer_size);
    enable = false;
    fill_with_zero(output);
    return;
  }

  perform(input, output);
}

void mc_live::perform(audio_bundle input, audio_bundle output) {
  auto vec_size = input.frame_count();

  // COPY INPUT TO CIRCULAR BUFFER (signal inlets only: one per model input
  // channel; the condition / matrix-noise control inlets follow them)
  int dim_offset = 0;
  for (int i(0); i < m_in_dim; i++) {
    for (int b(0); b < get_batches(); b++) {
      auto in = input.samples(dim_offset + b);
      m_in_buffer[i * get_batches() + b].put(in, vec_size);
    }
    dim_offset += chans[i];
  }

  if (m_in_buffer[0].full()) { // BUFFER IS FULL
    if (!m_use_thread) {
      // Single-threaded: inference runs on this thread below, so refreshing the
      // held conditioning + matrix noise here cannot race the model's read.
      m_inputs.snapshot();

      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_in_dim * get_batches(); c++)
        m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);

      // CALL MODEL PERFORM IN CURRENT THREAD
      model_perform(this);

      // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_out_dim * get_batches(); c++)
        m_out_buffer[c].put(m_out_model[c].get(), m_buffer_size);

    } else if (m_result_available_lock.try_acquire()) {
      // The worker finished the previous block and is parked until the release
      // below, so it is NOT reading m_inputs.supplied: refreshing the held
      // conditioning + matrix noise is safe ONLY here.
      m_inputs.snapshot();

      // TRANSFER MEMORY BETWEEN INPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_in_dim * get_batches(); c++)
        m_in_buffer[c].get(m_in_model[c].get(), m_buffer_size);

      // TRANSFER MEMORY BETWEEN OUTPUT CIRCULAR BUFFER AND MODEL BUFFER
      for (int c(0); c < m_out_dim * get_batches(); c++)
        m_out_buffer[c].put(m_out_model[c].get(), m_buffer_size);

      // SIGNAL PERFORM THREAD THAT DATA IS AVAILABLE
      m_data_available_lock.release();
    }
  }

  // COPY CIRCULAR BUFFER TO OUTPUT
  for (int b(0); b < get_batches(); b++) {
    for (int d(0); d < m_out_dim; d++) {
      auto out = output.samples(d * get_batches() + b);
      m_out_buffer[b * m_out_dim + d].get(out, vec_size);
    }
  }
}

long simplemc_multichanneloutputs(c74::max::t_object *x, long index,
                                  long count) {
  minwrap<mc_live> *ob = (minwrap<mc_live> *)(x);
  return ob->m_min_object.get_batches();
}

long simplemc_inputchanged(c74::max::t_object *x, long index, long count) {
  minwrap<mc_live> *ob = (minwrap<mc_live> *)(x);
  bool needs_refresh = false;
  // Max reports every inlet; only the signal inlets (one per model input
  // channel) are tracked — the condition / matrix-noise inlets follow them.
  if (index < 0 || index >= (long)ob->m_min_object.chans.size())
    return false;
  if (ob->m_min_object.chans[index] != count) {
    auto old_n_batch = ob->m_min_object.get_batches();
    ob->m_min_object.chans[index] = count;
    auto new_n_batch = ob->m_min_object.get_batches();
    if (old_n_batch != new_n_batch) {
      ob->m_min_object.reset_buffers();
    }
    needs_refresh = true;
  }
  return needs_refresh;
}

MIN_EXTERNAL(mc_live);
