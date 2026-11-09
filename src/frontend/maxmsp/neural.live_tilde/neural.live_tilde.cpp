#include "../../../backend/backend.h"
#include "../shared/circular_buffer.h"
#include "../shared/live_attrs.h"   // LiveAttrs, rebuild_live_attrs
#include "../shared/live_inputs.h"  // LiveInputs (conditions + matrix noise)
#include "../shared/neural_common.h"
#include "../shared/runtime_console.h" // console sink, prepare_and_report
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
  // The program is loaded and m_method is ready to run (prepared at load).
  // Gates enable_model and the audio callback.
  std::atomic<bool> m_prepared{false};
  void new_backend(); // a fresh m_model posting to this object's console
  std::string m_method;
  // Absolute path to the model's .pte program (derived from its .json metadata via
  // resolve_model_pte; the .pte itself may be absent — see backend load()).
  std::string m_path;
  int m_in_dim, m_in_ratio, m_out_dim, m_out_ratio, m_higher_ratio;

  // DYNAMIC ATTRIBUTES --------------------------------------------------------
  // One Max instance attribute per settable scalar the model's method declares,
  // created from the metadata at load (shared with mc/mcs: shared/live_attrs.h).
  // m_attrs.live is passed to the model directly in slot order each block.
  LiveAttrs m_attrs;

  // CONDITION + MATRIX NOISE INPUTS ----------------------------------------
  // One control inlet per condition-role input (after the signal inlets), then
  // one jit_matrix inlet per matrix-drivable noise input (+ `<name>_inlet`
  // toggle). State, snapshot threading and the toggles live in
  // shared/live_inputs.h; m_inputs.supplied is handed to perform.
  LiveInputs m_inputs;
  int m_cond_inlet_base = 0;  // index of the first condition inlet
  int m_noise_inlet_base = 0; // index of the first matrix-noise inlet

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
    
    message<> reload_m {this, "reload", "Reload the model, or load another one: reload [path]",
        MIN_FUNCTION {
            
            if (args.size() > 1) {
                cerr << "usage: reload [path]" << endl;
                return {};
            }
            if (args.empty() && m_path.empty()) {
                cerr << "no model to reload" << endl;
                return {};
            }
            if (args.empty() || args[0].a_type == 3) {
                    std::string path = m_path; // bare `reload`: the current model
                    if (!args.empty()) {
                        // Resolve the metadata and derive the sibling .pte (which may
                        // be absent — load() then keeps metadata-only, model disabled).
                        path = resolve_model_pte(std::string(args[0]));
                        if (path.empty()) {
                            cerr << "could not find model .json metadata for "
                                 << std::string(args[0]) << endl;
                            return {};
                        }
                    }
                    m_prepared = false; // the audio callback outputs silence from here
                    m_should_stop_perform_thread = true;
                    if (m_compute_thread)
                      m_compute_thread->join();
                    m_compute_thread.reset();
                    
                    new_backend();
                    m_path = path;
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
                    if (m_model->is_loaded() && m_model->has_method(m_method))
                        m_prepared = prepare_and_report(
                            *m_model, m_method, m_path, (c74::max::t_object *)(*this));
                    
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
                    rebuild_live_attrs<live>((c74::max::t_object *)(*this), m_attrs,
                             m_model.get(), m_method);
                    // Re-resolve the condition + matrix-noise inputs and the
                    // noise toggle attributes for the new model (the inlets, fixed
                    // at construction, do not change); re-sync the seed attribute
                    // onto the new metadata default.
                    m_inputs.build(m_model.get(), m_method, cerr);
                    rebuild_noise_inlet_attrs<live>((c74::max::t_object *)(*this),
                                                    m_inputs);
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
        install_default_console_sink();
        return {};
      }};

  // Conditioning as a bare list on a condition inlet: assigned to the condition
  // input at THIS inlet's position (the alternative to the by-name dictionary
  // path). `inlet` is the arriving inlet index (see receive_list).
  message<> list_msg{
      this, "list", "Conditioning values for this inlet's condition input.",
      MIN_FUNCTION {
        m_inputs.receive_list(args, inlet, m_cond_inlet_base, cerr);
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

  // A Jitter matrix on a matrix-noise inlet: snapshot the named float32/float64
  // matrix as this inlet's noise input (used when its `<name>_inlet` toggle is on). `inlet`
  // is the arriving inlet index (see receive_matrix).
  message<> jit_matrix_msg{
      this, "jit_matrix",
      "Float32 Jitter matrix used as this inlet's noise input "
      "(enable with attribute: '[name]_inlet').",
      [this](const atoms &args, const int inlet) -> atoms {
        m_inputs.receive_matrix(args, inlet, m_noise_inlet_base, cerr);
        return {};
      }};

  message<> anything{
      this, "anything", "callback for attributes",
      [this](const c74::min::atoms &args, const int inlet) -> c74::min::atoms {
        // reload/get_methods are model-level controls. The model's settable
        // scalars are dynamic Max attributes (see shared/live_attrs.h) added
        // per-instance AFTER class registration, so Max's message dispatch
        // doesn't route to them natively (the inspector / @box-args reach them
        // directly) — this catch-all backs the bare `<name> <value>` form.
        std::string sel = args[0];
        if (sel == "get_methods") {
          for (std::string method : m_model->get_available_methods())
            cout << method << endl;
        } else if (m_attrs.is_attr(sel) && args.size() >= 2) {
          set_live_attr((c74::max::t_object *)(*this), m_attrs, sel,
                        (double)args[1]); // bare `<name> <value>`
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
                                nn_instance->m_attrs.live, nn_instance->m_inputs.supplied);
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
                                    nn_instance->m_attrs.live,
                                    nn_instance->m_inputs.supplied);
      nn_instance->m_result_available_lock.release();
    }
  }
}

live::live(const atoms &args)
    : m_compute_thread(nullptr), m_in_dim(1), m_in_ratio(1), m_out_dim(1),
      m_out_ratio(1), m_buffer_size(4096), m_method("forward"),
      m_use_thread(true), m_data_available_lock(0), m_result_available_lock(1),
      m_should_stop_perform_thread(false) {

  new_backend();
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

  // Load the method now, so its setup (CUDA kernels, weights) isn't paid by the
  // first audio block.
  if (m_model->is_loaded())
    m_prepared = prepare_and_report(*m_model, m_method, m_path,
                                    (c74::max::t_object *)(*this));

  // Resolve the model's condition-role inputs (their names/dtypes/sizes) so the
  // ctor can give each its own control inlet and perform() can bind them.
  // Also resolve the matrix-drivable noise inputs (geometry per load) so the
  // ctor can give each its own jit_matrix inlet + `<name>_inlet` toggle.
  m_inputs.build(m_model.get(), m_method, cerr);

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
  // while the by-name dictionary path still reaches every condition). Each is
  // labelled with the tensor it expects (name + shape + dtype + description).
  m_cond_inlet_base = (int)m_inlets.size();
  for (const auto &label : LiveInputs::condition_labels(m_model.get(), m_method))
    m_inlets.push_back(std::make_unique<inlet<>>(this, label));

  // One jit_matrix inlet per matrix-drivable noise input, after the condition
  // inlets (also fixed at construction; the `<name>_inlet` toggle picks the
  // matrix over the seeded stream per block — see shared/live_inputs.h).
  m_noise_inlet_base = (int)m_inlets.size();
  for (const auto &label : m_inputs.noise_labels())
    m_inlets.push_back(std::make_unique<inlet<>>(this, label));

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
  rebuild_live_attrs<live>((c74::max::t_object *)(*this), m_attrs,
                             m_model.get(), m_method);
  // One `<name>_inlet` boolean attribute per matrix-noise input (toggles the
  // matrix inlet over the seeded stream).
  rebuild_noise_inlet_attrs<live>((c74::max::t_object *)(*this), m_inputs);
  // Adopt the metadata's default seed onto the `seed` attribute (fires set_seed).
  seed = m_model->gen_seed();

  if (m_use_thread)
    m_compute_thread = std::make_unique<std::thread>(model_perform_loop, this);
}

void live::new_backend() {
  m_model = std::make_unique<Backend>();
  m_model->set_log_sink(max_console_sink((c74::max::t_object *)(*this)));
}

live::~live() {
  m_should_stop_perform_thread = true;
  if (m_compute_thread)
    m_compute_thread->join();
}

void live::operator()(audio_bundle input, audio_bundle output) {
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
      // m_inputs.supplied. Both write into m_inputs.supplied, which model_perform then reads.
      m_inputs.snapshot();

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
      // m_inputs.supplied. Refreshing the held conditioning + matrix noise is therefore
      // safe ONLY here: the worker won't touch m_inputs.supplied again until we release
      // m_data_available_lock below. (Doing this before the try_acquire — as the
      // code previously did — lets the audio thread mutate the m_inputs.supplied map
      // while a busy worker is traversing it in Backend::perform, corrupting the
      // tree. That crash only surfaced once a condition/noise input populated
      // m_inputs.supplied; with an empty map there were no nodes to race over.)
      m_inputs.snapshot();

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

MIN_EXTERNAL(live);
