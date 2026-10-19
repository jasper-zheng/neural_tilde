// neural.gen~: run a latent-diffusion generator (e.g. Stable Audio 3) into a
// buffer~. The generation runs on a worker thread, never the audio thread; 
// the result is written into a named buffer~ and a "done" outlet is banged 
// once finished. The model's settable scalars are exposed as the Max 
// object's attributes (in the inspector).

//   [neural.gen~ <model.pte> [method] [buffer]]
//   <model condition>  one inlet per condition input (see below)
//   <attr> <value>     model controls       set <buffer> [ms]  target buffer~
//   init <buffer> [ms] audio-to-audio src   seed <n>          rng seed
//   init <input> <buffer> [ms]  source of one named buffer input
//   reload <path>      load a model         generate/bang     start generation
//
// The .pte is the model (diffusion, autoencoder, etc.) 
// The .json is the model's metadata, it contains the model's configuration:

// Conditioning inlets: the object creates one inlet per condition-role input
// (e.g. input_ids, attention_mask) in the order the metadata declares them.
// Each inlet accepts EITHER:
//  - a `dictionary`: (matched by key name to a condition, so it can land on any inlet)
//  - or a `list`: (matched by inlet position to a condition, so the first inlet fills the first condition, etc.) 
// A condition left unsupplied at generate time is zero-filled with a warning.
//
// Buffer inputs: a model whose metadata declares `buffer`-role inputs is fed
// each from a named buffer~. `init <input> <buffer>` sets the source of one input
// by name; `init <buffer>` sets the first (the only one, for most models). The
// host resamples / channel-maps / crops each by its own metadata, or copies it
// 1:1 when the input says "resample": false (control curves such as a 50 Hz F0,
// whose buffer~ rate is then ignored); past the source end it pads with the
// input's pad_value. An optional trailing offset in milliseconds — `init
// [input] <buffer> <ms>` — starts the read that far into the source (at the
// buffer~'s rate when resampling, at the declared rate for 1:1). Likewise `set
// <buffer> <ms>` places the generated output at `ms` into the target buffer~
// instead of resizing it (see on_done).
//
// Matrix noise: a `noise`-role gets its own extra inlet that accepts a 
// Jitter `jit_matrix` in the shape of [planes x H x W], plus a boolean 
// attribute `<name>_inlet` (e.g. `x_inlet`) toggle. 
// With the toggle on, noise is filled from the matrix input instead of the
// seed. All dims except the last two fold into the planecount; width(X) =
// shape[-1], height(Y) = shape[-2]; cell(p,x,y) -> the row-major tensor element.
// So 3-D `x` [1,256,32] => 1 plane (`jit.matrix 1 float32 32 256`) and 4-D
// `noises` [7,1,256,32] => 7 planes (`jit.matrix 7 float32 32 256`). With the
// toggle off (default), or if no valid matrix has arrived, the input is seeded
// as usual. (Maximum number of planes is 32; a >32 plane noise stays seeded, no matrix inlet.)
//

#include "../../../generative/gen_runner.h"
#include "../shared/cond_bytes.h"
#include "../shared/cond_label.h"   // cond_inlet_label
#include "../shared/matrix_noise.h" // MatrixNoise, detect_matrix_noise, read_jit_matrix
#include "../shared/neural_common.h" // resolve_model_pte
#include "../shared/runtime_console.h" // console sink, prepare_and_report
#include "c74_min.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

using namespace c74::min;

// cond_atoms_to_bytes lives in ../shared/cond_bytes.h (shared with neural.live~).

class gen : public object<gen> {
public:
  MIN_DIGEST{"Offline neural audio synthesis"};
  MIN_DESCRIPTION{"Load and run a neural audio synthesis model into a buffer~. "
                  "Supports models exported from PyTorch via ExecuTorch. "
                  "Hardware-accelerated on CPU / GPU / ANE (Apple Neural Engine)."};
  MIN_TAGS{"neural audio synthesis, generative models, diffusion"};
  MIN_AUTHOR{"Jasper Shuoyang Zheng"};
  MIN_RELATED{"neural.live~, neural.tokenizer, neural.gaussianize"};

  gen(const atoms &args = {});
  ~gen();

  // Bang fired (on the main thread) when a generation has filled the buffer~.
  outlet<> m_done{this, "(bang) generation finished"};

  // Target buffer~. create_messages=true so min-api owns the "set"/"dblclick"/
  // "notify" messages. Their lifetime matters: on patch close ~buffer_reference()
  // frees m_instance, which makes Max dispatch an unbind "notify" back to us — the
  // handler MUST outlive that. min-api nests these messages inside buffer_reference
  // (destroyed only after ~buffer_reference()'s body), so "notify" is always alive
  // during the unbind. We still take the 2nd `set` arg (start offset) by overriding
  // just "set" below (set_msg), which is never dispatched during teardown.
  buffer_reference m_buffer{this};

  // Buffer-input sources are kept by name (m_init_src) and read through a
  // short-lived buffer_reference at generate time (snapshot_init_audio), so a
  // model may take any number of buffer inputs.

  // ---- controls -----------------------------------------------------------
  // `seed` is a host RNG control, not a
  // model input, so it stays a normal static attribute.
  attribute<int> seed{this, "seed", 0,
                      description{"RNG seed for the generation noise."}};

  // Other per-model scalar controls (e.g. cfg_scale) are created
  // dynamically from the metadata as named Max instance attributes (see
  // add_dynamic_attributes / m_attr_slots). 

  // ---- documentation-only arguments --------------------------------------
  argument<symbol> path_arg{this, "model path",
                            "Path to the .pte model (with its .json metadata beside it)."};
  argument<symbol> method_arg{this, "method",
                              "Method to call (default \"forward\")."};
  argument<symbol> buffer_arg{this, "buffer", "Target buffer~ name."};

  // ---- messages -----------------------------------------------------------
  // Conditioning from the patch as a Max dictionary (e.g. one of the two
  // single-key dicts a neural.tokenizer emits). Matched BY KEY NAME against the
  // model's condition inputs and MERGED into the cache, so several dicts (on any
  // inlets) accumulate. An explicit generate/bang then runs the model.
  message<> dictionary_msg{
      this, "dictionary",
      "Receive conditioning (e.g. input_ids/attention_mask) as a dictionary; "
      "merged by key name.",
      MIN_FUNCTION {
        receive_dict(args);
        return {};
      }};

  // Conditioning as a bare list on a condition inlet: the values are assigned to
  // the condition input at THIS inlet's position (alternative to the by-name
  // dictionary path). `inlet` is the arriving inlet index (see receive_list).
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

  // A Jitter matrix on a matrix-noise inlet: snapshot the named float32/float64
  // matrix as this inlet's noise input (used when its `<name>_inlet` toggle is on; see
  // the file header for the matrix layout). `inlet` is the arriving inlet index.
  message<> jit_matrix_msg{
      this, "jit_matrix",
      "Float32 Jitter matrix used as this inlet's noise input "
      "(enable with attribute: '[name]_inlet').",
      MIN_FUNCTION {
        receive_matrix(args, inlet);
        return {};
      }};

  message<> generate_msg{this, "generate", "Start a generation.",
                         MIN_FUNCTION {
                           trigger();
                           return {};
                         }};
  message<> bang_msg{this, "bang", "Start a generation.", MIN_FUNCTION {
                       trigger();
                       return {};
                     }};

  message<> get_methods_msg{
      this, "get_methods", "Print the methods available in the loaded model.",
      MIN_FUNCTION {
        if (m_loaded)
          for (auto &m : m_runner.available_methods())
            cout << m << endl;
        return {};
      }};

  message<> reload_msg{
      this, "reload", "Load a generative model from a path.", MIN_FUNCTION {
        if (args.size() != 1) {
          cerr << "usage: reload <path>" << endl;
          return {};
        }
        if (m_busy.load()) {
          cerr << "busy generating, cannot reload now" << endl;
          return {};
        }
        load_model(std::string(args[0]));
        return {};
      }};

  // Buffer-input sources: `init <input> <buffer> [start_ms]` points the buffer
  // input named <input> at a named buffer~ (read at generate time), optionally
  // starting the read `start_ms` into it; `init <buffer> [start_ms]` does the
  // same for the first buffer input. `init` with no arg (or `init 0`) clears all
  // sources, so every buffer input is filled with its pad_value.
  message<> init_msg{
      this, "init",
      "Set a buffer input's source buffer~ and start offset in ms. Usage: init "
      "[input] [buffer] [start_ms]",
      MIN_FUNCTION {
        receive_init(args);
        return {};
      }};

  // Target buffer~: `set <buffer> [start_ms]`. With a start offset the output is
  // placed at `start_ms` into the (unresized) buffer; without one, on_done keeps
  // the legacy behavior of resizing the buffer to the model's length (see on_done).
  // This intentionally OVERRIDES min-api's auto "set": messages register by name
  // into the object's map (c74_min_message.h) and m_buffer (declared first) is
  // constructed before set_msg, so set_msg's registration wins for "set". Name-based
  // dispatch then routes `set` here so we can read the 2nd (offset) atom the auto
  // "set" would drop. dblclick/notify are left to m_buffer's auto messages: their
  // teardown lifetime must stay under min-api's control (see the m_buffer comment).
  message<> set_msg{
      this, "set",
      "Set the target buffer~ and output start offset in ms. Usage: set [buffer] [start_ms]",
      MIN_FUNCTION {
        if (args.empty() || args[0].type() != message_type::symbol_argument) {
          cerr << "usage: set <buffer> [start_ms]" << endl;
          return {};
        }
        m_buffer.set(symbol(args[0]));
        if (args.size() > 1) {
          m_out_offset_ms = std::max(0.0, (double)args[1]);
          m_have_out_offset = true;
        } else {
          m_out_offset_ms = 0.0;
          m_have_out_offset = false;
        }
        return {};
      }};

  // Catch-all that sets one of the model's dynamic attributes (named per the
  // metadata) from a bare `<name> <value>` message. Those attributes are added
  // per-instance (object_addattr) AFTER class registration, so Max's message
  // dispatch doesn't route to them natively — this handler is how a patch
  // message reaches them (the inspector and @box-args reach them directly).
  message<> anything{
      this, "anything", "Set a model attribute (e.g. cfg_scale).",
      MIN_FUNCTION {
        if (args.empty())
          return {};
        std::string sel = args[0];
        if (is_attr(sel) && args.size() >= 2)
          set_attr(sel, (double)args[1]); // bare `<name> <value>`
        else
          cerr << "no method/attribute '" << sel << "'" << endl;
        return {};
      }};

  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const atoms &args, const int inlet) -> atoms {
        cout << "neural.gen~ " << VERSION << " - " << NN_ENGINE_NAME
             << " - Jasper Shuoyang Zheng" << endl;
        cout << "docs: https://github.com/jasper-zheng/neural_tilde" << endl;
        install_default_console_sink();
        return {};
      }};

private:
  GenRunner m_runner;
  std::string m_method{"forward"};
  OutputSpec m_out_spec;
  // m_loaded: the metadata parsed, so the I/O (inlets/attributes) is known.
  // m_runnable: the .pte program is also present, so generation is possible.
  bool m_loaded{false};
  bool m_runnable{false};

  // One message inlet per condition-role input, in metadata order, created in the
  // constructor (ctor) (Max can't add inlets later). m_cond_inlets[i] is
  // the name of the condition reached by inlet i.
  std::vector<std::unique_ptr<inlet<>>> m_inlets;
  std::vector<std::string> m_cond_inlets;

  // One per Attribute-role input: a Max instance attribute named `name`, created
  // from the metadata at load time (clamp range + default kept for listing/reload).
  struct AttrSlot {
    std::string name;
    double min, max, def;
    std::string description;
  };
  std::vector<AttrSlot> m_attr_slots;

  // Cached conditioning: each supplied condition input's values, converted to its
  // declared dtype and stored as raw bytes keyed by input name. Filled by the
  // dictionary (by name) and list (by inlet) paths and MERGED, never wholesale
  // replaced, so the four-condition (positive + negative prompt) case accumulates
  // across separate messages. Only what was actually supplied lives here; missing
  // conditions are zero-filled per-job at generate time.
  std::map<std::string, std::vector<char>> m_cond;

  // Buffer-input sources from `init`: buffer-input name -> source buffer~ + read
  // offset (ms; applied in snapshot_init_audio, 0 => from the source start). An
  // input absent here is filled with its pad_value.
  struct InitSource {
    symbol buffer;
    double offset_ms;
  };
  std::map<std::string, InitSource> m_init_src;
  // Output start offset (ms, at the target buffer~'s rate) from `set`.
  // m_have_out_offset selects the output write mode in on_done: true => place
  // into the (unresized) target at m_out_offset_ms; false => legacy
  // resize-to-model-length.
  double m_out_offset_ms{0.0};
  bool m_have_out_offset{false};

  // ---- matrix-driven noise ---------------------------------
  // Each noise-role input whose shape folds to planes x H x W within Jitter's
  // plane cap. The MatrixNoise geometry + the matrix read live in the shared
  // matrix_noise.h (also used by neural.live~); re-derived on every load (so
  // reload tracks the model); the inlets, however, are fixed at construction.
  std::vector<MatrixNoise> m_matrix_noise;
  // Absolute inlet index -> matrix-noise input name (built once in the ctor).
  std::map<int, std::string> m_noise_inlet_name;
  // The `<name>_inlet` boolean attributes created from m_matrix_noise (for cleanup).
  std::vector<std::string> m_noise_inlet_slots;
  // Latest matrix snapshot per matrix-noise input (filled by receive_matrix) and
  // a flag for whether a valid one has arrived.
  std::map<std::string, std::vector<float>> m_noise;
  std::map<std::string, bool> m_noise_have;

  // worker thread + handoff
  std::thread m_worker;
  std::binary_semaphore m_data_available{0};
  std::atomic<bool> m_should_stop{false};
  std::atomic<bool> m_busy{false};

  // job snapshot + result
  std::map<std::string, std::vector<char>> m_job_cond;
  std::map<std::string, double> m_job_attrs;
  // Per-job buffer inputs (name -> channel-major floats at the input geometry);
  // an input missing here is filled with its pad_value inside generate().
  std::map<std::string, std::vector<float>> m_job_init;
  // Per-job snapshot of matrix-driven noise (name -> float values); only inputs
  // whose `<name>_inlet` toggle is on with a valid matrix appear here, the rest
  // stay seeded inside generate().
  std::map<std::string, std::vector<float>> m_job_noise;
  uint64_t m_job_seed{0};
  std::vector<float> m_result;
  int m_gen_ok{1};

  // Deferred (main-thread) delivery of a finished generation.
  queue<> m_done_q{this, MIN_FUNCTION {
                     on_done();
                     return {};
                   }};

  void load_model(std::string model_path);
  void add_dynamic_attributes();
  void build_condition_meta();
  bool is_attr(const std::string &name) const;
  void set_attr(const std::string &name, double value);
  void receive_dict(const atoms &args);
  void receive_list(const atoms &args, int inlet);
  void receive_matrix(const atoms &args, int inlet);
  void receive_init(const atoms &args);
  std::vector<const InputSpec *> buffer_inputs();
  void snapshot_init_audio();
  void detect_matrix_noise();
  void trigger();
  void worker_loop();
  void on_done();
};

bool gen::is_attr(const std::string &name) const {
  for (const auto &s : m_attr_slots)
    if (s.name == name)
      return true;
  return false;
}

void gen::set_attr(const std::string &name, double value) {
  if (!is_attr(name)) {
    cerr << "no attribute '" << name << "'" << endl;
    return;
  }
  // Max clamps to the attribute's filter range on set.
  c74::max::object_attr_setfloat((c74::max::t_object *)(*this),
                                 c74::max::gensym(name.c_str()), value);
}

// Create one Max instance attribute per Attribute-role input, named by the input
// (seconds_total, cfg_scale, ...). Max owns the storage; @box-args (applied by
// the wrapper after this ctor runs), the inspector, and `name value` messages all
// set it, clamped to [min,max]. Re-derived on every load (reload may differ).
void gen::add_dynamic_attributes() {
  c74::max::t_object *self = (c74::max::t_object *)(*this);
  for (const auto &s : m_attr_slots) // drop any attributes from a previous model
    c74::max::object_deleteattr(self, c74::max::gensym(s.name.c_str()));
  m_attr_slots.clear();
  for (const auto &an : m_noise_inlet_slots) // and the noise-inlet toggles
    c74::max::object_deleteattr(self, c74::max::gensym(an.c_str()));
  m_noise_inlet_slots.clear();
  if (!m_loaded)
    return;

  // One boolean `<name>_inlet` per matrix-drivable noise input: on => read that
  // noise from its jit_matrix inlet instead of the seed (see receive_matrix /
  // trigger). m_matrix_noise is filled by detect_matrix_noise() first. The toggle
  // attribute is created by the shared factory (Max-owned storage here; gen~ reads
  // it once per generate, so it needs no custom accessors).
  for (const auto &nz : m_matrix_noise) {
    std::string an = make_noise_inlet_attr(self, nz);
    if (!an.empty())
      m_noise_inlet_slots.push_back(an);
  }

  for (const auto &in : m_runner.gen_inputs(m_method)) {
    if (in.role != Role::Attribute)
      continue;
    // int64 inputs become Max `long` attributes (whole numbers in the
    // inspector); the value is still read as a double in trigger() and rounded
    // to int64 by GenRunner::generate.
    const bool is_int = in.dtype == executorch::aten::ScalarType::Long;
    c74::max::t_object *attr = (c74::max::t_object *)c74::max::attribute_new(
        in.name.c_str(), c74::max::gensym(is_int ? "long" : "float64"), 0,
        nullptr, nullptr);
    if (!attr)
      continue;
    if (in.max > in.min)
      c74::max::attr_addfilter_clip(attr, in.min, in.max, 1, 1);
    c74::max::object_addattr(self, attr);
    c74::max::object_attr_setfloat(self, c74::max::gensym(in.name.c_str()),
                                   in.def);
    // Surface the metadata description as the attribute's inspector label (the
    // per-object mirror of CLASS_ATTR_LABEL).
    if (!in.description.empty())
      c74::max::object_attr_addattr_format(
          self, in.name.c_str(), "label", c74::max::gensym("symbol"), 0, "s",
          c74::max::gensym(in.description.c_str()));
    m_attr_slots.push_back({in.name, in.min, in.max, in.def, in.description});
  }
}

void gen::load_model(std::string model_path) {
  m_loaded = false;
  m_runnable = false;
  try {
    // Resolve the mandatory .json metadata and derive the sibling .pte path.
    // In the case of .json loaded but .pte missing: the backend loads metadata-only and stays disabled.
    std::string abs = resolve_model_pte(model_path);
    if (abs.empty()) {
      cerr << "could not find model .json metadata for " << model_path << endl;
      return;
    }
    if (m_runner.load(abs)) {
      cerr << "error loading " << abs << endl;
      return;
    }
    if (!m_runner.is_generative()) {
      cerr << abs << " is not a .gen~ model (use neural.live~ for streaming models)"
           << endl;
      return;
    }
    m_out_spec = m_runner.output_spec(m_method);
    m_loaded = true;                      // metadata parsed: I/O is known
    // .pte present and the method loads: generation possible. Loading the
    // method here keeps its setup (CUDA kernels, weights) off the first generate.
    m_runnable = m_runner.runnable() &&
                 prepare_and_report(m_runner, m_method, abs, (c74::max::t_object *)(*this));
    detect_matrix_noise();    // find matrix-drivable noise inputs (for attrs)
    build_condition_meta();   // re-resolve condition inlets + reset cache (reload-safe)
    add_dynamic_attributes(); // expose the model's scalar params as attributes
    // Keep only the `init` sources whose input still exists (reload-safe); a
    // legacy `init <buffer>` sent before any model loaded (key "") goes to the
    // first buffer input.
    {
      std::map<std::string, InitSource> kept;
      const auto bis = buffer_inputs();
      for (const auto *bi : bis)
        if (m_init_src.count(bi->name))
          kept[bi->name] = m_init_src[bi->name];
      if (!bis.empty() && !kept.count(bis.front()->name) && m_init_src.count(""))
        kept[bis.front()->name] = m_init_src[""];
      m_init_src.swap(kept);
    }
    if (!m_runner.runnable()) {
      cerr << "no .pte program is loaded, generation is disabled"
           << endl;
      cout << "loaded metadata for " << abs << endl;
    }
  } catch (...) {
    cerr << "could not resolve " << model_path << endl;
  }
}

// Read a Max dictionary (sent as `dictionary <name>`) into the cached
// conditioning, MERGING by key: each condition-role input whose name is a key in
// this dict is converted to its declared dtype and stored; conditions absent from
// this dict are left untouched (they may arrive in another dict or as a list on
// their inlet). So the two single-key dicts a neural.tokenizer emits accumulate.
void gen::receive_dict(const atoms &args) {
  if (!m_loaded) {
    cerr << "no model loaded" << endl;
    return;
  }
  if (args.empty()) {
    cerr << "dictionary message has no payload" << endl;
    return;
  }

  // A `dictionary` message may carry the dict either as an object atom (a direct
  // t_dictionary*, e.g. from a patch cord) or as a registered-name symbol.
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
      cerr << "dictionary '" << std::string(name.c_str()) << "' not found" << endl;
      return;
    }
  }
  if (!d) {
    cerr << "received an empty dictionary" << endl;
    return;
  }

  bool matched_any = false;
  for (const auto &in : m_runner.gen_inputs(m_method)) {
    if (in.role != Role::Condition)
      continue;
    c74::max::t_symbol *key = c74::max::gensym(in.name.c_str());
    if (!c74::max::dictionary_hasentry(d, key))
      continue; // condition not present in THIS dict — leave the cache as is
    long argc = 0;
    c74::max::t_atom *argv = nullptr;
    c74::max::t_atom single;
    if (c74::max::dictionary_getatoms(d, key, &argc, &argv) !=
            c74::max::MAX_ERR_NONE ||
        !argv || argc <= 0) {
      // A scalar entry (e.g. JSON `"language": 3` for a [1, 1] condition):
      // read it as a 1-element list.
      if (c74::max::dictionary_getatom(d, key, &single) != c74::max::MAX_ERR_NONE)
        continue;
      argc = 1;
      argv = &single;
    }
    m_cond[in.name] = cond_atoms_to_bytes(
        in.dtype, argc, [&](long k) { return c74::max::atom_getlong(argv + k); },
        [&](long k) { return c74::max::atom_getfloat(argv + k); });
    matched_any = true;
  }
  if (retained)
    c74::max::dictobj_release(d);
  if (!matched_any)
    cerr << "dictionary held no keys matching a condition input" << endl;
}

// Read a bare list arriving on a condition inlet into the cached conditioning:
// the values are assigned to the condition input at this inlet's position (the
// alternative to the by-name dictionary path). Inlet i maps to m_cond_inlets[i].
void gen::receive_list(const atoms &args, int inlet) {
  if (!m_loaded) {
    cerr << "no model loaded" << endl;
    return;
  }
  if (inlet < 0 || inlet >= (int)m_cond_inlets.size()) {
    cerr << "inlet " << inlet << " has no condition input" << endl;
    return;
  }
  const std::string &name = m_cond_inlets[inlet];
  const InputSpec *spec = nullptr;
  for (const auto &in : m_runner.gen_inputs(m_method))
    if (in.role == Role::Condition && in.name == name) {
      spec = &in;
      break;
    }
  if (!spec)
    return;

  // Warn (but still store) on a length mismatch vs the input's declared numel —
  // generate() validates it again and will refuse a wrong-sized buffer.
  size_t numel = 1;
  for (int d : spec->shape)
    numel *= (size_t)d;
  if ((size_t)args.size() != numel)
    cerr << "list for '" << name << "' has " << args.size() << " value(s), expected "
         << numel << endl;

  m_cond[name] = cond_atoms_to_bytes(
      spec->dtype, (long)args.size(),
      [&](long k) { return (long long)args[k]; },
      [&](long k) { return (double)args[k]; });
}

gen::gen(const atoms &args) {
  // Runtime diagnostics (load errors, ExecuTorch's reason for a failed
  // generate) go to the Max console, from the worker thread too.
  m_runner.set_log_sink(max_console_sink((c74::max::t_object *)(*this)));
  if (args.size() > 1)
    m_method = std::string(args[1]);
  if (args.size() > 0)
    load_model(std::string(args[0]));
  if (args.size() > 2)
    m_buffer.set(symbol(args[2]));

  // One inlet per condition-role input, in metadata order. m_cond_inlets was filled by load_model
  // above (build_condition_meta), the same way m_matrix_noise is.
  if (m_cond_inlets.empty()) {
    m_inlets.push_back(std::make_unique<inlet<>>(
        this, "control: generate/bang, dictionary, init [input] <buffer> [ms], "
              "attributes"));
  } else {
    // Label each condition inlet with the tensor it expects (name + shape +
    // dtype + optional description), built from the same gen_inputs() the inlet
    // order derives from (so the i-th condition input lands on the i-th inlet).
    size_t i = 0;
    for (const auto &in : m_runner.gen_inputs(m_method)) {
      if (in.role != Role::Condition)
        continue;
      std::string label =
          cond_inlet_label(in) + " (list, or dictionary by name)";
      if (i == 0)
        label += " + control: generate/bang, init [input] <buffer> [ms], attributes";
      m_inlets.push_back(std::make_unique<inlet<>>(this, label));
      i++;
    }
  }

  // One extra inlet per matrix-drivable noise input (after the condition
  // inlets), accepting a jit_matrix. The toggle
  // `@<name>_inlet` decides whether the matrix or the seed is used per generate.
  {
    int base = (int)m_inlets.size();
    for (size_t i = 0; i < m_matrix_noise.size(); i++) {
      const auto &nz = m_matrix_noise[i];
      std::string label =
          nz.name + " noise (jit_matrix " + std::to_string(nz.n0) +
          " float32/float64 " + std::to_string(nz.n2) + " " +
          std::to_string(nz.n1) + "; enable @" +
          nz.name + "_inlet)";
      m_inlets.push_back(std::make_unique<inlet<>>(this, label));
      m_noise_inlet_name[base + (int)i] = nz.name;
    }
  }

  m_worker = std::thread(&gen::worker_loop, this);
}

gen::~gen() {
  m_should_stop.store(true);
  m_data_available.release(); // wake the worker so it can exit
  if (m_worker.joinable())
    m_worker.join();
}

// The loaded method's buffer-role inputs, in declared order.
std::vector<const InputSpec *> gen::buffer_inputs() {
  std::vector<const InputSpec *> v;
  if (!m_loaded)
    return v;
  for (const auto &in : m_runner.gen_inputs(m_method))
    if (in.role == Role::Buffer)
      v.push_back(&in);
  return v;
}

// `init` message: see init_msg. `init <input> <buffer> [ms]` is recognised when
// the first atom names a buffer input AND the second is a symbol; anything else
// with a leading symbol is the legacy `init <buffer> [ms]` (=> first input).
void gen::receive_init(const atoms &args) {
  if (args.empty() || args[0].type() != message_type::symbol_argument) {
    m_init_src.clear(); // `init` / `init 0`: clear every source
    return;
  }
  const auto bis = buffer_inputs();
  if (m_loaded && bis.empty()) {
    cerr << "the method has no buffer input, 'init' ignored" << endl;
    return;
  }
  const std::string a0 = std::string(args[0]);
  const InputSpec *named = nullptr;
  for (const auto *bi : bis)
    if (bi->name == a0)
      named = bi;
  if (named && args.size() >= 2 &&
      args[1].type() == message_type::symbol_argument) {
    double ms = args.size() > 2 ? std::max(0.0, (double)args[2]) : 0.0;
    m_init_src[named->name] = {symbol(args[1]), ms};
    return;
  }
  double ms = args.size() > 1 ? std::max(0.0, (double)args[1]) : 0.0;
  // Before a model is loaded there are no names to key by; keep the 0.0.4
  // meaning under the first input once one exists (load_model prunes).
  const std::string target = bis.empty() ? std::string() : bis.front()->name;
  if (bis.size() > 1)
    cwarn << "the method has " << bis.size() << " buffer inputs, 'init " << a0
          << "' sets the first ('" << target << "'); use 'init <input> <buffer>'"
          << endl;
  m_init_src[target] = {symbol(args[0]), ms};
}

// Read each buffer input's source buffer~ into m_job_init, preprocessed by that
// input's metadata (GenRunner::prepare_buffer_input: resample or 1:1, channel
// map, crop, pad_value). Runs on the main thread (buffer access) as part of the
// trigger snapshot. An input with no source, or whose buffer~ is missing, is
// left out (=> generate() fills it with its pad_value) with a warning.
void gen::snapshot_init_audio() {
  m_job_init.clear();
  for (const auto *bi : buffer_inputs()) {
    auto it = m_init_src.find(bi->name);
    if (it == m_init_src.end()) {
      cwarn << "no source for buffer input '" << bi->name << "', filling with "
            << bi->pad_value << " (send 'init " << bi->name << " <buffer>')"
            << endl;
      continue;
    }
    buffer_reference ref{this, nullptr, false};
    ref.set(it->second.buffer);
    if (!ref) {
      cerr << "buffer~ '" << std::string(it->second.buffer.c_str()) << "' for input '"
           << bi->name << "' not found, filling with " << bi->pad_value << endl;
      continue;
    }
    buffer_lock<false> b(ref);
    // An existing but empty buffer~ (no samples) is a 0-frame source => all pad.
    int sch = (int)b.channel_count();
    int sframes = b.valid() ? (int)b.frame_count() : 0;
    if (sframes == 0)
      cwarn << "buffer~ '" << std::string(it->second.buffer.c_str()) << "' for input '"
            << bi->name << "' is empty, filling with " << bi->pad_value << endl;
    double ssr = b.samplerate();
    std::vector<float> src((size_t)sch * sframes);
    for (int t = 0; t < sframes; t++)
      for (int c = 0; c < sch; c++)
        src[(size_t)c * sframes + t] = b.lookup((size_t)t, (size_t)c);
    const auto g = GenRunner::buffer_geometry(*bi, m_out_spec.sample_rate);
    if (!bi->resample && std::lround(ssr) != g.sr)
      cwarn << "buffer~ '" << std::string(it->second.buffer.c_str()) << "' is read 1:1 into '"
            << bi->name << "'; its sample rate " << ssr
            << " Hz is ignored (declared " << g.sr << " Hz)" << endl;
    GenRunner::prepare_buffer_input(*bi, m_out_spec.sample_rate, src, sch,
                                    (size_t)sframes, ssr, it->second.offset_ms,
                                    m_job_init[bi->name]);
  }
}

// Find the noise-role inputs that a Jitter matrix can drive: the shape folds to
// planes x H x W within Jitter's plane cap (all dims except the last two fold
// into the planecount). Re-derived on every load so a reload tracks the current
// model (the inlets, fixed at construction, do not change).
void gen::detect_matrix_noise() {
  m_matrix_noise.clear();
  m_noise.clear();      // drop a previous model's matrix snapshots on (re)load
  m_noise_have.clear();
  if (!m_loaded)
    return;
  m_matrix_noise = ::detect_matrix_noise(m_runner.gen_inputs(m_method), cerr);
}

// Resolve the loaded method's condition-role inputs into the inlet->name list
// m_cond_inlets (in declared order) and drop any conditioning cached from a
// previous model. Called from load_model (ctor + reload), mirroring how
// detect_matrix_noise resolves the noise inputs. gen~ runs entirely on the
// control thread, so — unlike live~'s build_condition_meta — it needs no mutex
// and no (dtype, numel) meta map: receive_dict/receive_list/trigger re-query the
// cheap gen_inputs() const-ref directly.
void gen::build_condition_meta() {
  m_cond_inlets.clear();
  m_cond.clear();
  if (!m_loaded)
    return;
  for (const auto &in : m_runner.gen_inputs(m_method))
    if (in.role == Role::Condition)
      m_cond_inlets.push_back(in.name);
}

// Snapshot a named Jitter float32 matrix arriving on a matrix-noise inlet into
// m_noise[name]. Validated against the input's folded geometry (planecount = n0,
// width = n2, height = n1); a mismatch is warned and ignored (the input then
// falls back to seeded noise at generate time). Read on the scheduler thread,
// copied out under the matrix lock since the sender owns/reuses it.
void gen::receive_matrix(const atoms &args, int inlet) {
  if (!m_loaded) {
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
  // Warn if a matrix arrives while this input's inlet is disabled: the snapshot
  // is still kept (so it's ready once enabled), but generate() ignores it and
  // uses seeded noise until `@<name>_inlet 1` is set. Same accessor as trigger().
  if (!c74::max::object_attr_getlong((c74::max::t_object *)(*this),
                                     c74::max::gensym((name + "_inlet").c_str())))
    cerr << "noise '" << name << "' received from inlet but not enabled. Please set @"
         << name << "_inlet 1 to use it (using seeded noise used until then)" << endl;

  // Decode the matrix into m_noise[name] via the shared reader (geometry checks
  // + the cell(p,x,y)->[p,y,x] transpose). Only mark it usable on a clean read.
  if (read_jit_matrix(args, *nz, m_noise[name], cerr))
    m_noise_have[name] = true;
}

void gen::trigger() {
  if (!m_loaded) {
    cerr << "no model loaded" << endl;
    return;
  }
  if (!m_runnable) {
    cerr << "the model can't run (see the load messages above), cannot generate"
         << endl;
    return;
  }
  if (m_busy.exchange(true)) {
    cerr << "still generating, ignoring" << endl;
    return;
  }
  // Snapshot conditioning for the worker. Any condition input not supplied is
  // zero-filled (a zero attention_mask ⇒ unconditional, so a forgotten negative
  // prompt just runs as plain text-to-audio) with a one-line warning; the model
  // therefore always runs. m_cond keeps only what was actually supplied, so the
  // fill is recomputed each generate.
  m_job_cond = m_cond;
  {
    using executorch::aten::ScalarType;
    int inlet_idx = 0;
    for (const auto &in : m_runner.gen_inputs(m_method)) {
      if (in.role != Role::Condition)
        continue;
      if (!m_job_cond.count(in.name)) {
        size_t numel = 1;
        for (int d : in.shape)
          numel *= (size_t)d;
        const size_t esize = (in.dtype == ScalarType::Long) ? 8 : 4;
        m_job_cond[in.name] = std::vector<char>(numel * esize, 0);
        cwarn << "condition '" << in.name
             << "' not supplied, filling zeros (inlet " << inlet_idx << ")"
             << endl;
      }
      inlet_idx++;
    }
  }
  // Read each model attribute's current (clamped) value from Max into the job map
  // by name.
  m_job_attrs.clear();
  for (const auto &s : m_attr_slots)
    m_job_attrs[s.name] = c74::max::object_attr_getfloat(
        (c74::max::t_object *)(*this), c74::max::gensym(s.name.c_str()));
  snapshot_init_audio(); // fills m_job_init (missing inputs => pad_value)
  // Snapshot matrix-driven noise for any matrix-noise input whose `<name>_inlet`
  // toggle is on and that has a valid matrix; the rest stay seeded.
  m_job_noise.clear();
  for (const auto &nz : m_matrix_noise) {
    std::string an = nz.name + "_inlet";
    long on = c74::max::object_attr_getlong((c74::max::t_object *)(*this),
                                            c74::max::gensym(an.c_str()));
    if (!on)
      continue;
    auto hv = m_noise_have.find(nz.name);
    if (hv != m_noise_have.end() && hv->second &&
        m_noise[nz.name].size() == nz.numel) {
      m_job_noise[nz.name] = m_noise[nz.name];
    } else {
      cerr << "noise '" << nz.name
           << "' inlet enabled but no valid matrix received, using seeded noise"
           << endl;
    }
  }
  m_job_seed = (uint64_t)(int)seed;
  m_data_available.release();
}

void gen::worker_loop() {
  while (!m_should_stop.load()) {
    if (m_data_available.try_acquire_for(std::chrono::milliseconds(200))) {
      if (m_should_stop.load())
        break;
      // Bind the snapshotted conditioning to the model's condition inputs by
      // name, plus each buffer onto its buffer input and any matrix-driven
      // noise onto its noise inputs (each a no-op if none was supplied).
      std::map<std::string, SuppliedInput> supplied;
      const auto &inputs = m_runner.gen_inputs(m_method);
      GenRunner::bind_condition(inputs, m_job_cond, supplied);
      GenRunner::bind_buffers(inputs, m_job_init, supplied);
      GenRunner::bind_noise(inputs, m_job_noise, supplied);
      m_gen_ok = m_runner.generate(m_method, supplied, m_job_attrs, m_job_seed,
                                   m_result);
      m_done_q.set(); // hand back to the main thread
    }
  }
}

void gen::on_done() {
  if (m_gen_ok != 0) {
    cerr << "generation failed" << endl;
    m_busy.store(false);
    return;
  }

  int len = m_out_spec.length;
  int ch = m_out_spec.channels;
  if (len <= 0 || ch <= 0) {
    if (m_out_spec.shape.size() == 3) {
      ch = m_out_spec.shape[1];
      len = m_out_spec.shape[2];
    } else if (m_out_spec.shape.size() == 2) {
      ch = m_out_spec.shape[0];
      len = m_out_spec.shape[1];
    }
  }

  // Where in the target buffer~ the output starts (frames). Legacy (no `set`
  // offset): resize the buffer to the model's length and write from frame 0.
  // Place-mode (`set <buffer> <ms>`): leave the buffer as the user sized it and
  // write at m_out_offset_ms — converted with the buffer's own sample rate — so
  // the ms value lands at the right spot on its timeline.
  int off = 0;
  // Existence, not buffer_lock::valid(): valid() is false for a buffer~ that
  // exists but has no samples yet (e.g. `[buffer~ f0]` with no size), which is
  // exactly the buffer the legacy mode resizes below.
  if (!m_buffer) {
    cerr << "target buffer~ not set or not found (send 'set <name>')" << endl;
    m_busy.store(false);
    return;
  }
  {
    // Separate lock from the write: resize may move memory; sizing the offset
    // needs the buffer's sr, which we read here.
    buffer_lock<false> b(m_buffer);
    if (m_have_out_offset) {
      double bsr = b.samplerate();
      if (bsr <= 0.0)
        bsr = m_out_spec.sample_rate > 0 ? m_out_spec.sample_rate : 44100.0;
      off = (int)std::llround(std::max(0.0, m_out_offset_ms) * 0.001 * bsr);
    } else if ((int)b.frame_count() != len) {
      b.resize_in_samples(len);
    }
  }

  // Write channel-major m_result into the (interleaved) buffer~ at `off`.
  {
    buffer_lock<false> b(m_buffer);
    if (!b.valid()) {
      cerr << "target buffer~ '" << std::string(m_buffer.name().c_str())
           << "' has no samples to write into" << endl;
      m_busy.store(false);
      return;
    }
    int bch = (int)b.channel_count();
    int wch = std::min(ch, bch);
    // Frames available from `off` to the buffer end; in place-mode the output is
    // cropped to fit (the region before `off` is left as it was).
    int wlen = std::max(0, std::min(len, (int)b.frame_count() - off));
    if (bch != ch)
      cerr << "buffer~ has " << bch << " channel(s), model has "
           << ch << " (writing " << wch << ")" << endl;
    if (m_have_out_offset && wlen < len)
      cerr << "output cropped: buffer~ holds " << b.frame_count()
           << " frame(s), need " << (off + len) << " for offset " << off << endl;
    for (int t = 0; t < wlen; t++)
      for (int c = 0; c < wch; c++)
        b.lookup(off + t, c) = m_result[(size_t)c * len + t];
    b.dirty();
  }

  // Stamp the buffer~'s sample rate to the model's, so it plays back at the
  // correct speed/pitch regardless of the patcher's rate — the data is at this
  // rate, and groove~/play~ resample by the buffer's reported sr. (Done outside
  // the lock; sets the rate label only, it does not resize the data.) Skipped in
  // place-mode: we're inserting into an existing timeline, not relabelling it.
  if (!m_have_out_offset && m_out_spec.sample_rate > 0) {
    c74::max::t_buffer_ref *ref =
        c74::max::buffer_ref_new((c74::max::t_object *)(*this), m_buffer.name());
    if (ref) {
      if (auto *buf = c74::max::buffer_ref_getobject(ref)) {
        c74::max::t_atom a;
        c74::max::atom_setfloat(&a, (double)m_out_spec.sample_rate);
        c74::max::object_method_typed((c74::max::t_object *)buf,
                                      c74::max::gensym("sr"), 1, &a, nullptr);
        // Read the rate back: if Max refused or clamped it (e.g. a 50 Hz control
        // curve), say so. The samples are unchanged either way, and a buffer
        // input with "resample": false ignores the buffer~'s rate.
        const double got = c74::max::buffer_getsamplerate(buf);
        if (std::fabs(got - (double)m_out_spec.sample_rate) > 1e-6)
          cwarn << "buffer~ '" << std::string(m_buffer.name().c_str())
                << "' did not take sample rate " << m_out_spec.sample_rate
                << " Hz (reports " << got << " Hz); the data is unchanged"
                << endl;
      }
      c74::max::object_free(ref);
    }
  }

  m_busy.store(false);
  m_done.send("bang");
}

MIN_EXTERNAL(gen);
