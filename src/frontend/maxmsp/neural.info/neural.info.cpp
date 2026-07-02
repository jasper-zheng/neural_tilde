// neural.info — inspect a model's `.json` metadata inside Max. Reads the metadata
// that ships beside a `<stem>.pte` program (it declares the model's full I/O
//
//   [neural.info mymodel.json]      load + emit on instantiate
//   read <path>                     (re)load the metadata and emit
//   bang                            re-emit the currently loaded dictionary

#include "../shared/neural_common.h" 
#include "c74_min.h"

#include <fstream>
#include <sstream>
#include <string>

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

using namespace c74::min;
namespace mx = c74::max;

class info : public object<info> {
public:
  MIN_DIGEST{"Inspect a model's .json metadata"};
  MIN_DESCRIPTION{"Read a neural.* model's .json metadata and output its contents as a Max dictionary"};
  MIN_TAGS{"neural audio synthesis, generative models, metadata"};
  MIN_AUTHOR{"Jasper Shuoyang Zheng"};
  MIN_RELATED{"neural.gen~, neural.live~, neural.tokenizer, dict"};

  info(const atoms &args = {});
  ~info();

  // Single left outlet: the model metadata as a Max dictionary. Sent as
  // `dictionary <name>` so downstream [dict.*] objects read it by name.
  outlet<> m_dict_out{this, "(dictionary) metadata"};

  argument<symbol> path_arg{this, "config.json",
                            "Path to a model .json metadata file."};

  message<> read_msg{
      this, "read", "Load a model .json metadata from a path and emit it.",
      MIN_FUNCTION {
        if (args.size() != 1) {
          cerr << "usage: read <path>" << endl;
          return {};
        }
        load_metadata(std::string(args[0]));
        return {};
      }};

  message<> bang_msg{
      this, "bang", "Emit the currently loaded dictionary.", MIN_FUNCTION {
        if (!m_loaded) {
          cerr << "no metadata loaded" << endl;
          return {};
        }
        m_dict_out.send("dictionary", m_dict_name);
        return {};
      }};

  message<> maxclass_setup{
      this, "maxclass_setup", [this](const atoms &, const int) -> atoms {
        cout << "neural.info " << VERSION << " - Jasper Shuoyang Zheng" << endl;
        return {};
      }};

private:
  bool m_loaded{false};
  // One persistent registered dictionary for the object's lifetime: stable name
  // (good for patch cords), repopulated on each load. Freed in the destructor.
  mx::t_dictionary *m_dict{nullptr};
  symbol m_dict_name;

  void load_metadata(std::string arg);
};

info::info(const atoms &args) {
  // Own one registered dictionary; Max generates a unique name (null in).
  m_dict = mx::dictionary_new();
  mx::t_symbol *nm = nullptr;
  m_dict = mx::dictobj_register(m_dict, &nm);
  m_dict_name = nm;

  if (args.size() > 0)
    load_metadata(std::string(args[0]));
}

info::~info() {
  if (m_dict)
    mx::object_free(m_dict);
}

// Resolve the metadata via Max's search path, read it, parse the JSON into a
// temp dictionary with Max's own parser, clone it into our persistent
// registered dictionary, and emit `dictionary <name>` on the left outlet.
void info::load_metadata(std::string arg) {
  m_loaded = false;

  std::string json_abs = resolve_metadata_json(arg);
  if (json_abs.empty()) {
    cerr << "could not find the JSON " << arg << endl;
    return;
  }

  std::ifstream f(json_abs, std::ios::binary);
  if (!f) {
    cerr << "could not open " << json_abs << endl;
    return;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  std::string contents = ss.str();
  if (contents.empty()) {
    cerr << "JSON is empty: " << json_abs << endl;
    return;
  }

  mx::t_dictionary *parsed = nullptr;
  char err[2048] = {0};
  mx::dictobj_dictionaryfromstring(&parsed, contents.c_str(), /*is_json=*/1, err);
  if (!parsed || err[0] != '\0') {
    cerr << "could not parse JSON " << json_abs << ": " << err << endl;
    if (parsed)
      mx::object_free(parsed);
    return;
  }

  mx::dictionary_clear(m_dict);
  mx::dictionary_clone_to_existing(parsed, m_dict);
  mx::object_free(parsed);

  m_loaded = true;
  cout << "loaded " << json_abs << endl;
  m_dict_out.send("dictionary", m_dict_name);
}

MIN_EXTERNAL(info);
