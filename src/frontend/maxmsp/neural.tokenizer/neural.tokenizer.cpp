// neural.tokenizer — HuggingFace text tokenizer in a Max object. Loads a
// `tokenizer.json` (tokenizers-cpp / Rust core) and turns a text prompt into the
// `input_ids` (int64) + `attention_mask` (int32) a generative model expects,
// emitting each as a single-key Max dictionary on its OWN outlet — outlet 0 =
// {ids_key: input_ids}, outlet 1 = {mask_key: attention_mask} — to wire into the
// matching condition inlets of [neural.gen~]. This is the ONLY external that
// pulls in the tokenizer/Rust, so the runner and signal objects stay lean.
//
//   [neural.tokenizer <tokenizer.json>]
//   prompt <text...>   tokenize + output two dictionaries  reload <path> reload
//   @max_length 256  @pad_token <pad>  @padding_side right
//   @ids_key input_ids  @mask_key attention_mask   (keys = the model input names)
//
// Negative prompt (classifier-free guidance): use a second instance configured
// with @ids_key neg_input_ids @mask_key neg_attention_mask, wired into the
// model's neg_input_ids / neg_attention_mask inlets.

#include "../../../tokenizer/tokenizer.h"
#include "../../../tokenizer/tokenizer_config.h"
#include "c74_min.h"

#include <string>
#include <vector>

#ifndef VERSION
#define VERSION "UNDEFINED"
#endif

using namespace c74::min;

class tokenizer_obj : public object<tokenizer_obj> {
public:
  MIN_DIGEST{"HuggingFace text tokenizer"};
  MIN_DESCRIPTION{"Tokenize a text prompt into integer IDs and attention mask, "
                  "output as two single-key dictionaries (one per outlet). Uses "
                  "HuggingFace tokenizers."};
  MIN_TAGS{"neural audio synthesis, generative models, tokenizer"};
  MIN_AUTHOR{"Jasper Shuoyang Zheng"};
  MIN_RELATED{"neural.live~, neural.gen~"};

  tokenizer_obj(const atoms &args = {});
  ~tokenizer_obj();

  // Two single-key dictionaries: outlet 0 = {ids_key: input_ids}, outlet 1 =
  // {mask_key: attention_mask}. Declaration order fixes the outlet order (ids
  // leftmost). neural.gen~ matches each by its key name onto the right condition.
  outlet<> m_ids_out{this, "(dictionary) token_ids"};
  outlet<> m_mask_out{this, "(dictionary) attention_mask"};

  // ---- attributes ---------------------------------------------------------
  attribute<int> max_length{this, "max_length", 256,
                            description{"Pad/truncate length."}};
  attribute<symbol> pad_token{this, "pad_token", "<pad>",
                              description{"Token used to pad to max_length."}};
  attribute<symbol> padding_side{
      this, "padding_side", "right",
      description{"Pad side: \"right\" or \"left\"."}};
  attribute<symbol> ids_key{
      this, "ids_key", "token_ids",
      description{"Dictionary key for the token ids (to match with neural.gen~ / live~)."}};
  attribute<symbol> mask_key{
      this, "mask_key", "attention_mask",
      description{"Dictionary key for the attention mask (to match with neural.gen~ / live~)."}};

  argument<symbol> path_arg{
      this, "tokenizer.json",
      "Path to a HuggingFace *.tokenizer.json "};

  // ---- messages -----------------------------------------------------------
  message<> prompt_msg{
      this, "prompt", "Tokenize the text and output a dictionary.",
      MIN_FUNCTION {
        std::string p;
        for (auto i = 0; i < args.size(); i++) {
          if (i)
            p += " ";
          p += std::string(args[i]);
        }
        tokenize(p);
        return {};
      }};

  message<> reload_msg{
      this, "reload", "Load a tokenizer.json from a path.", MIN_FUNCTION {
        if (args.size() != 1) {
          cerr << "usage: reload <path>" << endl;
          return {};
        }
        load_tokenizer(std::string(args[0]));
        return {};
      }};

  message<> maxclass_setup{
      this, "maxclass_setup",
      [this](const atoms &args, const int inlet) -> atoms {
        cout << "neural.tokenizer " << VERSION << " - Jasper Shuoyang Zheng"
             << endl;
        return {};
      }};

private:
  NnTokenizer m_tok;
  bool m_loaded{false};
  c74::max::t_dictionary *m_ids_dict{nullptr};
  c74::max::t_dictionary *m_mask_dict{nullptr};
  symbol m_ids_dict_name;
  symbol m_mask_dict_name;

  void load_tokenizer(std::string arg_path);
  void apply_config(const TokenizerConfig &cfg);
  void tokenize(const std::string &prompt);
};

// Push a loaded config onto the attributes (a live @attr/message still overrides).
void tokenizer_obj::apply_config(const TokenizerConfig &cfg) {
  max_length = cfg.max_length;
  pad_token = symbol(cfg.pad_token);
  padding_side = symbol(cfg.padding_side);
  ids_key = symbol(cfg.ids_key);
  mask_key = symbol(cfg.mask_key);
}

void tokenizer_obj::load_tokenizer(std::string arg_path) {
  m_loaded = false;

  // Bare-name convenience: an arg without a .json extension is treated as a
  // model stem -> load "<stem>.tokenizer.json". Manual mode's auto-config
  // lookup below then picks up "<stem>.tokenizer.config.json" automatically.
  const std::string json_suffix = ".json";
  bool has_json = arg_path.size() >= json_suffix.size() &&
                  arg_path.compare(arg_path.size() - json_suffix.size(),
                                   json_suffix.size(), json_suffix) == 0;
  if (!has_json)
    arg_path += ".tokenizer.json";

  try {
    path resolved(arg_path); // resolve via Max search path
    std::string abs = std::string(resolved);

    // Config mode: arg is "<...>.config.json" describing the tokenizer + settings.
    // (Suffix check avoids parsing the multi-MB tokenizer.json just to detect.)
    const std::string suffix = ".config.json";
    bool is_config = abs.size() >= suffix.size() &&
                     abs.compare(abs.size() - suffix.size(), suffix.size(),
                                 suffix) == 0;
    if (is_config) {
      TokenizerConfig cfg;
      std::string tok_json;
      if (!load_tokenizer_config(abs, cfg, tok_json)) {
        cerr << "could not read tokenizer config " << abs << endl;
        return;
      }
      if (!m_tok.load(tok_json)) {
        cerr << "could not load tokenizer " << tok_json << endl;
        return;
      }
      apply_config(cfg);
      m_loaded = true;
      cout << "loaded config " << abs << " (tokenizer: " << tok_json << ")"
           << endl;
      return;
    }

    // Manual mode: arg is a tokenizer.json directly. Load it, then auto-look for
    // its config by name ("<X>.json" -> "<X>.config.json"); if none is
    // found, post an error and fall back to the default/attribute settings.
    if (!m_tok.load(abs)) {
      cerr << "could not load " << abs << endl;
      return;
    }
    m_loaded = true;

    std::string cand;
    if (abs.size() >= 5 && abs.compare(abs.size() - 5, 5, ".json") == 0)
      cand = abs.substr(0, abs.size() - 5) + ".config.json";

    TokenizerConfig cfg;
    std::string tok_json_unused;
    if (!cand.empty() && load_tokenizer_config(cand, cfg, tok_json_unused)) {
      apply_config(cfg);
      cout << "loaded " << abs << " (auto-config: " << cand << ")" << endl;
    } else {
      cerr << "no tokenizer config found (" << cand
           << "); using default configuration (max_length=" << (int)max_length
           << ", pad_token=" << std::string(pad_token.get())
           << ", padding_side=" << std::string(padding_side.get()) << ")"
           << endl;
      cout << "loaded " << abs << endl;
    }
  } catch (...) {
    cerr << "could not resolve " << arg_path << endl;
  }
}

void tokenizer_obj::tokenize(const std::string &prompt) {
  if (!m_loaded) {
    cerr << "no tokenizer loaded" << endl;
    return;
  }
  std::vector<int64_t> ids;
  std::vector<int32_t> mask;
  if (!m_tok.encode(prompt, max_length, std::string(pad_token.get()),
                    std::string(padding_side.get()), ids, mask)) {
    cerr << "tokenization failed" << endl;
    return;
  }

  // Repopulate the two persistent registered dictionaries and send each on its
  // outlet. Each holds a single key — the model input name (ids_key/mask_key) —
  // so neural.gen~ matches it by name onto the right condition input.
  c74::max::dictionary_clear(m_ids_dict);
  std::vector<c74::max::t_atom> ids_atoms(ids.size());
  for (size_t i = 0; i < ids.size(); i++)
    c74::max::atom_setlong(&ids_atoms[i], (c74::max::t_atom_long)ids[i]);
  c74::max::dictionary_appendatoms(m_ids_dict,
                                   (c74::max::t_symbol *)ids_key.get(),
                                   (long)ids_atoms.size(), ids_atoms.data());

  c74::max::dictionary_clear(m_mask_dict);
  std::vector<c74::max::t_atom> mask_atoms(mask.size());
  for (size_t i = 0; i < mask.size(); i++)
    c74::max::atom_setlong(&mask_atoms[i], (c74::max::t_atom_long)mask[i]);
  c74::max::dictionary_appendatoms(m_mask_dict,
                                   (c74::max::t_symbol *)mask_key.get(),
                                   (long)mask_atoms.size(), mask_atoms.data());

  // Right-to-left: mask (outlet 1) then ids (outlet 0). Order is immaterial to
  // neural.gen~ (it caches both and waits for `generate`).
  m_mask_out.send("dictionary", m_mask_dict_name);
  m_ids_out.send("dictionary", m_ids_dict_name);
}

tokenizer_obj::tokenizer_obj(const atoms &args) {
  // Own two registered dictionaries for the object's lifetime (reused each
  // prompt): one carries the ids, the other the mask, each under its key.
  auto make_dict = [](symbol &name_out) -> c74::max::t_dictionary * {
    c74::max::t_dictionary *d = c74::max::dictionary_new();
    c74::max::t_symbol *nm = nullptr; // null => Max generates a unique name
    d = c74::max::dictobj_register(d, &nm);
    name_out = nm;
    return d;
  };
  m_ids_dict = make_dict(m_ids_dict_name);
  m_mask_dict = make_dict(m_mask_dict_name);

  if (args.size() > 0)
    load_tokenizer(std::string(args[0]));
}

tokenizer_obj::~tokenizer_obj() {
  if (m_ids_dict)
    c74::max::object_free(m_ids_dict);
  if (m_mask_dict)
    c74::max::object_free(m_mask_dict);
}

MIN_EXTERNAL(tokenizer_obj);
