#pragma once
// Dynamic Max attributes for the live objects (neural.live~, mc.neural.live~,
// mcs.neural.live~). One Max instance attribute per Attribute-role input the
// model's method declares, created from the metadata at load. The model consumes
// each as a trailing scalar input to its forward; the host owns the value.
//
// LiveAttrs::slots holds the metadata (name/range/default/description);
// LiveAttrs::live is the parallel live value, written by the attribute setter
// (control thread) and read by the perform thread, which hands it straight to
// Backend::perform as `attr_values` (declared order). The per-float race with the
// setter is benign (a value-copy was never atomic either).
//
// The owning object class T must expose a public `LiveAttrs m_attrs` member; the
// raw Max accessors recover the instance through the min wrapper.

#include "../../../backend/backend.h"
#include "c74_min.h"
#include <algorithm>
#include <string>
#include <vector>

struct LiveAttrs {
  struct Slot {
    std::string name;
    double min, max, def;
    std::string description;
  };
  std::vector<Slot> slots;
  std::vector<float> live; // parallel to slots; passed to Backend::perform

  bool is_attr(const std::string &name) const {
    for (const auto &s : slots)
      if (s.name == name)
        return true;
    return false;
  }

  // Setter side: store the (clamped) value into the live buffer.
  void store(const char *name, double value) {
    for (size_t i = 0; i < slots.size(); i++)
      if (slots[i].name == name) {
        const Slot &s = slots[i];
        if (s.max > s.min)
          value = std::min(std::max(value, s.min), s.max); // defensive clamp
        live[i] = (float)value;
        return;
      }
  }

  double load(const char *name) const {
    for (size_t i = 0; i < slots.size(); i++)
      if (slots[i].name == name)
        return (double)live[i];
    return 0.0;
  }
};

template <class T>
c74::max::t_max_err live_attr_set(c74::max::t_object *x, c74::max::t_object *attr,
                                  long ac, c74::max::t_atom *av) {
  if (!x || ac < 1 || !av)
    return 0;
  T &self = reinterpret_cast<c74::min::minwrap<T> *>(x)->m_min_object;
  auto *nm = (c74::max::t_symbol *)c74::max::object_method(
      attr, c74::max::gensym("getname"));
  if (nm)
    self.m_attrs.store(nm->s_name, c74::max::atom_getfloat(av));
  return 0;
}

template <class T>
c74::max::t_max_err live_attr_get(c74::max::t_object *x, c74::max::t_object *attr,
                                  long *ac, c74::max::t_atom **av) {
  T &self = reinterpret_cast<c74::min::minwrap<T> *>(x)->m_min_object;
  auto *nm = (c74::max::t_symbol *)c74::max::object_method(
      attr, c74::max::gensym("getname"));
  double v = nm ? self.m_attrs.load(nm->s_name) : 0.0;
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

// Re-derive the dynamic attributes from the loaded model's method: drop any
// previous model's attributes, then add one float64 attribute per Attribute-role
// input (clip filter from min/max, metadata default, description as label).
// Call only while the perform thread is quiesced (ctor / reload).
template <class T>
void rebuild_live_attrs(c74::max::t_object *self, LiveAttrs &attrs, Backend *model,
                        const std::string &method) {
  for (const auto &s : attrs.slots) // drop a previous model's attributes
    c74::max::object_deleteattr(self, c74::max::gensym(s.name.c_str()));
  attrs.slots.clear();
  attrs.live.clear();
  if (!model || !model->has_metadata())
    return;

  const auto specs = model->method_attributes(method);
  // Populate slots + live values first so the setter (fired by the default-set
  // below, and by @box-args after the ctor) can resolve each slot by name.
  for (const auto &a : specs) {
    attrs.slots.push_back({a.name, a.min, a.max, a.def, a.description});
    attrs.live.push_back((float)a.def);
  }

  for (const auto &a : specs) {
    c74::max::t_object *attr = (c74::max::t_object *)c74::max::attribute_new(
        a.name.c_str(), c74::max::gensym("float64"), 0,
        (c74::max::method)&live_attr_get<T>, (c74::max::method)&live_attr_set<T>);
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

// Set a dynamic attribute by name through Max (so the clip filter applies before
// the setter stores it). Returns false if `name` is not one of the attributes.
inline bool set_live_attr(c74::max::t_object *self, const LiveAttrs &attrs,
                          const std::string &name, double value) {
  if (!attrs.is_attr(name))
    return false;
  c74::max::object_attr_setfloat(self, c74::max::gensym(name.c_str()), value);
  return true;
}
