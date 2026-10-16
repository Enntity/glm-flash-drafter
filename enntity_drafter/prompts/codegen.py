"""Enntity-authored source files for the code-edit prompts.

Builds a small, realistic module (model + validation + in-memory store +
aggregate) for an entity in one of six languages, optionally with one
injected bug. Everything is generated from ``domains`` vocabularies.
"""
from __future__ import annotations

from .domains import fields, title, DOMAINS

LANGS = ("python", "typescript", "go", "rust", "java", "javascript")
EXT = {"python": "py", "typescript": "ts", "go": "go", "rust": "rs", "java": "java", "javascript": "js"}
BUGS = ("filter_inverted", "overwrite_total", "range_exclusive", "get_raises")
BUG_SYMPTOM = {
    "filter_inverted": "`{list_fn}` returns exactly the records that do NOT have the requested {group}",
    "overwrite_total": "`{total_fn}` only reports the last record's {measure} for each {group} instead of the sum",
    "range_exclusive": "validation rejects records whose {range} is exactly {lo}, which should be allowed",
    "get_raises": "looking up an unknown key crashes instead of returning an empty result",
}


def camel(s):
    p = s.split("_")
    return p[0] + "".join(x.capitalize() for x in p[1:])


class Spec:
    def __init__(self, domain, entity):
        self.domain, self.entity = domain, entity
        fs = [f for f in fields(domain, entity) if f.kind not in ("obj", "objlist")]
        self.fields = fs
        self.key = fs[0].name
        enums = [f for f in fs if f.kind == "enum"]
        strs = [f for f in fs[1:] if f.kind in ("str", "city", "name", "company")]
        self.group = (enums or strs or [fs[1]])[0]
        nums = [f for f in fs if f.kind in ("num", "int")]
        self.measure = nums[0] if nums else None
        ints = [f for f in fs if f.kind == "int"]
        self.range = ints[0] if ints else None
        self.T = title(entity)
        self.list_fn = f"list_by_{self.group.name}"
        self.total_fn = f"sum_{self.measure.name}_by_{self.group.name}" if self.measure else None

    def lohi(self):
        return tuple(map(int, self.range.arg.split(","))) if self.range else (0, 0)


def ptype(f, lang):
    k = f.kind
    num = k == "num"
    if lang == "python":
        return {"int": "int", "num": "float", "bool": "bool", "list": "list[str]"}.get(k, "str")
    if lang in ("typescript",):
        return {"int": "number", "num": "number", "bool": "boolean", "list": "string[]"}.get(k, "string")
    if lang == "go":
        return {"int": "int", "num": "float64", "bool": "bool", "list": "[]string"}.get(k, "string")
    if lang == "rust":
        return {"int": "i64", "num": "f64", "bool": "bool", "list": "Vec<String>"}.get(k, "String")
    if lang == "java":
        return {"int": "int", "num": "double", "bool": "boolean", "list": "List<String>"}.get(k, "String")
    return "any" if num else ""


def python_src(s: Spec, bug=None):
    lo, hi = s.lohi()
    g, m, key = s.group.name, s.measure, s.key
    L = [f'"""{s.T} records for {DOMAINS[s.domain][0]}."""', "from __future__ import annotations", "",
         "from dataclasses import dataclass", "from typing import Optional", ""]
    if s.group.kind == "enum":
        L.append(f"VALID_{g.upper()} = {{{', '.join(repr(v) for v in s.group.arg.split('|'))}}}")
    if s.range:
        L += [f"MIN_{s.range.name.upper()} = {lo}", f"MAX_{s.range.name.upper()} = {hi}"]
    L += ["", "", "@dataclass", f"class {s.T}:"]
    for f in s.fields:
        t = ptype(f, "python")
        L.append(f"    {f.name}: {t}")
    L += ["", "", f"def validate_{s.entity}(item: {s.T}) -> list[str]:", "    errors = []",
          f"    if not item.{key}:", f'        errors.append("{key} is required")']
    if s.group.kind == "enum":
        L += [f"    if item.{g} not in VALID_{g.upper()}:", f'        errors.append(f"invalid {g}: {{item.{g}}}")']
    if s.range:
        r = s.range.name
        op = "<" if bug == "range_exclusive" else "<="
        L += [f"    if not (MIN_{r.upper()} {op} item.{r} <= MAX_{r.upper()}):",
              f'        errors.append(f"{r} out of range: {{item.{r}}}")']
    L += ["    return errors", "", "", f"class {s.T}Store:", "    def __init__(self) -> None:",
          f"        self._items: dict[str, {s.T}] = {{}}", "",
          f"    def add(self, item: {s.T}) -> None:", f"        errors = validate_{s.entity}(item)", "        if errors:",
          '            raise ValueError("; ".join(errors))', f"        self._items[item.{key}] = item", "",
          f"    def get(self, key: str) -> Optional[{s.T}]:",
          "        return self._items[key]" if bug == "get_raises" else "        return self._items.get(key)", "",
          f"    def {s.list_fn}(self, {g}: str) -> list[{s.T}]:",
          f"        return [i for i in self._items.values() if i.{g} {'!=' if bug == 'filter_inverted' else '=='} {g}]", "",
          "    def remove(self, key: str) -> bool:", "        return self._items.pop(key, None) is not None", ""]
    if m:
        L += ["", f"def {s.total_fn}(items: list[{s.T}]) -> dict[str, float]:", "    totals: dict[str, float] = {}",
              "    for i in items:",
              f"        totals[i.{g}] = i.{m.name}" if bug == "overwrite_total"
              else f"        totals[i.{g}] = totals.get(i.{g}, 0) + i.{m.name}", "    return totals", ""]
    L += ["", f"def {s.entity}_from_dict(d: dict) -> {s.T}:",
          f"    return {s.T}(" + ", ".join(f"{f.name}=d[{f.name!r}]" for f in s.fields) + ")", ""]
    return "\n".join(L)


def ts_src(s: Spec, bug=None, js=False):
    lo, hi = s.lohi()
    g, m, key = s.group.name, s.measure, s.key
    ty = (lambda t: "") if js else (lambda t: f": {t}")
    L = [f"// {s.T} records for {DOMAINS[s.domain][0]}.", ""]
    if not js:
        L.append(f"export interface {s.T} {{")
        L += [f"  {camel(f.name)}: {ptype(f, 'typescript')};" for f in s.fields]
        L += ["}", ""]
    if s.group.kind == "enum":
        L.append(f"export const VALID_{g.upper()} = new Set{'' if js else '<string>'}([{', '.join(repr(v) for v in s.group.arg.split('|'))}]);")
    if s.range:
        L += [f"export const MIN_{s.range.name.upper()} = {lo};", f"export const MAX_{s.range.name.upper()} = {hi};"]
    k, gc = camel(key), camel(g)
    L += ["", f"export function validate{s.T}(item{ty(s.T)}){ty('string[]')} {{", f"  const errors{ty('string[]')} = [];",
          f"  if (!item.{k}) errors.push('{key} is required');"]
    if s.group.kind == "enum":
        L.append(f"  if (!VALID_{g.upper()}.has(item.{gc})) errors.push(`invalid {g}: ${{item.{gc}}}`);")
    if s.range:
        r, rc = s.range.name, camel(s.range.name)
        op = "<" if bug == "range_exclusive" else "<="
        L.append(f"  if (!(MIN_{r.upper()} {op} item.{rc} && item.{rc} <= MAX_{r.upper()})) errors.push(`{r} out of range: ${{item.{rc}}}`);")
    L += ["  return errors;", "}", "", f"export class {s.T}Store {{",
          f"  {'' if js else 'private '}items = new Map{'' if js else f'<string, {s.T}>'}();", "",
          f"  add(item{ty(s.T)}){ty('void')} {{", f"    const errors = validate{s.T}(item);",
          "    if (errors.length > 0) throw new Error(errors.join('; '));", f"    this.items.set(item.{k}, item);", "  }", "",
          f"  get(key{ty('string')}){ty(s.T + ' | undefined')} {{"]
    if bug == "get_raises":
        L += ["    const item = this.items.get(key);", "    if (!item) throw new Error(`unknown key ${key}`);", "    return item;"]
    else:
        L.append("    return this.items.get(key);")
    L += ["  }", "", f"  {camel(s.list_fn)}({gc}{ty('string')}){ty(s.T + '[]')} {{",
          f"    return [...this.items.values()].filter((i) => i.{gc} {'!==' if bug == 'filter_inverted' else '==='} {gc});",
          "  }", "", f"  remove(key{ty('string')}){ty('boolean')} {{", "    return this.items.delete(key);", "  }", "}", ""]
    if m:
        mc = camel(m.name)
        L += [f"export function {camel(s.total_fn)}(items{ty(s.T + '[]')}){ty('Record<string, number>')} {{",
              f"  const totals{ty('Record<string, number>')} = {{}};", "  for (const i of items) {",
              f"    totals[i.{gc}] = i.{mc};" if bug == "overwrite_total" else f"    totals[i.{gc}] = (totals[i.{gc}] ?? 0) + i.{mc};",
              "  }", "  return totals;", "}", ""]
    return "\n".join(L)


def go_src(s: Spec, bug=None):
    lo, hi = s.lohi()
    g, m, key = s.group.name, s.measure, s.key
    G = lambda n: title(n).replace("Id", "ID")
    L = [f"// Package {s.entity.replace('_', '')} manages {s.T} records for {DOMAINS[s.domain][0]}.",
         f"package {s.entity.replace('_', '')}", "", "import (", '\t"errors"', '\t"fmt"', '\t"strings"', ")", ""]
    if s.group.kind == "enum":
        L.append(f"var valid{G(g)} = map[string]bool{{" + ", ".join(f'"{v}": true' for v in s.group.arg.split("|")) + "}")
    if s.range:
        L += ["const (", f"\tmin{G(s.range.name)} = {lo}", f"\tmax{G(s.range.name)} = {hi}", ")"]
    L += ["", "// ErrNotFound is returned when a key is unknown.", 'var ErrNotFound = errors.New("not found")', "",
          f"type {s.T} struct {{"]
    L += [f"\t{G(f.name)} {ptype(f, 'go')} `json:\"{f.name}\"`" for f in s.fields]
    L += ["}", "", f"func Validate(item {s.T}) error {{", "\tvar problems []string",
          f"\tif item.{G(key)} == \"\" {{", f'\t\tproblems = append(problems, "{key} is required")', "\t}"]
    if s.group.kind == "enum":
        L += [f"\tif !valid{G(g)}[item.{G(g)}] {{", f'\t\tproblems = append(problems, fmt.Sprintf("invalid {g}: %s", item.{G(g)}))', "\t}"]
    if s.range:
        r = G(s.range.name)
        op = "<=" if bug == "range_exclusive" else "<"
        L += [f"\tif item.{r} {op} min{r} || item.{r} > max{r} {{",
              f'\t\tproblems = append(problems, fmt.Sprintf("{s.range.name} out of range: %d", item.{r}))', "\t}"]
    L += ["\tif len(problems) > 0 {", '\t\treturn errors.New(strings.Join(problems, "; "))', "\t}", "\treturn nil", "}", "",
          f"type Store struct {{", f"\titems map[string]{s.T}", "}", "",
          f"func NewStore() *Store {{", f"\treturn &Store{{items: make(map[string]{s.T})}}", "}", "",
          f"func (s *Store) Add(item {s.T}) error {{", "\tif err := Validate(item); err != nil {", "\t\treturn err", "\t}",
          f"\ts.items[item.{G(key)}] = item", "\treturn nil", "}", "",
          f"func (s *Store) Get(key string) ({s.T}, error) {{"]
    if bug == "get_raises":
        L += ["\titem := s.items[key]", "\treturn item, nil"]
    else:
        L += ["\titem, ok := s.items[key]", "\tif !ok {", f"\t\treturn {s.T}{{}}, ErrNotFound", "\t}", "\treturn item, nil"]
    L += ["}", "", f"func (s *Store) {G(s.list_fn)}({camel(g)} string) []{s.T} {{", f"\tvar out []{s.T}",
          "\tfor _, item := range s.items {", f"\t\tif item.{G(g)} {'!=' if bug == 'filter_inverted' else '=='} {camel(g)} {{",
          "\t\t\tout = append(out, item)", "\t\t}", "\t}", "\treturn out", "}", ""]
    if m:
        conv = f"float64(item.{G(m.name)})" if m.kind == "int" else f"item.{G(m.name)}"
        L += [f"func {G(s.total_fn)}(items []{s.T}) map[string]float64 {{", "\ttotals := make(map[string]float64)",
              "\tfor _, item := range items {",
              f"\t\ttotals[item.{G(g)}] = {conv}" if bug == "overwrite_total" else f"\t\ttotals[item.{G(g)}] += {conv}",
              "\t}", "\treturn totals", "}", ""]
    return "\n".join(L)


def rust_src(s: Spec, bug=None):
    lo, hi = s.lohi()
    g, m, key = s.group.name, s.measure, s.key
    L = [f"//! {s.T} records for {DOMAINS[s.domain][0]}.", "", "use std::collections::HashMap;", ""]
    if s.group.kind == "enum":
        L.append(f"pub const VALID_{g.upper()}: &[&str] = &[{', '.join(chr(34) + v + chr(34) for v in s.group.arg.split('|'))}];")
    if s.range:
        L += [f"pub const MIN_{s.range.name.upper()}: i64 = {lo};", f"pub const MAX_{s.range.name.upper()}: i64 = {hi};"]
    L += ["", "#[derive(Debug, Clone, PartialEq)]", f"pub struct {s.T} {{"]
    L += [f"    pub {f.name}: {ptype(f, 'rust')}," for f in s.fields]
    L += ["}", "", f"pub fn validate(item: &{s.T}) -> Result<(), String> {{", "    let mut problems = Vec::new();",
          f"    if item.{key}.is_empty() {{", f'        problems.push("{key} is required".to_string());', "    }"]
    if s.group.kind == "enum":
        L += [f"    if !VALID_{g.upper()}.contains(&item.{g}.as_str()) {{",
              f'        problems.push(format!("invalid {g}: {{}}", item.{g}));', "    }"]
    if s.range:
        r = s.range.name
        cond = f"item.{r} <= MIN_{r.upper()}" if bug == "range_exclusive" else f"item.{r} < MIN_{r.upper()}"
        L += [f"    if {cond} || item.{r} > MAX_{r.upper()} {{",
              f'        problems.push(format!("{r} out of range: {{}}", item.{r}));', "    }"]
    L += ["    if problems.is_empty() {", "        Ok(())", "    } else {", '        Err(problems.join("; "))', "    }", "}", "",
          "#[derive(Default)]", f"pub struct {s.T}Store {{", f"    items: HashMap<String, {s.T}>,", "}", "",
          f"impl {s.T}Store {{", "    pub fn new() -> Self {", "        Self::default()", "    }", "",
          f"    pub fn add(&mut self, item: {s.T}) -> Result<(), String> {{", "        validate(&item)?;",
          f"        self.items.insert(item.{key}.clone(), item);", "        Ok(())", "    }", ""]
    if bug == "get_raises":
        L += [f"    pub fn get(&self, key: &str) -> &{s.T} {{", "        &self.items[key]", "    }"]
    else:
        L += [f"    pub fn get(&self, key: &str) -> Option<&{s.T}> {{", "        self.items.get(key)", "    }"]
    L += ["", f"    pub fn {s.list_fn}(&self, {g}: &str) -> Vec<&{s.T}> {{",
          f"        self.items.values().filter(|i| i.{g} {'!=' if bug == 'filter_inverted' else '=='} {g}).collect()", "    }", "}", ""]
    if m:
        conv = f"i.{m.name} as f64" if m.kind == "int" else f"i.{m.name}"
        L += [f"pub fn {s.total_fn}(items: &[{s.T}]) -> HashMap<String, f64> {{", "    let mut totals = HashMap::new();",
              "    for i in items {",
              f"        totals.insert(i.{g}.clone(), {conv});" if bug == "overwrite_total"
              else f"        *totals.entry(i.{g}.clone()).or_insert(0.0) += {conv};", "    }", "    totals", "}", ""]
    return "\n".join(L)


def java_src(s: Spec, bug=None):
    lo, hi = s.lohi()
    g, m, key = s.group.name, s.measure, s.key
    J = camel
    L = [f"package com.example.{s.domain};", "", "import java.util.*;", "import java.util.stream.Collectors;", "",
         f"/** {s.T} records for {DOMAINS[s.domain][0]}. */", f"public final class {s.T}Store {{", ""]
    if s.group.kind == "enum":
        L.append(f"    public static final Set<String> VALID_{g.upper()} = Set.of({', '.join(chr(34) + v + chr(34) for v in s.group.arg.split('|'))});")
    if s.range:
        L += [f"    public static final int MIN_{s.range.name.upper()} = {lo};", f"    public static final int MAX_{s.range.name.upper()} = {hi};"]
    L += ["", f"    public record {s.T}(" + ", ".join(f"{ptype(f, 'java')} {J(f.name)}" for f in s.fields) + ") {}", "",
          f"    private final Map<String, {s.T}> items = new HashMap<>();", "",
          f"    public static List<String> validate({s.T} item) {{", "        List<String> errors = new ArrayList<>();",
          f"        if (item.{J(key)}() == null || item.{J(key)}().isEmpty()) errors.add(\"{key} is required\");"]
    if s.group.kind == "enum":
        L.append(f"        if (!VALID_{g.upper()}.contains(item.{J(g)}())) errors.add(\"invalid {g}: \" + item.{J(g)}());")
    if s.range:
        r = s.range.name
        op = "<=" if bug == "range_exclusive" else "<"
        L.append(f"        if (item.{J(r)}() {op} MIN_{r.upper()} || item.{J(r)}() > MAX_{r.upper()}) errors.add(\"{r} out of range: \" + item.{J(r)}());")
    L += ["        return errors;", "    }", "", f"    public void add({s.T} item) {{", "        List<String> errors = validate(item);",
          "        if (!errors.isEmpty()) throw new IllegalArgumentException(String.join(\"; \", errors));",
          f"        items.put(item.{J(key)}(), item);", "    }", ""]
    if bug == "get_raises":
        L += [f"    public {s.T} get(String key) {{", "        return Objects.requireNonNull(items.get(key));", "    }"]
    else:
        L += [f"    public Optional<{s.T}> get(String key) {{", "        return Optional.ofNullable(items.get(key));", "    }"]
    L += ["", f"    public List<{s.T}> {J(s.list_fn)}(String {J(g)}) {{",
          f"        return items.values().stream().filter(i -> {'!' if bug == 'filter_inverted' else ''}i.{J(g)}().equals({J(g)})).collect(Collectors.toList());",
          "    }", ""]
    if m:
        L += [f"    public static Map<String, Double> {J(s.total_fn)}(List<{s.T}> list) {{",
              "        Map<String, Double> totals = new HashMap<>();", f"        for ({s.T} i : list) {{",
              f"            totals.put(i.{J(g)}(), (double) i.{J(m.name)}());" if bug == "overwrite_total"
              else f"            totals.merge(i.{J(g)}(), (double) i.{J(m.name)}(), Double::sum);", "        }",
              "        return totals;", "    }"]
    L += ["}", ""]
    return "\n".join(L)


def source(lang, s: Spec, bug=None):
    return {"python": python_src, "go": go_src, "rust": rust_src, "java": java_src,
            "typescript": ts_src, "javascript": lambda s, b: ts_src(s, b, js=True)}[lang](s, bug)


def names(lang, s: Spec):
    """Language-specific identifiers for the store's functions."""
    if lang == "python":
        return {"list_fn": s.list_fn, "total_fn": s.total_fn, "validate_fn": f"validate_{s.entity}", "get_fn": "get"}
    if lang in ("typescript", "javascript"):
        return {"list_fn": camel(s.list_fn), "total_fn": camel(s.total_fn) if s.total_fn else None,
                "validate_fn": f"validate{s.T}", "get_fn": "get"}
    if lang == "go":
        G = lambda n: title(n).replace("Id", "ID")
        return {"list_fn": G(s.list_fn), "total_fn": G(s.total_fn) if s.total_fn else None, "validate_fn": "Validate", "get_fn": "Get"}
    if lang == "rust":
        return {"list_fn": s.list_fn, "total_fn": s.total_fn, "validate_fn": "validate", "get_fn": "get"}
    return {"list_fn": camel(s.list_fn), "total_fn": camel(s.total_fn) if s.total_fn else None, "validate_fn": "validate", "get_fn": "get"}


def path_for(lang, s: Spec):
    base = {"python": f"{s.domain}/{s.entity}_store.py", "typescript": f"src/{s.domain}/{camel(s.entity)}Store.ts",
            "javascript": f"src/{s.domain}/{camel(s.entity)}Store.js", "go": f"internal/{s.entity.replace('_', '')}/store.go",
            "rust": f"src/{s.entity}.rs", "java": f"src/main/java/com/example/{s.domain}/{s.T}Store.java"}
    return base[lang]
