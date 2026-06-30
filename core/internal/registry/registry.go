// Package registry carrega e navega a hierarquia de ferramentas do SIFT.
//
// A hierarquia tem 3 níveis: categoria -> serviço -> função.
// Cada função carrega um schema compacto:
//
//	p (parâmetros): "<tipo>:<req>:<default>:<descrição>"
//	    req = "n" (necessário/obrigatório) ou "o" (opcional)
//	r (response whitelist): campos que execute_tool deixa passar
package registry

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// Function é o nó-folha: uma ferramenta executável.
type Function struct {
	D    string            `json:"d"`              // descrição (1 frase)
	P    map[string]string `json:"p,omitempty"`    // params compactos
	R    []string          `json:"r,omitempty"`    // response whitelist
	Risk bool              `json:"risk,omitempty"` // ação de alto impacto (enviar, deletar)
}

// Service agrupa funções relacionadas (ex.: gmail).
type Service struct {
	D   string              `json:"d"`
	Fns map[string]Function `json:"fns"`
}

// Category agrupa serviços de um provedor/plataforma (ex.: google_workspace).
type Category struct {
	D        string             `json:"d"`
	Services map[string]Service `json:"services"`
}

// Registry é a árvore inteira.
type Registry map[string]Category

// Param é a forma decodificada de um parâmetro compacto.
type Param struct {
	Name     string
	Type     string
	Required bool
	Default  string
	Desc     string
}

// Load lê o registry de um arquivo JSON.
func Load(path string) (Registry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("lendo registry %q: %w", path, err)
	}
	var r Registry
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("parse registry: %w", err)
	}
	return r, nil
}

// ParseParam decodifica "<tipo>:<req>:<default>:<desc>".
func ParseParam(name, compact string) Param {
	parts := strings.SplitN(compact, ":", 4)
	for len(parts) < 4 {
		parts = append(parts, "")
	}
	return Param{
		Name:     name,
		Type:     parts[0],
		Required: parts[1] == "n",
		Default:  parts[2],
		Desc:     parts[3],
	}
}

// Lookup resolve um path ("cat", "cat.svc" ou "cat.svc.fn").
// Retorna o nó encontrado (Category, Service ou Function) e seu nível.
func (r Registry) Lookup(path string) (node any, level string, err error) {
	path = strings.Trim(path, ". ")
	if path == "" {
		return r, "root", nil
	}
	parts := strings.Split(path, ".")
	cat, ok := r[parts[0]]
	if !ok {
		return nil, "", fmt.Errorf("categoria %q não encontrada", parts[0])
	}
	if len(parts) == 1 {
		return cat, "category", nil
	}
	svc, ok := cat.Services[parts[1]]
	if !ok {
		return nil, "", fmt.Errorf("serviço %q não encontrado em %q", parts[1], parts[0])
	}
	if len(parts) == 2 {
		return svc, "service", nil
	}
	fn, ok := svc.Fns[parts[2]]
	if !ok {
		return nil, "", fmt.Errorf("função %q não encontrada em %q", parts[2], strings.Join(parts[:2], "."))
	}
	if len(parts) == 3 {
		return fn, "function", nil
	}
	return nil, "", fmt.Errorf("path inválido: %q (profundidade máxima é categoria.serviço.função)", path)
}

// Schema devolve uma visão compacta apropriada ao nível do path, pronta para
// o agente (usada por get_tool_schema). Em "" lista categorias; em "cat" lista
// serviços; em "cat.svc" lista funções (compactas); em "cat.svc.fn" devolve a
// função completa.
func (r Registry) Schema(path string) (any, error) {
	node, level, err := r.Lookup(path)
	if err != nil {
		return nil, err
	}
	switch level {
	case "root":
		out := map[string]string{}
		for name, cat := range r {
			out[name] = cat.D
		}
		return map[string]any{"level": "categories", "items": out}, nil
	case "category":
		cat := node.(Category)
		svcs := map[string]string{}
		for name, svc := range cat.Services {
			svcs[name] = svc.D
		}
		return map[string]any{"level": "services", "path": path, "d": cat.D, "services": svcs}, nil
	case "service":
		svc := node.(Service)
		return map[string]any{"level": "functions", "path": path, "d": svc.D, "fns": svc.Fns}, nil
	case "function":
		fn := node.(Function)
		return map[string]any{"level": "function", "path": path, "schema": fn}, nil
	}
	return nil, fmt.Errorf("nível desconhecido: %s", level)
}

// SearchEntry é um nó pesquisável (serviço ou função) com texto para embed.
type SearchEntry struct {
	Path string // ex.: "google_workspace.gmail" ou "google_workspace.gmail.read"
	Kind string // "service" | "function"
	D    string // descrição exibida ao agente
	Text string // texto rico usado para gerar o embedding
}

// SearchEntries achata o registry nos nós pesquisáveis, em ordem estável.
func (r Registry) SearchEntries() []SearchEntry {
	var out []SearchEntry
	cats := sortedKeys(r)
	for _, cn := range cats {
		cat := r[cn]
		for _, sn := range sortedKeys(cat.Services) {
			svc := cat.Services[sn]
			svcPath := cn + "." + sn
			out = append(out, SearchEntry{
				Path: svcPath,
				Kind: "service",
				D:    svc.D,
				Text: fmt.Sprintf("%s %s: %s. %s", cn, sn, svc.D, cat.D),
			})
			for _, fn := range sortedKeys(svc.Fns) {
				f := svc.Fns[fn]
				out = append(out, SearchEntry{
					Path: svcPath + "." + fn,
					Kind: "function",
					D:    f.D,
					Text: fmt.Sprintf("%s %s %s: %s", cn, sn, fn, f.D),
				})
			}
		}
	}
	return out
}

// RiskyPaths devolve o conjunto de paths de função marcados como Risk=true.
func (r Registry) RiskyPaths() map[string]bool {
	out := map[string]bool{}
	for cn, cat := range r {
		for sn, svc := range cat.Services {
			for fn, f := range svc.Fns {
				if f.Risk {
					out[cn+"."+sn+"."+fn] = true
				}
			}
		}
	}
	return out
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
