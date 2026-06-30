// Package gateway implementa o "cérebro" do SIFT: as 3 meta-ferramentas que o
// agente enxerga (search_tools, get_tool_schema, execute_tool), mais a
// filtragem de resposta, o cache de schemas e o índice semântico de descoberta.
//
// O agente NUNCA vê o catálogo inteiro — só as 3 meta-tools. Ele descobre o
// resto navegando: Search -> Inspect -> Trigger (com Filter automático).
package gateway

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"sync"

	"sift/internal/embed"
	"sift/internal/registry"
	"sift/internal/toon"
)

// indexedEntry é um nó pesquisável já com seu vetor de embedding.
type indexedEntry struct {
	registry.SearchEntry
	Vec []float32
}

// Gateway agrega tudo o que as meta-tools precisam.
type Gateway struct {
	reg   registry.Registry
	embed *embed.Client

	mu          sync.RWMutex
	index       []indexedEntry
	schemaCache map[string]any
	toonCache   map[string]string
}

// New constrói o gateway. O índice semântico é montado depois, em BuildIndex.
func New(reg registry.Registry, ec *embed.Client) *Gateway {
	return &Gateway{
		reg:         reg,
		embed:       ec,
		schemaCache: map[string]any{},
		toonCache:   map[string]string{},
	}
}

// Registry expõe o registry subjacente (usado por benchmarks/inspeção).
func (g *Gateway) Registry() registry.Registry { return g.reg }

// BuildIndex embeda todos os nós pesquisáveis (serviços e funções) numa única
// chamada em lote ao sidecar. Chame uma vez no startup.
func (g *Gateway) BuildIndex(ctx context.Context) error {
	entries := g.reg.SearchEntries()
	texts := make([]string, len(entries))
	for i, e := range entries {
		texts[i] = e.Text
	}
	vecs, err := g.embed.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("montando índice semântico: %w", err)
	}
	if len(vecs) != len(entries) {
		return fmt.Errorf("índice: esperava %d vetores, recebi %d", len(entries), len(vecs))
	}
	idx := make([]indexedEntry, len(entries))
	for i := range entries {
		idx[i] = indexedEntry{SearchEntry: entries[i], Vec: vecs[i]}
	}
	g.mu.Lock()
	g.index = idx
	g.mu.Unlock()
	return nil
}

// SearchResult é um item devolvido por search_tools.
type SearchResult struct {
	Path  string  `json:"path"`
	Kind  string  `json:"kind"`
	D     string  `json:"d"`
	Score float64 `json:"score"`
}

// SearchTools faz a descoberta semântica: embeda a query e devolve os nós mais
// próximos (serviços e funções), ordenados por similaridade.
func (g *Gateway) SearchTools(ctx context.Context, query string, topK int) ([]SearchResult, error) {
	if topK <= 0 {
		topK = 5
	}
	g.mu.RLock()
	idx := g.index
	g.mu.RUnlock()
	if len(idx) == 0 {
		return nil, fmt.Errorf("índice semântico vazio (chame BuildIndex)")
	}

	qv, err := g.embed.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	q := qv[0]

	results := make([]SearchResult, len(idx))
	for i, e := range idx {
		results[i] = SearchResult{
			Path:  e.Path,
			Kind:  e.Kind,
			D:     e.D,
			Score: embed.Cosine(q, e.Vec),
		}
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

// GetToolSchema devolve a visão compacta do path, com cache.
func (g *Gateway) GetToolSchema(path string) (any, error) {
	g.mu.RLock()
	if cached, ok := g.schemaCache[path]; ok {
		g.mu.RUnlock()
		return cached, nil
	}
	g.mu.RUnlock()

	schema, err := g.reg.Schema(path)
	if err != nil {
		return nil, err
	}
	g.mu.Lock()
	g.schemaCache[path] = schema
	g.mu.Unlock()
	return schema, nil
}

// SchemaTOON devolve a visão do path em TOON (1 linha por tool) — é o formato
// que o agente vê em get_tool_schema, ~85-90% mais barato que JSON. Com cache.
func (g *Gateway) SchemaTOON(path string) (string, error) {
	g.mu.RLock()
	if cached, ok := g.toonCache[path]; ok {
		g.mu.RUnlock()
		return cached, nil
	}
	g.mu.RUnlock()

	node, level, err := g.reg.Lookup(path)
	if err != nil {
		return "", err
	}
	var out string
	switch level {
	case "root":
		out = "# categorias (chame get_tool_schema no path p/ descer)\n" + toon.EncodeCategories(g.reg)
	case "category":
		out = "# serviços de " + path + "\n" + toon.EncodeCategory(path, node.(registry.Category))
	case "service":
		out = "# funções de " + path + " (path|desc|param:tipo:req[:default]|r:campos[|risk])\n" + toon.EncodeService(path, node.(registry.Service))
	case "function":
		out = toon.EncodeFunction(path, node.(registry.Function))
	}

	g.mu.Lock()
	g.toonCache[path] = out
	g.mu.Unlock()
	return out, nil
}

// ExecuteTool valida os parâmetros, executa a função (mock) e aplica a
// filtragem de resposta (whitelist "r").
func (g *Gateway) ExecuteTool(ctx context.Context, path string, params map[string]any) (map[string]any, error) {
	node, level, err := g.reg.Lookup(path)
	if err != nil {
		return nil, err
	}
	if level != "function" {
		return nil, fmt.Errorf("execute_tool exige um path de função (categoria.serviço.função), recebi nível %q", level)
	}
	fn := node.(registry.Function)

	args, err := prepareArgs(fn, params)
	if err != nil {
		return nil, err
	}

	raw, err := mockExecute(path, args)
	if err != nil {
		return nil, err
	}

	// Filtragem de resposta: se a função declara "r", só esses campos passam.
	if len(fn.R) > 0 {
		filtered := make(map[string]any, len(fn.R))
		for _, k := range fn.R {
			if v, ok := raw[k]; ok {
				filtered[k] = v
			}
		}
		return filtered, nil
	}
	return raw, nil
}

// prepareArgs aplica defaults, checa obrigatórios e coage tipos básicos.
func prepareArgs(fn registry.Function, params map[string]any) (map[string]any, error) {
	out := map[string]any{}
	for name, compact := range fn.P {
		p := registry.ParseParam(name, compact)
		val, provided := params[name]
		if !provided || val == nil || val == "" {
			if p.Required {
				return nil, fmt.Errorf("parâmetro obrigatório ausente: %q (%s)", name, p.Desc)
			}
			if p.Default != "" {
				out[name] = coerce(p.Type, p.Default)
			}
			continue
		}
		out[name] = coerceAny(p.Type, val)
	}
	return out, nil
}

func coerce(typ, s string) any {
	if typ == "number" {
		if n, err := strconv.ParseFloat(s, 64); err == nil {
			return n
		}
	}
	return s
}

func coerceAny(typ string, v any) any {
	if typ == "number" {
		switch t := v.(type) {
		case string:
			if n, err := strconv.ParseFloat(t, 64); err == nil {
				return n
			}
		}
	}
	return v
}
