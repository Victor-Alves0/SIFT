// Package bench implementa o harness de avaliação do SIFT (ramo Benchmarks do
// mapa). Foca nas FILTER-LEVEL METRICS — determinísticas e sem custo de LLM:
// medem se a camada de descoberta expõe a ferramenta certa antes de o agente
// agir. Também compara o custo em tokens do schema TOON vs JSON.
//
// (As downstream agent metrics — que exigem rodar o LLM — ficam para um modo
// opt-in separado, para não gastar tokens em CI.)
package bench

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"sift/internal/gateway"
	"sift/internal/registry"
	"sift/internal/toon"
)

// Task é um caso rotulado: a query do usuário e o path "ouro" esperado.
type Task struct {
	Query      string `json:"query"`
	Gold       string `json:"gold"`
	NeedsRisky bool   `json:"needs_risky"`
}

// LoadTasks lê as tasks de um arquivo JSON.
func LoadTasks(path string) ([]Task, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tasks []Task
	if err := json.Unmarshal(data, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

// TaskResult é o detalhe por task.
type TaskResult struct {
	Query      string
	Gold       string
	GoldRank   int // posição 1-based no menu; 0 = ausente
	Visible    int
	RiskyShown []string
}

// FilterMetrics agrega as métricas de filtro sobre o conjunto de tasks.
type FilterMetrics struct {
	Tasks                 int
	TopK                  int
	GoldExposure          float64 // % de tasks com o gold visível no top-K
	NoVisibleToolRate     float64 // % de tasks SEM o gold no menu
	AvgVisibleTools       float64 // tamanho médio do menu
	AvgExtraTools         float64 // ferramentas além do gold (ruído)
	MRR                   float64 // mean reciprocal rank do gold
	RiskyExposureRate     float64 // % de tasks com alguma tool de risco no menu
	UnauthorizedRiskyRate float64 // % com tool de risco exposta SEM a task precisar
	PerTask               []TaskResult
}

// RunFilter roda a descoberta para cada task e calcula as métricas de filtro.
func RunFilter(ctx context.Context, gw *gateway.Gateway, tasks []Task, topK int) (FilterMetrics, error) {
	risky := gw.Registry().RiskyPaths()
	m := FilterMetrics{Tasks: len(tasks), TopK: topK}

	var goldHits, sumVisible, sumExtra, riskyTasks, unauthorizedTasks int
	var sumRR float64

	for _, t := range tasks {
		res, err := gw.SearchTools(ctx, t.Query, topK)
		if err != nil {
			return m, err
		}

		tr := TaskResult{Query: t.Query, Gold: t.Gold, Visible: len(res)}
		for i, r := range res {
			if r.Path == t.Gold {
				tr.GoldRank = i + 1
			}
			if risky[r.Path] {
				tr.RiskyShown = append(tr.RiskyShown, r.Path)
			}
		}

		sumVisible += tr.Visible
		if tr.GoldRank > 0 {
			goldHits++
			sumRR += 1.0 / float64(tr.GoldRank)
			sumExtra += tr.Visible - 1
		} else {
			sumExtra += tr.Visible
		}
		if len(tr.RiskyShown) > 0 {
			riskyTasks++
			if !t.NeedsRisky {
				unauthorizedTasks++
			}
		}
		m.PerTask = append(m.PerTask, tr)
	}

	n := float64(len(tasks))
	if n > 0 {
		m.GoldExposure = float64(goldHits) / n
		m.NoVisibleToolRate = float64(len(tasks)-goldHits) / n
		m.AvgVisibleTools = float64(sumVisible) / n
		m.AvgExtraTools = float64(sumExtra) / n
		m.MRR = sumRR / n
		m.RiskyExposureRate = float64(riskyTasks) / n
		m.UnauthorizedRiskyRate = float64(unauthorizedTasks) / n
	}
	return m, nil
}

// TokenReport compara o custo em tokens dos schemas (TOON vs JSON).
type TokenReport struct {
	Functions          int
	ToonTokens         int
	CompactJSONTokens  int // o formato compacto do nosso registry
	VerboseJSONTokens  int // JSON Schema estilo OpenAPI (baseline da indústria)
	ReductionVsCompact float64
	ReductionVsVerbose float64
}

// RunTokens computa a economia de tokens do TOON sobre todas as funções.
func RunTokens(reg registry.Registry) TokenReport {
	var rep TokenReport
	for cn, cat := range reg {
		for sn, svc := range cat.Services {
			for fn, f := range svc.Fns {
				path := cn + "." + sn + "." + fn
				rep.Functions++
				rep.ToonTokens += toon.EstimateTokens(toon.EncodeFunction(path, f))

				compact, _ := json.Marshal(f)
				rep.CompactJSONTokens += toon.EstimateTokens(string(compact))

				rep.VerboseJSONTokens += toon.EstimateTokens(verboseJSONSchema(path, f))
			}
		}
	}
	if rep.CompactJSONTokens > 0 {
		rep.ReductionVsCompact = 1 - float64(rep.ToonTokens)/float64(rep.CompactJSONTokens)
	}
	if rep.VerboseJSONTokens > 0 {
		rep.ReductionVsVerbose = 1 - float64(rep.ToonTokens)/float64(rep.VerboseJSONTokens)
	}
	return rep
}

// verboseJSONSchema reconstrói o JSON Schema estilo OpenAPI/Function-Calling que
// um provedor tradicional enviaria — o baseline contra o qual o TOON economiza.
func verboseJSONSchema(path string, fn registry.Function) string {
	props := map[string]any{}
	var required []string
	for name, compact := range fn.P {
		p := registry.ParseParam(name, compact)
		jt := "string"
		if p.Type == "number" {
			jt = "number"
		}
		prop := map[string]any{"type": jt, "description": p.Desc}
		if p.Default != "" {
			prop["default"] = p.Default
		}
		props[name] = prop
		if p.Required {
			required = append(required, name)
		}
	}
	schema := map[string]any{
		"name":        path,
		"description": fn.D,
		"parameters": map[string]any{
			"type":       "object",
			"properties": props,
			"required":   required,
		},
	}
	b, _ := json.MarshalIndent(schema, "", "  ")
	return string(b)
}

// Format imprime um relatório legível das métricas de filtro.
func (m FilterMetrics) Format() string {
	s := fmt.Sprintf(`FILTER-LEVEL METRICS  (tasks=%d, top_k=%d)
  Gold next-tool exposure : %5.1f%%   (alvo: alto)
  No-visible-tool rate    : %5.1f%%   (alvo: 0%%)
  Average visible tools   : %5.2f    (alvo: baixo)
  Extra tools exposed     : %5.2f    (ruído por task)
  MRR (rank do gold)      : %5.3f    (alvo: ->1.0)
  Risky-tool exposure     : %5.1f%%
  Unauthorized risky      : %5.1f%%   (alvo: 0%%)
`,
		m.Tasks, m.TopK,
		m.GoldExposure*100, m.NoVisibleToolRate*100,
		m.AvgVisibleTools, m.AvgExtraTools, m.MRR,
		m.RiskyExposureRate*100, m.UnauthorizedRiskyRate*100)
	return s
}

// Format imprime o relatório de tokens.
func (r TokenReport) Format() string {
	avgToon := 0
	avgVerbose := 0
	if r.Functions > 0 {
		avgToon = r.ToonTokens / r.Functions
		avgVerbose = r.VerboseJSONTokens / r.Functions
	}
	return fmt.Sprintf(`TOKEN COST  (%d funções, estimativa ~4 chars/token)
  TOON total           : %5d tokens  (~%d/tool)
  JSON compacto total  : %5d tokens
  JSON Schema verboso  : %5d tokens  (~%d/tool, baseline OpenAPI)
  Redução TOON vs compacto : %5.1f%%
  Redução TOON vs verboso  : %5.1f%%
`,
		r.Functions,
		r.ToonTokens, avgToon,
		r.CompactJSONTokens,
		r.VerboseJSONTokens, avgVerbose,
		r.ReductionVsCompact*100, r.ReductionVsVerbose*100)
}
