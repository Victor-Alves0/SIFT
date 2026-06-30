// Package toon implementa o codec TOON (Token-Optimized Object Notation) do
// mapa do SIFT: reduz cada ferramenta a UMA linha, cortando ~85-90% dos tokens
// de schema frente ao JSON Schema tradicional.
//
// Formato de uma função:
//
//	path|descrição|param:tipo:req[:default]|...|r:campo1,campo2[|risk]
//
// onde req = "n" (obrigatório) ou "o" (opcional). Exemplo:
//
//	google_workspace.gmail.read|Read emails from inbox|q:string:o:is:unread|m:number:o:10|r:id,subject,from,snippet,date
package toon

import (
	"sort"
	"strings"

	"sift/internal/registry"
)

// EncodeFunction serializa uma função em uma única linha TOON.
func EncodeFunction(path string, fn registry.Function) string {
	parts := []string{path, clean(fn.D)}

	names := make([]string, 0, len(fn.P))
	for n := range fn.P {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		p := registry.ParseParam(n, fn.P[n])
		req := "o"
		if p.Required {
			req = "n"
		}
		seg := n + ":" + p.Type + ":" + req
		if p.Default != "" {
			seg += ":" + p.Default
		}
		parts = append(parts, seg)
	}
	if len(fn.R) > 0 {
		parts = append(parts, "r:"+strings.Join(fn.R, ","))
	}
	if fn.Risk {
		parts = append(parts, "risk")
	}
	return strings.Join(parts, "|")
}

// EncodeService devolve uma linha TOON por função do serviço (ordenadas).
func EncodeService(svcPath string, svc registry.Service) string {
	names := make([]string, 0, len(svc.Fns))
	for n := range svc.Fns {
		names = append(names, n)
	}
	sort.Strings(names)
	lines := make([]string, 0, len(names))
	for _, n := range names {
		lines = append(lines, EncodeFunction(svcPath+"."+n, svc.Fns[n]))
	}
	return strings.Join(lines, "\n")
}

// EncodeCategory lista os serviços de uma categoria, um por linha: path|descrição.
func EncodeCategory(catPath string, cat registry.Category) string {
	names := make([]string, 0, len(cat.Services))
	for n := range cat.Services {
		names = append(names, n)
	}
	sort.Strings(names)
	lines := make([]string, 0, len(names))
	for _, n := range names {
		lines = append(lines, catPath+"."+n+"|"+clean(cat.Services[n].D))
	}
	return strings.Join(lines, "\n")
}

// EncodeCategories lista as categorias da raiz, uma por linha: nome|descrição.
func EncodeCategories(reg registry.Registry) string {
	names := make([]string, 0, len(reg))
	for n := range reg {
		names = append(names, n)
	}
	sort.Strings(names)
	lines := make([]string, 0, len(names))
	for _, n := range names {
		lines = append(lines, n+"|"+clean(reg[n].D))
	}
	return strings.Join(lines, "\n")
}

// EstimateTokens é uma aproximação offline de contagem de tokens (~4 chars/token).
// Serve para comparar formatos no benchmark, não para faturamento exato.
func EstimateTokens(s string) int {
	if s == "" {
		return 0
	}
	return (len(s) + 3) / 4
}

// clean remove os separadores do TOON do texto livre (descrições).
func clean(s string) string {
	s = strings.ReplaceAll(s, "|", "/")
	s = strings.ReplaceAll(s, "\n", " ")
	return strings.TrimSpace(s)
}
