// Package agent roda o loop do agente: dá ao modelo APENAS as 3 meta-tools e
// despacha as chamadas para o gateway, até o modelo produzir a resposta final.
package agent

import (
	"context"
	"encoding/json"
	"fmt"

	"sift/internal/gateway"
	"sift/internal/openrouter"
)

const systemPrompt = `Você é um assistente que descobre ferramentas dinamicamente — você NUNCA recebe o catálogo inteiro.

Você tem APENAS 3 ferramentas:
1. search_tools(q)            — encontra ferramentas por linguagem natural; devolve caminhos (path) com score.
2. get_tool_schema(path)      — devolve o schema compacto de um nível: "" (categorias), "cat", "cat.serviço" ou "cat.serviço.função".
3. execute_tool(path, params) — executa a função (path completo "cat.serviço.função") com os parâmetros.

FORMATO TOON (retorno de get_tool_schema): uma linha por ferramenta, campos separados por "|":
  path|descrição|param:tipo:req[:default]|...|r:campos[|risk]
  - req = "n" (obrigatório) ou "o" (opcional); se houver default, vem depois.
  - "r:" lista os campos que a resposta vai conter.
  - "risk" marca ação de alto impacto (enviar, deletar) — confirme a intenção antes de executar.

REGRAS:
- SEMPRE comece com search_tools para qualquer tarefa.
- Navegue a hierarquia: categoria → serviço → função. Use get_tool_schema antes de executar para ver os parâmetros.
- Só chame execute_tool com um path de função completo e parâmetros válidos.
- Os resultados já vêm filtrados (apenas os campos relevantes). Não invente campos.
- Quando tiver a informação necessária, responda ao usuário em português, de forma curta e direta.`

// MaxSteps limita o número de rodadas de tool-calling por tarefa.
const MaxSteps = 12

// Agent costura modelo + gateway.
type Agent struct {
	llm     *openrouter.Client
	gw      *gateway.Gateway
	Verbose bool
}

// New cria o agente.
func New(llm *openrouter.Client, gw *gateway.Gateway, verbose bool) *Agent {
	return &Agent{llm: llm, gw: gw, Verbose: verbose}
}

// metaTools é o único conjunto de ferramentas exposto ao modelo.
func metaTools() []openrouter.Tool {
	strProp := func(desc string) map[string]any {
		return map[string]any{"type": "string", "description": desc}
	}
	return []openrouter.Tool{
		{Type: "function", Function: openrouter.FunctionDef{
			Name:        "search_tools",
			Description: "Descobre ferramentas por linguagem natural. Devolve caminhos (path) candidatos com score de relevância.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"q": strProp("a necessidade em linguagem natural, ex: 'ler último email'")},
				"required":   []string{"q"},
			},
		}},
		{Type: "function", Function: openrouter.FunctionDef{
			Name:        "get_tool_schema",
			Description: "Devolve o schema compacto de um nível da hierarquia. path vazio lista as categorias.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"path": strProp("ex: '', 'google_workspace', 'google_workspace.gmail' ou 'google_workspace.gmail.read'")},
				"required":   []string{"path"},
			},
		}},
		{Type: "function", Function: openrouter.FunctionDef{
			Name:        "execute_tool",
			Description: "Executa uma função (path completo categoria.serviço.função) e devolve o resultado já filtrado.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"path":   strProp("path completo da função, ex: 'google_workspace.gmail.read'"),
					"params": map[string]any{"type": "object", "description": "parâmetros da função conforme o schema"},
				},
				"required": []string{"path"},
			},
		}},
	}
}

// Run executa a tarefa do usuário e devolve a resposta final do agente.
func (a *Agent) Run(ctx context.Context, userMsg string) (string, error) {
	messages := []openrouter.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userMsg},
	}
	tools := metaTools()

	for step := 0; step < MaxSteps; step++ {
		msg, err := a.llm.Chat(ctx, messages, tools)
		if err != nil {
			return "", err
		}
		messages = append(messages, *msg)

		if len(msg.ToolCalls) == 0 {
			return msg.Content, nil
		}

		for _, call := range msg.ToolCalls {
			result := a.dispatch(ctx, call)
			if a.Verbose {
				fmt.Printf("  ↳ %s(%s)\n    = %s\n", call.Function.Name, truncate(call.Function.Arguments, 120), truncate(result, 240))
			}
			messages = append(messages, openrouter.Message{
				Role:       "tool",
				ToolCallID: call.ID,
				Name:       call.Function.Name,
				Content:    result,
			})
		}
	}
	return "", fmt.Errorf("limite de %d passos atingido sem resposta final", MaxSteps)
}

// dispatch executa uma meta-tool e devolve o resultado serializado em JSON.
func (a *Agent) dispatch(ctx context.Context, call openrouter.ToolCall) string {
	switch call.Function.Name {
	case "search_tools":
		var args struct {
			Q    string `json:"q"`
			TopK int    `json:"top_k"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return errJSON(err)
		}
		res, err := a.gw.SearchTools(ctx, args.Q, args.TopK)
		if err != nil {
			return errJSON(err)
		}
		return toJSON(res)

	case "get_tool_schema":
		var args struct {
			Path string `json:"path"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return errJSON(err)
		}
		// Formato TOON (1 linha por tool) — devolvido como texto cru ao agente.
		res, err := a.gw.SchemaTOON(args.Path)
		if err != nil {
			return errJSON(err)
		}
		return res

	case "execute_tool":
		var args struct {
			Path   string         `json:"path"`
			Params map[string]any `json:"params"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return errJSON(err)
		}
		res, err := a.gw.ExecuteTool(ctx, args.Path, args.Params)
		if err != nil {
			return errJSON(err)
		}
		return toJSON(res)
	}
	return errJSON(fmt.Errorf("meta-tool desconhecida: %s", call.Function.Name))
}

func toJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return errJSON(err)
	}
	return string(b)
}

func errJSON(err error) string {
	b, _ := json.Marshal(map[string]string{"error": err.Error()})
	return string(b)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
