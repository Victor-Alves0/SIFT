package gateway

import (
	"context"
	"strings"
	"testing"

	"sift/internal/registry"
)

// newTestGW monta um gateway sem embed client (os testes aqui não usam busca).
func newTestGW(t *testing.T) *Gateway {
	t.Helper()
	reg, err := registry.Load("../../data/registry.json")
	if err != nil {
		t.Fatalf("carregando registry: %v", err)
	}
	return New(reg, nil)
}

func TestResponseFiltering(t *testing.T) {
	gw := newTestGW(t)
	res, err := gw.ExecuteTool(context.Background(), "google_workspace.gmail.read", map[string]any{"m": 1})
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	// a whitelist "r" de gmail.read NÃO inclui "body" — deve ter sido filtrado.
	if _, leaked := res["body"]; leaked {
		t.Error("campo 'body' vazou apesar de não estar na whitelist 'r'")
	}
	for _, want := range []string{"id", "subject", "from", "snippet", "date"} {
		if _, ok := res[want]; !ok {
			t.Errorf("campo esperado ausente após filtragem: %q", want)
		}
	}
}

func TestRequiredParamMissing(t *testing.T) {
	gw := newTestGW(t)
	_, err := gw.ExecuteTool(context.Background(), "google_workspace.gmail.send", map[string]any{"subject": "oi"})
	if err == nil {
		t.Fatal("esperava erro por faltar parâmetro obrigatório 'to'")
	}
}

func TestExecuteRejectsNonFunctionPath(t *testing.T) {
	gw := newTestGW(t)
	if _, err := gw.ExecuteTool(context.Background(), "google_workspace.gmail", nil); err == nil {
		t.Fatal("execute_tool deveria recusar path de serviço")
	}
}

func TestSchemaTOON(t *testing.T) {
	gw := newTestGW(t)
	line, err := gw.SchemaTOON("google_workspace.gmail.send")
	if err != nil {
		t.Fatalf("SchemaTOON: %v", err)
	}
	if !strings.Contains(line, "risk") {
		t.Errorf("send deveria expor marcador 'risk': %q", line)
	}
	if !strings.Contains(line, "to:string:n") {
		t.Errorf("send deveria listar 'to' obrigatório: %q", line)
	}

	// cache: segunda chamada idêntica
	line2, _ := gw.SchemaTOON("google_workspace.gmail.send")
	if line != line2 {
		t.Error("SchemaTOON não foi estável entre chamadas (cache)")
	}
}

func TestGetToolSchemaRoot(t *testing.T) {
	gw := newTestGW(t)
	res, err := gw.GetToolSchema("")
	if err != nil {
		t.Fatalf("GetToolSchema(\"\"): %v", err)
	}
	m := res.(map[string]any)
	if m["level"] != "categories" {
		t.Errorf("nível raiz = %v, queria 'categories'", m["level"])
	}
}
