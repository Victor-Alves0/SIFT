package toon

import (
	"strings"
	"testing"

	"sift/internal/registry"
)

func TestEncodeFunction(t *testing.T) {
	fn := registry.Function{
		D: "Read emails from inbox",
		P: map[string]string{
			"q": "string:o:is:unread:Gmail search query",
			"m": "number:o:10:max results",
		},
		R: []string{"id", "subject", "from"},
	}
	got := EncodeFunction("google_workspace.gmail.read", fn)

	// uma única linha
	if strings.Contains(got, "\n") {
		t.Fatalf("função deve caber em 1 linha: %q", got)
	}
	for _, want := range []string{
		"google_workspace.gmail.read",
		"Read emails from inbox",
		"m:number:o:10", // param com default
		"q:string:o",    // param opcional
		"r:id,subject,from",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("linha TOON não contém %q\n  linha: %s", want, got)
		}
	}
}

func TestEncodeRiskMarker(t *testing.T) {
	fn := registry.Function{D: "Send email", P: map[string]string{"to": "string:n::dest"}, Risk: true}
	got := EncodeFunction("g.gmail.send", fn)
	if !strings.HasSuffix(got, "|risk") {
		t.Fatalf("tool de risco deve terminar em |risk: %q", got)
	}
	if !strings.Contains(got, "to:string:n") {
		t.Fatalf("param obrigatório mal codificado: %q", got)
	}
}

func TestCleanRemovesPipe(t *testing.T) {
	fn := registry.Function{D: "a | b\nc"}
	got := EncodeFunction("x", fn)
	if strings.Count(got, "|") != 1 { // só o separador path|desc
		t.Fatalf("pipe da descrição não foi limpo: %q", got)
	}
}
