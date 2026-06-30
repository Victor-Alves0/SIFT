package registry

import "testing"

func TestParseParam(t *testing.T) {
	opt := ParseParam("m", "number:o:10:max results")
	if opt.Type != "number" || opt.Required || opt.Default != "10" || opt.Desc != "max results" {
		t.Fatalf("opcional mal parseado: %+v", opt)
	}
	req := ParseParam("q", "string:n::search query")
	if req.Type != "string" || !req.Required || req.Default != "" || req.Desc != "search query" {
		t.Fatalf("obrigatório mal parseado: %+v", req)
	}
}

func TestLookupLevels(t *testing.T) {
	reg := load(t)
	cases := map[string]string{
		"":                                "root",
		"google_workspace":                "category",
		"google_workspace.gmail":          "service",
		"google_workspace.gmail.read":     "function",
	}
	for path, want := range cases {
		_, level, err := reg.Lookup(path)
		if err != nil {
			t.Fatalf("Lookup(%q) erro: %v", path, err)
		}
		if level != want {
			t.Fatalf("Lookup(%q) nível = %q, queria %q", path, level, want)
		}
	}
	if _, _, err := reg.Lookup("nao.existe"); err == nil {
		t.Fatal("esperava erro para path inexistente")
	}
}

func TestRiskyPaths(t *testing.T) {
	reg := load(t)
	risky := reg.RiskyPaths()
	if !risky["google_workspace.gmail.send"] {
		t.Error("gmail.send deveria ser risky")
	}
	if !risky["google_workspace.drive.delete"] {
		t.Error("drive.delete deveria ser risky")
	}
	if risky["google_workspace.gmail.read"] {
		t.Error("gmail.read NÃO deveria ser risky")
	}
}

func load(t *testing.T) Registry {
	t.Helper()
	reg, err := Load("../../data/registry.json")
	if err != nil {
		t.Fatalf("carregando registry: %v", err)
	}
	return reg
}
