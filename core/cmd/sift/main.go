// Comando sift — CLI e servidor HTTP do core.
//
// Uso:
//
//	sift chat "<mensagem>"   roda o agente ponta-a-ponta (precisa OPENROUTER_API)
//	sift search "<query>"    só a descoberta semântica (não precisa do LLM)
//	sift schema "<path>"     inspeciona um nível da hierarquia
//	sift serve               sobe a API HTTP (acoplável a outros sistemas)
//	sift health              verifica o sidecar de embeddings
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"sift/internal/agent"
	"sift/internal/bench"
	"sift/internal/embed"
	"sift/internal/gateway"
	"sift/internal/openrouter"
	"sift/internal/registry"
)

func main() {
	loadDotEnv()

	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	cmd := os.Args[1]
	ctx := context.Background()

	switch cmd {
	case "chat":
		mustArg(2, "mensagem")
		runChat(ctx, strings.Join(os.Args[2:], " "))
	case "search":
		mustArg(2, "query")
		runSearch(ctx, strings.Join(os.Args[2:], " "))
	case "schema":
		path := ""
		if len(os.Args) > 2 {
			path = os.Args[2]
		}
		runSchema(path)
	case "serve":
		runServe(ctx)
	case "bench":
		runBench(ctx)
	case "health":
		runHealth(ctx)
	default:
		fmt.Printf("comando desconhecido: %q\n\n", cmd)
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Print(`SIFT — Search · Inspect · Filter · Trigger

  sift chat "<mensagem>"   roda o agente ponta-a-ponta (precisa OPENROUTER_API)
  sift search "<query>"    só a descoberta semântica (não precisa do LLM)
  sift schema "<path>"     inspeciona a hierarquia ("" lista categorias)
  sift serve               sobe a API HTTP em :8080
  sift bench               roda as filter-level metrics + custo de tokens (sem LLM)
  sift health              verifica o sidecar de embeddings
`)
}

// --- construção das dependências ---

func buildGateway(ctx context.Context) (*gateway.Gateway, error) {
	embedURL := envOr("SIFT_EMBED_URL", "http://127.0.0.1:8088")
	ec := embed.New(embedURL)
	if err := ec.Health(ctx); err != nil {
		return nil, fmt.Errorf("%w\n  → suba o sidecar:  cd embed-svc && python server.py", err)
	}

	reg, err := registry.Load(registryPath())
	if err != nil {
		return nil, err
	}

	gw := gateway.New(reg, ec)
	fmt.Println("• montando índice semântico...")
	if err := gw.BuildIndex(ctx); err != nil {
		return nil, err
	}
	return gw, nil
}

func runChat(ctx context.Context, msg string) {
	apiKey := os.Getenv("OPENROUTER_API")
	if apiKey == "" {
		fatal(fmt.Errorf("OPENROUTER_API não definida (configure no .env)"))
	}
	gw, err := buildGateway(ctx)
	if err != nil {
		fatal(err)
	}
	model := envOr("SIFT_MODEL", "anthropic/claude-haiku-4.5")
	llm := openrouter.New(apiKey, model)
	ag := agent.New(llm, gw, true)

	fmt.Printf("• modelo: %s\n• usuário: %s\n\n", model, msg)
	answer, err := ag.Run(ctx, msg)
	if err != nil {
		fatal(err)
	}
	fmt.Printf("\n🤖 %s\n", answer)
}

func runSearch(ctx context.Context, q string) {
	gw, err := buildGateway(ctx)
	if err != nil {
		fatal(err)
	}
	res, err := gw.SearchTools(ctx, q, 5)
	if err != nil {
		fatal(err)
	}
	fmt.Printf("\nresultados para %q:\n", q)
	for _, r := range res {
		fmt.Printf("  %.3f  [%s]  %s — %s\n", r.Score, r.Kind, r.Path, r.D)
	}
}

func runSchema(path string) {
	reg, err := registry.Load(registryPath())
	if err != nil {
		fatal(err)
	}
	schema, err := reg.Schema(path)
	if err != nil {
		fatal(err)
	}
	b, _ := json.MarshalIndent(schema, "", "  ")
	fmt.Println(string(b))
}

func runBench(ctx context.Context) {
	topK := 5
	if len(os.Args) > 2 {
		if k, err := strconv.Atoi(os.Args[2]); err == nil && k > 0 {
			topK = k
		}
	}

	// Custo de tokens (TOON vs JSON) — não precisa do sidecar.
	reg, err := registry.Load(registryPath())
	if err != nil {
		fatal(err)
	}
	fmt.Println(bench.RunTokens(reg).Format())

	// Filter-level metrics — precisa do índice semântico.
	gw, err := buildGateway(ctx)
	if err != nil {
		fatal(err)
	}
	tasks, err := bench.LoadTasks(benchTasksPath())
	if err != nil {
		fatal(err)
	}
	m, err := bench.RunFilter(ctx, gw, tasks, topK)
	if err != nil {
		fatal(err)
	}
	fmt.Println()
	fmt.Print(m.Format())

	// destaca falhas de exposição (gold ausente do menu)
	for _, tr := range m.PerTask {
		if tr.GoldRank == 0 {
			fmt.Printf("  ⚠ gold ausente: %q esperava %s\n", tr.Query, tr.Gold)
		}
	}
}

func runHealth(ctx context.Context) {
	ec := embed.New(envOr("SIFT_EMBED_URL", "http://127.0.0.1:8088"))
	if err := ec.Health(ctx); err != nil {
		fatal(err)
	}
	fmt.Println("✓ sidecar de embeddings OK")
}

// --- servidor HTTP (acoplável) ---

func runServe(ctx context.Context) {
	gw, err := buildGateway(ctx)
	if err != nil {
		fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			Q    string `json:"q"`
			TopK int    `json:"top_k"`
		}
		if !decode(w, r, &in) {
			return
		}
		res, err := gw.SearchTools(r.Context(), in.Q, in.TopK)
		respond(w, res, err)
	})
	mux.HandleFunc("/schema", func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			Path string `json:"path"`
		}
		if !decode(w, r, &in) {
			return
		}
		res, err := gw.GetToolSchema(in.Path)
		respond(w, res, err)
	})
	mux.HandleFunc("/execute", func(w http.ResponseWriter, r *http.Request) {
		var in struct {
			Path   string         `json:"path"`
			Params map[string]any `json:"params"`
		}
		if !decode(w, r, &in) {
			return
		}
		res, err := gw.ExecuteTool(r.Context(), in.Path, in.Params)
		respond(w, res, err)
	})
	mux.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		apiKey := os.Getenv("OPENROUTER_API")
		if apiKey == "" {
			respond(w, nil, fmt.Errorf("OPENROUTER_API não definida"))
			return
		}
		var in struct {
			Message string `json:"message"`
		}
		if !decode(w, r, &in) {
			return
		}
		llm := openrouter.New(apiKey, envOr("SIFT_MODEL", "anthropic/claude-haiku-4.5"))
		ag := agent.New(llm, gw, false)
		answer, err := ag.Run(r.Context(), in.Message)
		respond(w, map[string]string{"answer": answer}, err)
	})

	addr := envOr("SIFT_ADDR", ":8080")
	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	fmt.Printf("• SIFT API ouvindo em %s\n", addr)
	if err := srv.ListenAndServe(); err != nil {
		fatal(err)
	}
}

// --- helpers ---

func registryPath() string {
	if p := os.Getenv("SIFT_REGISTRY"); p != "" {
		return p
	}
	// tenta cwd/data e, depois, ao lado do executável
	if _, err := os.Stat("data/registry.json"); err == nil {
		return "data/registry.json"
	}
	if exe, err := os.Executable(); err == nil {
		alt := filepath.Join(filepath.Dir(exe), "data", "registry.json")
		if _, err := os.Stat(alt); err == nil {
			return alt
		}
	}
	return "data/registry.json"
}

func benchTasksPath() string {
	if p := os.Getenv("SIFT_BENCH_TASKS"); p != "" {
		return p
	}
	if _, err := os.Stat("data/bench_tasks.json"); err == nil {
		return "data/bench_tasks.json"
	}
	if exe, err := os.Executable(); err == nil {
		alt := filepath.Join(filepath.Dir(exe), "data", "bench_tasks.json")
		if _, err := os.Stat(alt); err == nil {
			return alt
		}
	}
	return "data/bench_tasks.json"
}

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		respond(w, nil, fmt.Errorf("JSON inválido: %w", err))
		return false
	}
	return true
}

func respond(w http.ResponseWriter, v any, err error) {
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		writeJSON(w, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, v)
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	_ = json.NewEncoder(w).Encode(v)
}

func mustArg(idx int, name string) {
	if len(os.Args) <= idx {
		fatal(fmt.Errorf("faltou argumento: %s", name))
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "erro: %v\n", err)
	os.Exit(1)
}

// loadDotEnv carrega pares KEY=VALUE de um .env (raiz do projeto ou cwd) para o
// ambiente, sem sobrescrever variáveis já definidas. Parser simples, sem deps.
func loadDotEnv() {
	for _, p := range []string{".env", filepath.Join("..", ".env")} {
		f, err := os.Open(p)
		if err != nil {
			continue
		}
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" || strings.HasPrefix(line, "#") {
				continue
			}
			k, v, ok := strings.Cut(line, "=")
			if !ok {
				continue
			}
			k = strings.TrimSpace(k)
			v = strings.Trim(strings.TrimSpace(v), `"'`)
			if _, exists := os.LookupEnv(k); !exists {
				os.Setenv(k, v)
			}
		}
		f.Close()
		return
	}
}
