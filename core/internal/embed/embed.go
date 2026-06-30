// Package embed é o client HTTP do core para o sidecar de embeddings (Python).
//
// Contrato (qualquer serviço que o respeite pode substituir o sidecar):
//
//	GET  /health             -> {"status":"ok","model":"...","dim":N}
//	POST /embed {"texts":[]}  -> {"vectors":[[...]],"dim":N}
package embed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"time"
)

// Client fala com o sidecar de embeddings.
type Client struct {
	BaseURL string
	HTTP    *http.Client
}

// New cria um client com timeout razoável.
func New(baseURL string) *Client {
	return &Client{
		BaseURL: baseURL,
		HTTP:    &http.Client{Timeout: 60 * time.Second},
	}
}

type embedReq struct {
	Texts []string `json:"texts"`
}

type embedResp struct {
	Vectors [][]float32 `json:"vectors"`
	Dim     int         `json:"dim"`
}

// Health verifica se o sidecar está no ar (e o modelo carregado).
func (c *Client) Health(ctx context.Context) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/health", nil)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("embed sidecar inacessível em %s: %w", c.BaseURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("embed sidecar retornou status %d", resp.StatusCode)
	}
	return nil
}

// Embed devolve um vetor por texto, na mesma ordem.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	body, _ := json.Marshal(embedReq{Texts: texts})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+"/embed", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("chamando /embed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("/embed retornou status %d", resp.StatusCode)
	}
	var out embedResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decodificando resposta de /embed: %w", err)
	}
	return out.Vectors, nil
}

// Cosine calcula a similaridade do cosseno entre dois vetores.
func Cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
