// Package openrouter é um client mínimo da API de chat completions do
// OpenRouter, com suporte a tool calling (function calling).
package openrouter

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const endpoint = "https://openrouter.ai/api/v1/chat/completions"

// Client chama o OpenRouter.
type Client struct {
	APIKey string
	Model  string
	HTTP   *http.Client
}

// New cria o client.
func New(apiKey, model string) *Client {
	return &Client{
		APIKey: apiKey,
		Model:  model,
		HTTP:   &http.Client{Timeout: 120 * time.Second},
	}
}

// --- Tipos da API (subconjunto que usamos) ---

// Message é uma mensagem do diálogo.
type Message struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	Name       string     `json:"name,omitempty"`
}

// ToolCall é uma chamada de ferramenta pedida pelo modelo.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall traz nome e argumentos (JSON string) da função.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool é a definição de uma ferramenta exposta ao modelo.
type Tool struct {
	Type     string       `json:"type"`
	Function FunctionDef  `json:"function"`
}

// FunctionDef descreve uma função no formato JSON-Schema esperado pela API.
type FunctionDef struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Parameters  any    `json:"parameters"`
}

type chatRequest struct {
	Model      string    `json:"model"`
	Messages   []Message `json:"messages"`
	Tools      []Tool    `json:"tools,omitempty"`
	ToolChoice string    `json:"tool_choice,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Chat faz uma rodada de completion e devolve a mensagem do assistente.
func (c *Client) Chat(ctx context.Context, messages []Message, tools []Tool) (*Message, error) {
	reqBody := chatRequest{
		Model:    c.Model,
		Messages: messages,
		Tools:    tools,
	}
	body, _ := json.Marshal(reqBody)

	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	req.Header.Set("Content-Type", "application/json")
	// Headers opcionais de atribuição do OpenRouter.
	req.Header.Set("HTTP-Referer", "https://github.com/sift")
	req.Header.Set("X-Title", "SIFT")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("chamando OpenRouter: %w", err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OpenRouter status %d: %s", resp.StatusCode, string(raw))
	}

	var out chatResponse
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decodificando resposta: %w (corpo: %s)", err, string(raw))
	}
	if out.Error != nil {
		return nil, fmt.Errorf("OpenRouter: %s", out.Error.Message)
	}
	if len(out.Choices) == 0 {
		return nil, fmt.Errorf("OpenRouter não retornou choices")
	}
	return &out.Choices[0].Message, nil
}
