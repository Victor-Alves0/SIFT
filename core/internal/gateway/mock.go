package gateway

import (
	"fmt"
	"strings"
	"time"
)

// mockExecute simula a chamada real da API por trás de cada função. Numa
// integração real, isto vira o adaptador HTTP/SDK do provedor — a forma do
// retorno (chaves) é o que importa, pois a filtragem usa a whitelist "r".
//
// Os dados são fixos/derivados dos argumentos só para o MVP funcionar
// ponta-a-ponta sem credenciais externas.
func mockExecute(path string, args map[string]any) (map[string]any, error) {
	now := time.Now().UTC().Format(time.RFC3339)

	switch path {
	case "google_workspace.gmail.read", "microsoft.outlook.read":
		return map[string]any{
			"id":      "msg_1001",
			"subject": "Reunião amanhã às 10h",
			"from":    "joao@empresa.com",
			"snippet": "Confirmando nossa reunião de amanhã. Trago o relatório.",
			"preview": "Confirmando nossa reunião de amanhã. Trago o relatório.",
			"body":    "Olá! Confirmando nossa reunião de amanhã às 10h. Abraço, João.",
			"date":    now,
		}, nil

	case "google_workspace.gmail.query":
		return map[string]any{
			"id":      "msg_1001",
			"subject": "Reunião amanhã às 10h",
			"from":    "joao@empresa.com",
			"date":    now,
		}, nil

	case "google_workspace.gmail.send":
		to, _ := args["to"].(string)
		return map[string]any{
			"id":     "sent_2002",
			"status": fmt.Sprintf("enviado para %s", to),
		}, nil

	case "google_workspace.drive.list":
		return map[string]any{
			"id":           "file_3003",
			"name":         "relatorio-q2.pdf",
			"mimeType":     "application/pdf",
			"modifiedTime": now,
		}, nil

	case "google_workspace.drive.read":
		return map[string]any{
			"id":      args["id"],
			"name":    "relatorio-q2.pdf",
			"content": "Resumo executivo do segundo trimestre...",
		}, nil

	case "google_workspace.drive.delete":
		return map[string]any{
			"id":     args["id"],
			"status": "deleted",
		}, nil

	case "local.filesystem.list":
		return map[string]any{
			"name":  "README.md",
			"size":  float64(1280),
			"isDir": false,
		}, nil

	case "local.filesystem.read":
		p, _ := args["path"].(string)
		return map[string]any{
			"path":    p,
			"content": "conteúdo simulado de " + p,
		}, nil

	case "web.search.run":
		q, _ := args["q"].(string)
		return map[string]any{
			"title":   "Resultado para: " + q,
			"url":     "https://example.com/" + strings.ReplaceAll(q, " ", "-"),
			"snippet": "Trecho relevante sobre " + q + " ...",
		}, nil
	}

	return nil, fmt.Errorf("executor mock não implementado para %q", path)
}
